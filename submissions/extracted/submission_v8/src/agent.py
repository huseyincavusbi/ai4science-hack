#!/usr/bin/env python3
"""AI4Science agent — Anthropic SDK + task detection + code execution + fallback."""
from __future__ import annotations

import argparse, csv, json, math, os, re, shutil, subprocess, sys, tempfile, textwrap, time, zipfile
from pathlib import Path

MODEL = "bedrock-claude-opus"
MAX_TOKENS = 4096
MAX_RETRIES = 3

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--submission-dir", type=Path, default=Path.cwd())
    return p.parse_args()

# ── Task utils ───────────────────────────────────────────────────────────────

def copy_task(task_dir, ws_dir):
    if ws_dir.exists(): shutil.rmtree(ws_dir)
    shutil.copytree(task_dir, ws_dir)
    data = ws_dir / "data"
    if data.is_dir():
        for z in sorted(data.glob("*.zip")):
            if zipfile.is_zipfile(z):
                with zipfile.ZipFile(z) as zf: zf.extractall(ws_dir)

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)

def read_json(path):
    if not path.is_file(): return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    return d if isinstance(d, dict) else {}

def output_files(task_config):
    o = task_config.get("output")
    if not isinstance(o, dict): return ["predictions.csv"]
    if isinstance(o.get("files"), list) and o["files"]: return [str(x) for x in o["files"]]
    return [str(o.get("file", "predictions.csv"))]

def output_cols(task_config):
    o = task_config.get("output")
    if isinstance(o, dict) and isinstance(o.get("columns"), list):
        return [str(c) for c in o["columns"]]
    return []

# ── Task detection ───────────────────────────────────────────────────────────

def detect_type(task_config, task_dir):
    pt = str(task_config.get("problem_type", ""))
    if pt: return pt
    md = task_dir / "task.md"
    txt = md.read_text(encoding="utf-8").lower() if md.is_file() else ""
    if any(w in txt for w in ("treatment effect","causal","pehe","counterfactual")): return "causal_effect_estimation"
    if any(w in txt for w in ("domain generalization","domain shift","distribution shift")): return "binary_classification"
    if any(w in txt for w in ("backdoor","artifact","trigger","spurious")): return "classification"
    if any(w in txt for w in ("feature engineering","descriptor","features.csv")): return "feature_engineering"
    if "spearman" in txt or "correlation" in txt: return "regression"
    if "auprc" in txt or "auroc" in txt: return "binary_classification"
    return "unknown"

def detect_modality(task_dir, task_config):
    inf = task_config.get("input_files", {})
    if isinstance(inf, dict):
        for v in inf.values():
            if isinstance(v, str) and any(v.lower().endswith(e) for e in (".zip",".npz",".png",".jpg")): return "image"
    data = task_dir / "data"
    if data.is_dir():
        for f in data.iterdir():
            if f.suffix in (".zip",".npz") and "image" in f.name.lower(): return "image"
    tcsv = task_dir / "data" / "train.csv"
    if tcsv.is_file():
        cols, _ = read_csv(tcsv)
        for c in cols:
            if any(k in c.lower() for k in ("image_path","image_index")): return "image"
    return "tabular"

# ── System prompt ────────────────────────────────────────────────────────────

STRATEGIES = {
    "binary_classification": """
### Binary Classification
- Check class balance. Use class_weight='balanced' if imbalanced.
- Images: pretrained ResNet-18 (torchvision), heavy augmentation (ColorJitter, RandomFlip, RandomRotation).
- Tabular: LogisticRegression, RandomForest, GradientBoosting. StandardScaler for linear models.
- Split 80/20 stratified. Report validation accuracy.
- Output: binary 0/1 or float [0,1] as task specifies.
""",
    "classification": """
### Multi-class Classification
- Count classes. Check imbalance.
- Images 28x28: CNN (Conv2d 32→64→128, BatchNorm, Dropout(0.3), Linear). Adam(1e-3), CrossEntropyLoss.
- Backdoor/artifact: train clean model first. Inspect images for trigger patterns (per-pixel variance).
  Apply iterative pruning. Use RandAugment. Consider 3-model ensemble.
- Output: integer labels 0-9.
""",
    "causal_effect_estimation": """
### Causal Effect Estimation — NOT standard supervised learning!
- Step 1: Check covariate balance. Standardized mean diffs T=1 vs T=0.
- Step 2: Fit propensity score (GradientBoostingClassifier, 200 trees, max_depth=3) → clip to [0.05, 0.95].
- Step 3: T-learner — GradientBoostingRegressor(300 trees, max_depth=4, lr=0.05) separately on T=1 and T=0. Effect = mu1 - mu0.
- Step 4: Validate on held-out train split (20%). Report ATE and per-unit stats.
- Step 4: Optionally add X-learner (propensity-weighted) and Doubly Robust (DR) estimators.
  If ensembling: weight DR=0.5, T=0.25, X=0.25 (DR most robust).
- Step 5: Evaluate on held-out train split (20%). Report ATE.
- Step 6: Write predictions.csv with id,prediction (float per-unit treatment effect).
- If ensembling T/X/DR-learners: weight DR most heavily (DR=0.5, T=0.25, X=0.25)
  as Doubly Robust is most robust to misspecification.
- random_state=42 for reproducibility.
""",
    "regression": """
### Regression
- Check target distribution. Log-transform if highly skewed.
- RidgeCV, RandomForestRegressor, GradientBoostingRegressor.
- Cross-validate (5-fold). Pick best.
- Output: float predictions.
""",
    "feature_engineering": """
### Feature Engineering — create features, scorer fits model on them
- Read input columns. Create derived features (ratios, interactions, polynomials).
- All features numeric, finite. Drop constant columns.
- Train + test files: identical column names and order. 'id' first column.
""",
    "unknown": """
### General Strategy
- Read task.md. Understand input, target, metric.
- Explore data: shapes, types, distributions.
- Images → PyTorch. Tabular → sklearn. Validate on train split.
- Output exact format specified.
""",
}

MODALITY_GUIDE = {
    "image": """
### Image Data
- .zip: extract with zipfile. .npz: np.load, uint8 (N,H,W). Normalize [0,1] or ImageNet stats.
- torch.utils.data.DataLoader, batch=64, num_workers=2.
- GPU available (NVIDIA L4). Use device='cuda' if torch.cuda.is_available().
""",
    "tabular": """
### Tabular Data
- pandas read_csv. Check dtypes, nulls, summary stats.
- sklearn Pipeline for preprocessing. Handle NaN/Inf.
""",
}

def build_system_prompt(problem_type, modality):
    s = STRATEGIES.get(problem_type, STRATEGIES["unknown"])
    m = MODALITY_GUIDE.get(modality, MODALITY_GUIDE["tabular"])
    return f"""You are an autonomous ML research agent. Solve the task in a single Python script.

## Environment
Python 3.11: numpy, pandas, scikit-learn, torch, torchvision, causalml, PIL.
GPU: NVIDIA L4. Time limit: ~50 minutes.

## REQUIRED WORKFLOW

### STEP 0: Data Audit (MANDATORY)
- Read task.json + task.md. Check every file it lists actually exists.
- For CSVs: print shape, columns, dtypes, null counts, label distribution.
- For images: check array shapes, value range, count matches CSV rows.
- Flag any data quality issues.

### STEP 1: Understand the Problem
- Identify: input features, target variable, evaluation metric, output format.
- Understand the non-trivial aspect (domain shift, spurious correlation, confounded treatment, etc).

### STEP 2: Implement
- Write a COMPLETE Python script in a ```python``` code block.
- Include: data loading, preprocessing, train/val split, model training, metrics, prediction generation.
- Print progress and key metrics. Handle edge cases (NaN, empty files, single-class).

### STEP 3: Output
- Write predictions to the output path specified in the task message.
- Format must match sample_submission.csv or task.json output columns EXACTLY.

{s}

{m}

## Code Requirements
- Always use find_file() to locate data files, never hardcode "data/train.csv".
  The working directory on Codabench is NOT the task directory. Search:
  ["data/train.csv", "/app/input_data/data/train.csv", glob("**/train.csv")].
- Your script must be SELF-CONTAINED (all imports, all logic).
- File paths: search for data files using os.path.exists or glob. Try multiple paths.
  Codabench mounts task data at /app/input_data/. Your cwd varies.
  Example: glob.glob('**/train.csv', recursive=True) or check ['data/train.csv', '/app/input_data/data/train.csv'].
- Use if __name__ == '__main__' guard.
- Write predictions.csv to the path specified in the message below.
- Print "DONE" at the end so execution is recognized as successful.
"""

# ── Agent ────────────────────────────────────────────────────────────────────

def extract_code(text):
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None

def run_code(code, task_dir, timeout=2400):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(code)
        tmp = f.name
    try:
        r = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=timeout, cwd=str(task_dir))
        out = r.stdout + ("\nSTDERR:\n" + r.stderr if r.stderr else "")
        return r.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT after 40 min"
    except Exception as e:
        return False, f"ERROR: {e}"
    finally:
        try: os.unlink(tmp)
        except: pass

def write_fallback(ws_task_dir, output_path, task_config):
    tcsv = ws_task_dir / "data" / "test.csv"
    if not tcsv.is_file(): return
    _, test_rows = read_csv(tcsv)
    train_csv = ws_task_dir / "data" / "train.csv"
    _, train_rows = read_csv(train_csv) if train_csv.is_file() else ([], [])
    cols = output_cols(task_config) or ["id", "prediction"]
    sample = ws_task_dir / "sample_submission.csv"
    if sample.is_file():
        sc, _ = read_csv(sample)
        if sc: cols = sc
    val = 0.0
    target = str(task_config.get("target", {}).get("name", "label") if isinstance(task_config.get("target"), dict) else "label")
    if train_rows and target in train_rows[0]:
        vals = []
        for r in train_rows:
            try: v = float(r.get(target, 0))
            except: continue
            if math.isfinite(v): vals.append(v)
        if vals:
            pt = str(task_config.get("problem_type", ""))
            val = max(set(vals), key=vals.count) if "classification" in pt else sum(vals) / len(vals)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in test_rows:
            rid = row.get("id", "").strip()
            if rid:
                out = {cols[0]: rid}
                for c in cols[1:]: out[c] = str(val)
                w.writerow(out)

# ── Validate ─────────────────────────────────────────────────────────────────

def is_finite(v):
    try: return math.isfinite(float(v))
    except: return False

def validate_output(pred_csv, expected_cols, expected_ids, numeric_cols):
    if not pred_csv.is_file():
        raise ValueError(f"{pred_csv} not found")
    fns, rows = read_csv(pred_csv)
    if fns != expected_cols:
        raise ValueError(f"columns {fns} != expected {expected_cols}")
    preds = {}
    for row in rows:
        clean = {}
        for col in expected_cols:
            val = (row.get(col) or "").strip()
            if col == "id":
                if not val: raise ValueError("empty id")
            elif col in numeric_cols and not is_finite(val):
                raise ValueError(f"non-finite {col}")
            clean[col] = val
        rid = clean["id"]
        if rid in preds: raise ValueError(f"duplicate id {rid}")
        preds[rid] = clean
    exp, act = set(expected_ids), set(preds)
    if missing := sorted(exp - act): raise ValueError(f"missing ids: {missing[0]}")
    if extra := sorted(act - exp): raise ValueError(f"unexpected ids: {extra[0]}")
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=expected_cols)
        w.writeheader()
        for rid in expected_ids: w.writerow(preds[rid])

def infer_numeric(rows, cols):
    num = set()
    for c in cols:
        if c == "id": continue
        vs = [(r.get(c) or "").strip() for r in rows]
        vs = [v for v in vs if v]
        if vs and all(is_finite(v) for v in vs): num.add(c)
    return num

def validate_ids(rows, source):
    ids = [(r.get("id") or "").strip() for r in rows]
    if not ids: raise ValueError(f"{source}: no rows")
    if any(not i for i in ids): raise ValueError(f"{source}: empty id")
    if len(set(ids)) != len(ids): raise ValueError(f"{source}: duplicate ids")
    return ids

def read_contract(task_dir):
    tc = read_json(task_dir / "task.json") if (task_dir / "task.json").is_file() else {}
    sp = task_dir / "sample_submission.csv"
    sc, sr = read_csv(sp) if sp.is_file() else ([], [])
    exp_cols = output_cols(tc) or sc or ["id", "prediction"]
    if "id" not in exp_cols: raise ValueError("id column required")
    exp_ids = validate_ids(sr, sp) if sr else read_test_ids(task_dir, tc)
    return exp_cols, exp_ids, infer_numeric(sr, exp_cols)

def read_test_ids(task_dir, tc):
    cands = []
    inf = tc.get("input_files")
    if isinstance(inf, dict):
        for k in ("test", "test_manifest"):
            v = inf.get(k)
            if isinstance(v, str): cands.append(task_dir / v)
    cands.append(task_dir / "data" / "test.csv")
    seen = set()
    for c in cands:
        if c in seen: continue
        seen.add(c)
        if c.is_file():
            fns, rows = read_csv(c)
            if "id" not in fns: raise ValueError(f"{c}: no id column")
            return validate_ids(rows, c)
    raise FileNotFoundError("no test manifest found")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    task_dir = args.task.resolve()
    output_path = args.output.resolve()
    output_dir = args.output_dir.resolve()

    ws_dir = output_dir / "workspace"
    ws_task = ws_dir / "task"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if ws_dir.exists(): shutil.rmtree(ws_dir)
    ws_dir.mkdir(parents=True)

    copy_task(task_dir, ws_task)
    task_config = read_json(ws_task / "task.json")
    ofiles = output_files(task_config)
    problem_type = detect_type(task_config, ws_task)
    modality = detect_modality(ws_task, task_config)

    print(f"[Agent] {problem_type} | {modality}", flush=True)

    system_prompt = build_system_prompt(problem_type, modality)

    task_md = (ws_task / "task.md").read_text(encoding="utf-8") if (ws_task / "task.md").is_file() else ""
    task_json_str = (ws_task / "task.json").read_text(encoding="utf-8") if (ws_task / "task.json").is_file() else ""
    data_files = ""
    dd = ws_task / "data"
    if dd.is_dir():
        lines = []
        for f in sorted(dd.rglob("*")):
            if f.is_file():
                try: sz = f"{f.stat().st_size:,}B"
                except: sz = "?"
                lines.append(f"  {f.relative_to(ws_task)} ({sz})")
        data_files = "\n".join(lines)

    train_head = ""
    train_csv = ws_task / "data" / "train.csv"
    if train_csv.is_file():
        cols, rows = read_csv(train_csv)
        train_head = f"Columns: {cols}\nRows: {len(rows)}\n"
        for r in rows[:3]: train_head += f"  {r}\n"

    user_msg = f"""## Task
{task_md}

## task.json
{task_json_str}

## Data Files
{data_files}

## Training Data (first rows)
{train_head}

## Output
Write predictions.csv to: {output_path}

Follow the workflow: audit data → understand problem → implement → output.
Write your solution as a ```python``` code block that writes predictions.csv."""

    import anthropic
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    client = anthropic.Anthropic(base_url=base_url)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[Agent] Attempt {attempt}/{MAX_RETRIES}", flush=True)
        try:
            resp = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=system_prompt,
                                          messages=[{"role": "user", "content": user_msg}])
            out = resp.content[0].text
            print(f"[Agent] Claude responded ({len(out)} chars)", flush=True)
        except Exception as e:
            print(f"[Agent] API ERROR: {e}", flush=True)
            time.sleep(5)
            continue

        code = extract_code(out)
        if not code:
            print("[Agent] No code block found, retrying...")
            user_msg += f"\n\nPrevious response (no code):\n{out[:500]}\n\nPlease write a ```python``` code block."
            time.sleep(2)
            continue

        print("=== GENERATED CODE ===", flush=True)
        print(code, flush=True)
        print("======================", flush=True)
        print(f"[Agent] Executing code ({len(code)} chars)...", flush=True)
        ok, run_out = run_code(code, ws_dir)
        print(f"[Agent] {'OK' if ok else 'FAILED'}", flush=True)
        print(run_out[-1000:], flush=True)

        if ok:
            try:
                exp_cols, exp_ids, num_cols = read_contract(ws_task)
                candidate = ws_dir / ofiles[0]
                if not candidate.is_file(): candidate = output_dir / ofiles[0]
                if not candidate.is_file(): candidate = output_path
                validate_output(candidate, exp_cols, exp_ids, num_cols)
                if candidate != output_path: shutil.copyfile(candidate, output_path)
                print("[Agent] VALIDATED", flush=True)
                return 0
            except Exception as e:
                print(f"[Agent] Validation failed: {e}", flush=True)
                user_msg += f"\n\nCode ran but predictions invalid: {e}\nFix and retry."
        else:
            user_msg += f"\n\nCode failed:\n{run_out[-800:]}\n\nFix errors and retry."

        time.sleep(2)

    # Fallback
    print("[Agent] Attempting fallback...", flush=True)
    try:
        write_fallback(ws_task, output_path, task_config)
        print("[Agent] Fallback written", flush=True)
        return 0
    except Exception as e:
        print(f"[Agent] Fallback failed: {e}", flush=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
