#!/usr/bin/env python3
"""Autonomous ML research agent — LLM-agnostic via OpenAI-compatible API.

Identical contract to run_claude_agent.py:
  --task TASK_DIR --output OUTPUT_PATH --output-dir OUTPUT_DIR --submission-dir SUBMISSION_DIR

Supports any OpenAI-compatible endpoint:
  - Hackathon LiteLLM proxy (Claude via Bedrock)
  - Gemini via OpenAI-compatible mode
  - Any OpenAI-compatible provider

Set via env:
  LLM_API_KEY          — API key (required)
  LLM_BASE_URL         — Base URL (default: https://api.openai.com/v1)
  LLM_MODEL            — Model name (default: bedrock-claude-sonnet)
  LLM_MAX_TURNS        — Max tool-use turns (default: 50)

Differences from the Claude baseline:
  - Detects problem type + data modality, injects ML strategy guidance
  - Tool-use loop: read_file / run_command / write_file / finish
  - 50 turn budget (vs 30)
  - Retry on validation failure with error feedback
  - No Node.js or Claude Code dependency
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI (identical to run_claude_agent.py)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--submission-dir", required=True, type=Path)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Task utilities (shared logic, same as run_claude_agent.py)
# ---------------------------------------------------------------------------

def copy_task(task_dir: Path, workspace_task_dir: Path) -> None:
    if workspace_task_dir.exists():
        shutil.rmtree(workspace_task_dir)
    shutil.copytree(task_dir, workspace_task_dir)
    data_dir = workspace_task_dir / "data"
    if data_dir.is_dir():
        for archive_path in sorted(data_dir.glob("*.zip")):
            if zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path) as archive:
                    archive.extractall(workspace_task_dir)


def read_task_config(task_dir: Path) -> dict[str, object]:
    task_json = task_dir / "task.json"
    if not task_json.is_file():
        return {}
    payload = json.loads(task_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{task_json} must contain a JSON object")
    return payload


def output_columns_from_task(task_config: dict[str, object]) -> list[str]:
    output = task_config.get("output")
    if not isinstance(output, dict):
        return []
    columns = output.get("columns")
    if not isinstance(columns, list):
        return []
    return [str(c) for c in columns]


def output_files_from_task(task_config: dict[str, object]) -> list[str]:
    output = task_config.get("output")
    if not isinstance(output, dict):
        return ["predictions.csv"]
    files = output.get("files")
    if isinstance(files, list) and files and all(isinstance(item, str) for item in files):
        return [str(item) for item in files]
    file_name = output.get("file")
    if isinstance(file_name, str) and file_name:
        return [file_name]
    return ["predictions.csv"]


def is_multi_file_output(task_config: dict[str, object]) -> bool:
    output = task_config.get("output")
    return isinstance(output, dict) and isinstance(output.get("files"), list)


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} is missing a header")
        return list(reader.fieldnames), list(reader)


def is_finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Task type and data modality detection
# ---------------------------------------------------------------------------

def detect_problem_type(task_config: dict[str, object], task_dir: Path) -> str:
    problem = str(task_config.get("problem_type", ""))
    if problem:
        return problem

    metric = str(task_config.get("metric", "")).lower()
    task_md = task_dir / "task.md"
    text = task_md.read_text(encoding="utf-8").lower() if task_md.is_file() else ""

    if any(w in text for w in ("treatment effect", "causal", "pehe", "counterfactual")):
        return "causal_effect_estimation"
    if any(w in text for w in ("domain generalization", "domain shift", "distribution shift")):
        return "binary_classification"
    if any(w in text for w in ("backdoor", "artifact", "trigger", "spurious")):
        return "classification"
    if any(w in text for w in ("feature engineering", "descriptor", "features.csv")):
        return "feature_engineering"
    if "spearman" in metric or "correlation" in metric:
        return "regression"
    if "auprc" in metric or "auroc" in metric:
        return "binary_classification"
    if "accuracy" in metric or "acc" in metric:
        return "classification"
    if "r2" in metric or "r²" in metric:
        return "regression"
    return "unknown"


def detect_data_modality(task_dir: Path, task_config: dict[str, object]) -> str:
    input_files = task_config.get("input_files", {})
    if isinstance(input_files, dict):
        for v in input_files.values():
            if isinstance(v, str) and any(v.lower().endswith(ext) for ext in (".zip", ".npz", ".png", ".jpg")):
                return "image"
            if isinstance(v, str) and "image" in v.lower():
                return "image"

    data_dir = task_dir / "data"
    if data_dir.is_dir():
        for f in data_dir.iterdir():
            if f.suffix in (".zip", ".npz") and "image" in f.name.lower():
                return "image"

    train_csv = task_dir / "data" / "train.csv"
    if train_csv.is_file():
        cols, _ = read_csv_rows(train_csv)
        for col in cols:
            cl = col.lower()
            if any(kw in cl for kw in ("image_path", "image_index", "img", "picture")):
                return "image"
            if any(kw in cl for kw in ("sequence", "aa_seq", "smiles", "peptide")):
                return "sequence"
    return "tabular"


# ---------------------------------------------------------------------------
# Strategy-aware system prompt
# ---------------------------------------------------------------------------

def build_system_prompt(problem_type: str, modality: str, output_files: list[str], multi_file: bool) -> str:

    STRATEGIES = {
        "binary_classification": """
### Binary Classification
- Check class balance immediately. If imbalanced, use class_weight='balanced'.
- **Images**: Use a pretrained ResNet-18 from torchvision. Apply heavy augmentation
  (ColorJitter, RandomHorizontalFlip, RandomRotation) especially if domain shift is suspected.
- **Tabular**: Try LogisticRegression, RandomForest, GradientBoosting. Scale features.
- Split 80/20 stratified for internal validation. Report validation accuracy.
- Output: binary 0/1 or float [0,1] exactly as task specifies.
""",
        "classification": """
### Multi-class Classification
- Count classes. Check for imbalance.
- **Images (28x28 MNIST-like)**: Lightweight CNN: 2-3 conv layers (32→64→128 filters),
  BatchNorm, Dropout(0.3), dense head. Train with Adam(1e-3), CrossEntropyLoss.
- **Backdoor/artifact tasks**: Train a clean model first. Then:
  - Inspect images for visual trigger patterns (compute per-pixel variance across classes)
  - Apply iterative pruning: remove neurons with highest activation on suspect samples
  - Use strong augmentation (RandAugment) to dilute spurious correlations
  - Consider an ensemble of 3 independently-trained models
- Output: integer class labels 0-9.
""",
        "regression": """
### Regression
- Check target distribution. Log-transform if highly skewed (spanning orders of magnitude).
- **Tabular**: RidgeCV, RandomForestRegressor, GradientBoostingRegressor.
- Cross-validate (5-fold). Pick best R²/correlation model.
- Output: float predictions.
""",
        "causal_effect_estimation": """
### Causal Effect Estimation (NOT standard supervised learning)
You are estimating counterfactuals: "what would the outcome be if treatment were
applied vs not applied, for the SAME unit?"

- **Step 1**: Check covariate balance. Compute standardized mean differences between
  treated and control for each covariate.
- **Step 2**: Fit a propensity score model (LogisticRegression predicting treatment from X).
- **Step 3**: Try these estimators:
  - T-learner: model_treated(X) and model_control(X), effect = diff
  - X-learner: T-learner + propensity weighting, better with imbalanced groups
  - Doubly Robust: combines outcome model + propensity model, robust to misspecification
- **Step 4**: Evaluate on held-out train split. Compare ATE estimates.
- Output: one float per test row (estimated treatment effect). NOT a probability.
- Validate that predicted effects are reasonable scale vs observed outcomes.
""",
        "feature_engineering": """
### Feature Engineering (NOT prediction - create features, scorer fits model)
- Read element/descriptor input columns.
- Create features: element properties, stoichiometric ratios, pairwise interactions,
  polynomial combinations, statistical moments of element groups.
- All features must be numeric and finite. Drop constant/near-constant columns.
- Train and test feature files MUST have identical column names in same order.
- Include 'id' as first column in each file.
- Number of features: 10-200 is reasonable.
""",
    }

    strategy = STRATEGIES.get(problem_type, """
### General Strategy
- Read task.md. Understand input, target, metric.
- Explore data: shapes, types, distributions.
- Images → PyTorch. Tabular → sklearn. Always validate on a train split.
- Output exact format specified.
""")

    MODALITY = {
        "image": """
### Image Data
- .zip files: extract with zipfile. .npz files: np.load (uint8, shape N×H×W or N×H×W×C).
- Use torch.utils.data.DataLoader, batch_size=64, num_workers=2.
- Normalize: [0,1] or ImageNet stats for pretrained models.
- GPU (NVIDIA L4) available. Use device='cuda' if torch.cuda.is_available().
- 28×28 grayscale → simple CNN. Larger RGB → ResNet-18 pretrained.
""",
        "tabular": """
### Tabular Data
- Load with pandas. Check dtypes, missing values, summary stats.
- Categorical: one-hot or label encode. Numeric: handle outliers, scale for linear models.
- Use sklearn Pipeline for reproducible preprocessing.
- Handle NaN/Inf values before training.
""",
        "sequence": """
### Sequence Data
- AA sequences: BLOSUM62 encoding, one-hot, or physicochemical properties.
- SMILES: Morgan fingerprints (1024-bit, radius=2) via RDKit if available, or character n-grams.
""",
    }

    modality_guide = MODALITY.get(modality, MODALITY["tabular"])

    if multi_file:
        output_guide = f"""
### Output (Multi-File Feature Task)
Create EXACTLY these files: {', '.join(output_files)}
Train and test files MUST share identical columns (names + order).
id column first. All feature values finite numeric.
"""
    else:
        output_guide = """
### Output (Single Prediction File)
Write predictions.csv with columns matching task.json or sample_submission.csv.
All test IDs present, no extras, no duplicates. Numeric columns must be finite.
"""

    return f"""You are an autonomous AI research agent solving machine learning tasks.

## Environment
- Python 3.11 with numpy, pandas, scikit-learn, torch, torchvision, causalml, PIL.
- GPU: NVIDIA L4 (CUDA available).
- Workspace: /app/output/claude_science_ai_workspace/
- Task data (read-only): /app/input_data/
- Output directory: /app/output/

## Workflow
1. Read task.md and task.json to understand the problem.
2. Explore data files (shapes, types, distributions).
3. Plan your ML approach based on the problem type.
4. Write and execute Python training scripts. Always include a validation split.
5. Write predictions to the required output file(s).
6. Call the finish tool when done.

{strategy}

{modality_guide}

{output_guide}

## Tool Usage
- write_file: create Python scripts. Make them complete and self-contained.
- run_command: execute scripts. Use python3 one-liners for data exploration.
  Example: python3 -c "import pandas as pd; print(pd.read_csv('task/data/train.csv').describe())"
- read_file: examine task instructions, data files, script output.
- finish: signal task completion with a brief summary.

## Constraints
- Time limit ~55 minutes. Be efficient. Don't do exhaustive grid search.
- Always validate output format before calling finish.
- Stay inside workspace. Do NOT read reference/, scoring/ or hidden labels.
- Train your own models on provided data. No external API inference.
"""


# ---------------------------------------------------------------------------
# Task prompt (user message with concrete paths and file contents)
# ---------------------------------------------------------------------------

def build_task_prompt(
    task_dir: Path,
    workspace_task_dir: Path,
    workspace_dir: Path,
    output_dir: Path,
    output_path: Path,
    output_files: list[str],
    multi_file: bool,
) -> str:
    task_md = ""
    md_path = workspace_task_dir / "task.md"
    if md_path.is_file():
        task_md = md_path.read_text(encoding="utf-8")

    task_json = ""
    jp = workspace_task_dir / "task.json"
    if jp.is_file():
        task_json = jp.read_text(encoding="utf-8")

    data_files = ""
    data_dir = workspace_task_dir / "data"
    if data_dir.is_dir():
        lines = []
        for f in sorted(data_dir.rglob("*")):
            if f.is_file():
                try:
                    size = f.stat().st_size
                    sz = f"{size:,}B" if size < 1_000_000 else f"{size/1_000_000:.1f}MB"
                except OSError:
                    sz = "?"
                lines.append(f"  {f.relative_to(workspace_task_dir)} ({sz})")
        data_files = "\n".join(lines)

    if multi_file:
        output_target = "\n".join(f"- {output_dir}/{f}" for f in output_files)
    else:
        output_target = str(output_path)

    return f"""## Task

### task.md
{task_md}

### task.json
{task_json}

### Data Files
{data_files}

## Paths
- Original task (read-only): {task_dir}
- Writable task copy: {workspace_task_dir}
- Writable workspace: {workspace_dir}
- Required output: {output_target}

Start by reading task.md and exploring the data. Then implement, train, and produce predictions.
"""


# ---------------------------------------------------------------------------
# Output validation (identical to run_claude_agent.py)
# ---------------------------------------------------------------------------

def validate_ids(rows: list[dict[str, str]], source: Path) -> list[str]:
    ids = [(row.get("id") or "").strip() for row in rows]
    if not ids:
        raise ValueError(f"{source} contains no rows")
    if any(not r for r in ids):
        raise ValueError(f"{source} contains an empty id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{source} contains duplicate ids")
    return ids


def infer_numeric_columns(rows: list[dict[str, str]], columns: list[str]) -> set[str]:
    numeric: set[str] = set()
    for col in columns:
        if col == "id":
            continue
        vals = [(r.get(col) or "").strip() for r in rows]
        vals = [v for v in vals if v]
        if vals and all(is_finite_number(v) for v in vals):
            numeric.add(col)
    return numeric


def read_test_ids(task_dir: Path, task_config: dict[str, object]) -> list[str]:
    candidates: list[Path] = []
    input_files = task_config.get("input_files")
    if isinstance(input_files, dict):
        for k in ("test", "test_manifest"):
            v = input_files.get(k)
            if isinstance(v, str):
                candidates.append(task_dir / v)
    candidates.append(task_dir / "data" / "test.csv")
    seen: set[Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.is_file():
            fns, rows = read_csv_rows(c)
            if "id" not in fns:
                raise ValueError(f"{c} must contain an id column")
            return validate_ids(rows, c)
    raise FileNotFoundError("could not find test manifest with ids")


def read_input_ids(task_dir: Path, task_config: dict[str, object], key: str) -> list[str]:
    input_files = task_config.get("input_files")
    if isinstance(input_files, dict):
        v = input_files.get(key)
        if isinstance(v, str):
            p = task_dir / v
            if p.is_file():
                fns, rows = read_csv_rows(p)
                if "id" not in fns:
                    raise ValueError(f"{p} must contain an id column")
                return validate_ids(rows, p)
    p = task_dir / "data" / f"{key}.csv"
    fns, rows = read_csv_rows(p)
    if "id" not in fns:
        raise ValueError(f"{p} must contain an id column")
    return validate_ids(rows, p)


def read_output_contract(task_dir: Path) -> tuple[list[str], list[str], set[str]]:
    task_config = read_task_config(task_dir)
    sample_path = task_dir / "sample_submission.csv"
    sample_cols: list[str] = []
    sample_rows: list[dict[str, str]] = []
    if sample_path.is_file():
        sample_cols, sample_rows = read_csv_rows(sample_path)
    expected_cols = output_columns_from_task(task_config) or sample_cols or ["id", "prediction"]
    if "id" not in expected_cols:
        raise ValueError("output columns must include id")
    expected_ids = validate_ids(sample_rows, sample_path) if sample_rows else read_test_ids(task_dir, task_config)
    numeric = infer_numeric_columns(sample_rows, expected_cols)
    return expected_cols, expected_ids, numeric


def validate_predictions(pred_csv: Path, *, expected_columns: list[str], expected_ids: list[str], numeric_columns: set[str]) -> None:
    if not pred_csv.is_file():
        raise ValueError(f"{pred_csv} was not created")
    fns, rows = read_csv_rows(pred_csv)
    if fns != expected_columns:
        raise ValueError(f"predictions columns {fns} != expected {expected_columns}")
    preds: dict[str, dict[str, str]] = {}
    for row in rows:
        if None in row:
            raise ValueError("row has too many columns")
        clean: dict[str, str] = {}
        for col in expected_columns:
            val = (row.get(col) or "").strip()
            if col == "id":
                if not val:
                    raise ValueError("empty id")
            else:
                if val == "":
                    raise ValueError(f"empty {col} for id {row.get('id','').strip()!r}")
                if col in numeric_columns and not is_finite_number(val):
                    raise ValueError(f"non-finite {col} for id {row.get('id','').strip()!r}")
            clean[col] = val
        rid = clean["id"]
        if rid in preds:
            raise ValueError(f"duplicate id {rid}")
        preds[rid] = clean
    exp = set(expected_ids)
    act = set(preds)
    if missing := sorted(exp - act):
        raise ValueError(f"missing {len(missing)} ids, first: {missing[0]}")
    if extra := sorted(act - exp):
        raise ValueError(f"{len(extra)} unexpected ids, first: {extra[0]}")
    with pred_csv.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=expected_columns)
        w.writeheader()
        for rid in expected_ids:
            w.writerow(preds[rid])


def validate_feature_file(path: Path, expected_ids: list[str]) -> list[str]:
    if not path.is_file():
        raise ValueError(f"{path} was not created")
    fns, rows = read_csv_rows(path)
    if not fns or fns[0] != "id":
        raise ValueError(f"{path.name} must start with id column")
    feat_cols = fns[1:]
    if not feat_cols:
        raise ValueError(f"{path.name} must have at least one feature column")
    if len(set(feat_cols)) != len(feat_cols):
        raise ValueError(f"{path.name} duplicate feature columns")
    seen: set[str] = set()
    for row in rows:
        if None in row:
            raise ValueError(f"{path.name} row has too many columns")
        rid = (row.get("id") or "").strip()
        if not rid:
            raise ValueError(f"{path.name} empty id")
        if rid in seen:
            raise ValueError(f"{path.name} duplicate id {rid!r}")
        seen.add(rid)
        for col in feat_cols:
            v = (row.get(col) or "").strip()
            if not is_finite_number(v):
                raise ValueError(f"{path.name} non-finite {col} for id {rid!r}")
    exp = set(expected_ids)
    if missing := sorted(exp - seen):
        raise ValueError(f"{path.name} missing {len(missing)} ids, first: {missing[0]}")
    if extra := sorted(seen - exp):
        raise ValueError(f"{path.name} {len(extra)} unexpected ids, first: {extra[0]}")
    return feat_cols


def _find_output(workspace_dir: Path, output_dir: Path, rel: str) -> Path:
    for base in (workspace_dir, output_dir):
        c = base / rel
        if c.is_file():
            return c
    return workspace_dir / rel


def validate_feature_outputs(*, workspace_task_dir: Path, workspace_dir: Path, output_dir: Path, task_config: dict[str, object], output_files: list[str]) -> None:
    if len(output_files) != 2:
        raise ValueError("multi-file tasks must declare exactly 2 output files")
    train_ids = read_input_ids(workspace_task_dir, task_config, "train")
    test_ids = read_input_ids(workspace_task_dir, task_config, "test")
    tp = _find_output(workspace_dir, output_dir, output_files[0])
    ep = _find_output(workspace_dir, output_dir, output_files[1])
    tc = validate_feature_file(tp, train_ids)
    ec = validate_feature_file(ep, test_ids)
    if tc != ec:
        raise ValueError("train and test feature files must share same columns")
    for src, rel in ((tp, output_files[0]), (ep, output_files[1])):
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)


# ---------------------------------------------------------------------------
# Gemini API tool-use agent loop
# ---------------------------------------------------------------------------

MAX_RESULT_CHARS = 12000
CMD_TIMEOUT_S = 600


def _tool_read_file(path_str: str, workspace_dir: Path) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = workspace_dir / p
    if not p.exists():
        return f"ERROR: not found: {p}"
    if p.is_dir():
        entries = sorted(p.iterdir())
        return "Directory:\n" + "\n".join(f"  {e.name}{'/' if e.is_dir() else ''}" for e in entries)
    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"ERROR: binary file: {p}"
    if len(content) > MAX_RESULT_CHARS:
        content = content[:MAX_RESULT_CHARS] + f"\n\n... truncated ({len(content)} chars total)"
    return content


def _tool_run_command(command: str, workspace_dir: Path, output_dir: Path) -> str:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["WORKSPACE"] = str(workspace_dir)
    env["OUTPUT_DIR"] = str(output_dir)
    try:
        r = subprocess.run(["bash", "-lc", command], cwd=workspace_dir, capture_output=True, text=True, timeout=CMD_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {CMD_TIMEOUT_S}s"
    except Exception as e:
        return f"ERROR: {e}"
    out = r.stdout
    if r.stderr:
        out += "\n[stderr]\n" + r.stderr
    if r.returncode != 0:
        out += f"\n[exit={r.returncode}]"
    if len(out) > MAX_RESULT_CHARS:
        lines = out.split("\n")
        out = "\n".join(lines[:80]) + f"\n\n... {len(lines)-120} lines omitted ...\n\n" + "\n".join(lines[-40:])
    return out.strip() or "(no output)"


def _tool_write_file(path_str: str, content: str, workspace_dir: Path) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = workspace_dir / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: wrote {len(content)} chars to {p}"


def run_agent_loop(
    system_instruction: str,
    task_prompt: str,
    workspace_dir: Path,
    output_dir: Path,
    log_path: Path,
) -> bool:
    """Agent loop using OpenAI-compatible function calling."""
    from openai import OpenAI

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL", "bedrock-claude-sonnet")
    max_turns = int(os.environ.get("LLM_MAX_TURNS", "50"))

    if not api_key:
        raise ValueError("LLM_API_KEY or OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url=base_url)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file or list a directory. Use to examine task.md, task.json, data files, or check output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path relative to workspace, e.g. 'task/task.md' or 'data/train.csv'",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Execute a bash command in the workspace. Use for data exploration or running training scripts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Bash command to execute",
                        }
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write or overwrite a file. Use to create Python scripts and output csv files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path, e.g. 'train_model.py' or 'predictions.csv'",
                        },
                        "content": {
                            "type": "string",
                            "description": "Full file contents",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Signal task completion after writing and validating output files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of approach and results",
                        }
                    },
                    "required": ["summary"],
                },
            },
        },
    ]

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": task_prompt},
    ]

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Model: {model} | Base URL: {base_url} | Max turns: {max_turns}\n\n=== START ===\n\n")
        log.flush()

        turn = 0
        for turn in range(1, max_turns + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    temperature=0.2,
                    max_tokens=4096,
                )
            except Exception as exc:
                log.write(f"API ERROR turn {turn}: {exc}\n")
                break

            msg = response.choices[0].message
            content = msg.content or ""
            tool_calls = msg.tool_calls or []

            if content:
                log.write(f"\n--- Turn {turn} (text) ---\n{content}\n")
                log.flush()
                print(f"[Agent turn {turn}] {content[:200]}", flush=True)

            if not tool_calls:
                log.write(f"\nTurn {turn}: no tool calls, agent done.\n")
                break

            # Append assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                log.write(f"\n--- Turn {turn} tool: {name}({json.dumps(args, default=str)[:400]}) ---\n")
                log.flush()

                t0 = time.time()
                try:
                    if name == "read_file":
                        result = _tool_read_file(str(args.get("path", "")), workspace_dir)
                    elif name == "run_command":
                        result = _tool_run_command(str(args.get("command", "")), workspace_dir, output_dir)
                    elif name == "write_file":
                        result = _tool_write_file(str(args.get("path", "")), str(args.get("content", "")), workspace_dir)
                    elif name == "finish":
                        log.write(f"Agent finished: {args.get('summary','')}\n")
                        log.flush()
                        return True
                    else:
                        result = f"ERROR: unknown tool '{name}'"
                except Exception as exc:
                    result = f"ERROR: {exc}"

                dt = time.time() - t0
                log.write(f"({dt:.1f}s) {result[:2000]}\n")
                log.flush()

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result[:MAX_RESULT_CHARS],
                })

        log.write(f"\n=== END after {turn} turns ===\n")
    return False


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    task_dir = args.task.resolve()
    output_path = args.output.resolve()
    output_dir = args.output_dir.resolve()
    submission_dir = args.submission_dir.resolve()

    # Same workspace layout as Claude agent so run.sh paths stay consistent
    workspace_dir = output_dir / "claude_science_ai_workspace"
    workspace_task_dir = workspace_dir / "task"
    log_path = output_dir / "agent.log"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True)

    copy_task(task_dir, workspace_task_dir)
    task_config = read_task_config(workspace_task_dir)
    output_files = output_files_from_task(task_config)
    multi_file = is_multi_file_output(task_config)

    problem_type = detect_problem_type(task_config, workspace_task_dir)
    modality = detect_data_modality(workspace_task_dir, task_config)
    print(f"[Agent] problem_type={problem_type}  modality={modality}", flush=True)

    system_prompt = build_system_prompt(problem_type, modality, output_files, multi_file)
    task_prompt = build_task_prompt(task_dir, workspace_task_dir, workspace_dir, output_dir, output_path, output_files, multi_file)

    (workspace_dir / "SYSTEM.txt").write_text(system_prompt, encoding="utf-8")
    (workspace_dir / "TASK.txt").write_text(task_prompt, encoding="utf-8")

    for attempt in (1, 2):
        alog = log_path if attempt == 1 else log_path.with_suffix(f".attempt{attempt}.log")
        print(f"[Agent] Attempt {attempt}/2", flush=True)
        run_agent_loop(system_prompt, task_prompt, workspace_dir, output_dir, alog)

        try:
            if multi_file:
                validate_feature_outputs(workspace_task_dir=workspace_task_dir, workspace_dir=workspace_dir, output_dir=output_dir, task_config=task_config, output_files=output_files)
            else:
                expected_cols, expected_ids, numeric_cols = read_output_contract(workspace_task_dir)
                candidate = workspace_dir / output_files[0]
                if not candidate.is_file():
                    candidate = output_dir / output_files[0]
                if not candidate.is_file():
                    candidate = output_path
                validate_predictions(candidate, expected_columns=expected_cols, expected_ids=expected_ids, numeric_columns=numeric_cols)
                if candidate != output_path:
                    shutil.copyfile(candidate, output_path)
            print("[Agent] Output validated OK", flush=True)
            return 0
        except Exception as exc:
            print(f"[Agent] Validation FAILED: {exc}", flush=True)
            if attempt == 1:
                task_prompt += f"\n\n## PREVIOUS ATTEMPT FAILED\nError: {exc}\nFix column names, IDs, and numeric format. Try again."
            else:
                msg = textwrap.dedent(f"""Agent failed: {exc}\nLog: {log_path}""").strip()
                (output_dir / "baseline_error.txt").write_text(msg + "\n", encoding="utf-8")
                print(msg, file=sys.stderr)
                return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
