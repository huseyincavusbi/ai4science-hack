"""
AI4Science Hackathon — Science of AI/ML Track
Autonomous domain generalization agent.

Usage:
    python agent.py --task_dir /path/to/task

The agent:
1. Reads task.json + metadata.json to diagnose the problem
2. Uses Claude to reason about the spurious correlation and select a strategy
3. Generates and executes Python training code
4. Writes predictions.csv to the task directory
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-5"  # use opus for best reasoning
MAX_TOKENS = 4096
MAX_CODE_RETRIES = 3

# ── Load system prompt ─────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_PATH = os.path.join(SCRIPT_DIR, "prompts", "system_prompt.txt")

with open(SYSTEM_PROMPT_PATH, "r") as f:
    SYSTEM_PROMPT = f.read()

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def read_file_safe(path: str, max_bytes: int = 8000) -> str:
    """Read a file, truncating if large (to stay within context limits)."""
    if not os.path.exists(path):
        return f"[File not found: {path}]"
    with open(path, "rb") as f:
        raw = f.read(max_bytes)
    text = raw.decode("utf-8", errors="replace")
    if os.path.getsize(path) > max_bytes:
        text += f"\n... [truncated, full file is {os.path.getsize(path)} bytes]"
    return text


def extract_code_block(text: str) -> str | None:
    """Extract the first ```python ... ``` block from agent output."""
    import re
    pattern = r"```python\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fallback: look for plain ``` blocks
    pattern2 = r"```\s*(.*?)```"
    match2 = re.search(pattern2, text, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    return None


def run_code(code: str, task_dir: str) -> tuple[bool, str]:
    """Write code to a temp file and execute it. Returns (success, output)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=2400,  # 40 min hard limit for training
            cwd=task_dir,
        )
        output = result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else "")
        success = result.returncode == 0
        return success, output
    except subprocess.TimeoutExpired:
        return False, "ERROR: Code execution timed out after 40 minutes."
    except Exception as e:
        return False, f"ERROR: Failed to run code: {e}"
    finally:
        os.unlink(tmp_path)


def verify_predictions(task_dir: str, metadata: dict) -> tuple[bool, str]:
    """Check predictions.csv is valid."""
    import csv

    pred_path = os.path.join(task_dir, "predictions.csv")
    if not os.path.exists(pred_path):
        return False, "predictions.csv not found"

    with open(pred_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return False, "predictions.csv is empty"

    expected = metadata.get("num_test")
    if expected and len(rows) != expected:
        return False, f"Row count mismatch: got {len(rows)}, expected {expected}"

    values = set(str(r.get("prediction", "")).strip() for r in rows)
    if values <= {"0"} or values <= {"1"}:
        return False, f"All predictions are the same class: {values}"

    for r in rows:
        if r.get("prediction", "").strip() not in {"0", "1"}:
            return False, f"Non-binary prediction found: {r}"

    return True, f"OK — {len(rows)} predictions, classes: {values}"


# ── Agent loop ────────────────────────────────────────────────────────────────

def build_user_message(task_dir: str) -> str:
    task_json   = read_file_safe(os.path.join(task_dir, "task.json"))
    meta_json   = read_file_safe(os.path.join(task_dir, "metadata.json"))
    train_head  = read_file_safe(os.path.join(task_dir, "data", "train.csv"), max_bytes=2000)
    test_head   = read_file_safe(os.path.join(task_dir, "data", "test.csv"),  max_bytes=2000)

    return f"""Solve the task in this directory: {task_dir}

## task.json
{task_json}

## metadata.json
{meta_json}

## data/train.csv (first 2KB)
{train_head}

## data/test.csv (first 2KB)
{test_head}

Follow your instructions exactly. Diagnose the problem, select a strategy, then write a complete Python script in a ```python``` code block that trains a model and writes predictions.csv to {task_dir}.
"""


def run_agent(task_dir: str):
    client = anthropic.Anthropic()
    metadata = load_json(os.path.join(task_dir, "metadata.json"))

    print(f"\n{'='*60}")
    print(f"AGENT START — task: {task_dir}")
    print(f"{'='*60}\n")

    messages = [{"role": "user", "content": build_user_message(task_dir)}]

    for attempt in range(1, MAX_CODE_RETRIES + 1):
        print(f"[Attempt {attempt}/{MAX_CODE_RETRIES}] Calling Claude...")

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        agent_output = response.content[0].text
        print("\n--- Agent reasoning ---")
        print(agent_output[:2000], "..." if len(agent_output) > 2000 else "")
        print("-----------------------\n")

        code = extract_code_block(agent_output)
        if not code:
            print("WARNING: No code block found in agent output. Retrying...")
            messages.append({"role": "assistant", "content": agent_output})
            messages.append({
                "role": "user",
                "content": "You did not provide a Python code block. Please provide your complete solution as a ```python``` code block now."
            })
            continue

        print(f"[Attempt {attempt}] Executing generated code...")
        success, output = run_code(code, task_dir)
        print("--- Execution output ---")
        print(output[:3000])
        print("------------------------\n")

        if success:
            valid, msg = verify_predictions(task_dir, metadata)
            if valid:
                print(f"✅ SUCCESS — predictions.csv verified: {msg}")
                return True
            else:
                print(f"⚠️  Predictions invalid: {msg}")
                error_feedback = f"Code ran but predictions.csv failed validation: {msg}\nExecution output:\n{output}"
        else:
            error_feedback = f"Code execution failed.\nExecution output:\n{output}"

        if attempt < MAX_CODE_RETRIES:
            print(f"Feeding error back to agent for retry...\n")
            messages.append({"role": "assistant", "content": agent_output})
            messages.append({
                "role": "user",
                "content": f"{error_feedback}\n\nPlease fix the code and provide a corrected ```python``` code block."
            })

    print("❌ FAILED — agent could not produce valid predictions after max retries.")
    return False


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI4Science domain generalization agent")
    parser.add_argument(
        "--task_dir",
        type=str,
        default=".",
        help="Path to the task directory (default: current directory)"
    )
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    if not os.path.isdir(task_dir):
        print(f"ERROR: task_dir not found: {task_dir}")
        sys.exit(1)

    success = run_agent(task_dir)
    sys.exit(0 if success else 1)
