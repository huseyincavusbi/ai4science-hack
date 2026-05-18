#!/usr/bin/env bash
set -euo pipefail

TASK_DIR=""
OUTPUT_PATH=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK_DIR="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TASK_DIR" || -z "$OUTPUT_PATH" || -z "$OUTPUT_DIR" ]]; then
  echo "usage: ./run.sh --task TASK_DIR --output predictions.csv --output-dir OUTPUT_DIR" >&2
  exit 2
fi

SUBMISSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_FILE="$SUBMISSION_DIR/.env.secrets"
if [[ -f "$SECRETS_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$SECRETS_FILE"
  set +a
fi

mkdir -p "$OUTPUT_DIR" "$(dirname "$OUTPUT_PATH")"

BASELINE_RUNTIME_DIR="${BASELINE_RUNTIME_DIR:-${TMPDIR:-/tmp}/science-ai-baseline-runtime}"
mkdir -p "$BASELINE_RUNTIME_DIR"

export HOME="${HOME:-$BASELINE_RUNTIME_DIR/home}"
mkdir -p "$HOME"
export PATH="$HOME/.local/bin:$PATH"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$BASELINE_RUNTIME_DIR/pip-cache}"
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS="${CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS:-1}"

install_node_with_npm() {
  if [[ "${BASELINE_INSTALL_NODE:-1}" != "1" ]]; then
    echo "npm is unavailable; set BASELINE_INSTALL_NODE=1 or provide npm on PATH" >&2
    exit 127
  fi

  local node_dir="${BASELINE_NODE_DIR:-$BASELINE_RUNTIME_DIR/nodejs}"
  if [[ -x "$node_dir/bin/npm" ]]; then
    export PATH="$node_dir/bin:$PATH"
    return
  fi

  echo "npm is unavailable; downloading Node.js with npm into $node_dir" >&2
  mkdir -p "$(dirname "$node_dir")"

  BASELINE_NODE_DIR="$node_dir" python3 - <<'PY'
import hashlib
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(127)


def urlread(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


node_dir = Path(os.environ["BASELINE_NODE_DIR"]).resolve()
node_major = os.environ.get("BASELINE_NODE_MAJOR", "22").lstrip("v")
dist_url = os.environ.get(
    "BASELINE_NODE_DIST_URL",
    f"https://nodejs.org/dist/latest-v{node_major}.x",
).rstrip("/")

machine = platform.machine().lower()
if machine in {"x86_64", "amd64"}:
    arch = "x64"
elif machine in {"aarch64", "arm64"}:
    arch = "arm64"
else:
    fail(f"unsupported CPU architecture for Node.js bootstrap: {machine}")

target_suffix = f"-linux-{arch}.tar.xz"
try:
    shasums = urlread(f"{dist_url}/SHASUMS256.txt").decode("utf-8")
except Exception as exc:
    fail(f"failed to download Node.js checksum manifest from {dist_url}: {exc}")

filename = ""
expected_sha256 = ""
for line in shasums.splitlines():
    parts = line.strip().split()
    if len(parts) == 2 and parts[1].startswith("node-v") and parts[1].endswith(target_suffix):
        expected_sha256, filename = parts
        break

if not filename:
    fail(f"could not find a Node.js linux {arch} tarball in {dist_url}")

tmp_parent = node_dir.parent
tmp_parent.mkdir(parents=True, exist_ok=True)
work_dir = Path(tempfile.mkdtemp(prefix=".node-download-", dir=tmp_parent))
try:
    archive_path = work_dir / filename
    try:
        archive_path.write_bytes(urlread(f"{dist_url}/{filename}"))
    except Exception as exc:
        fail(f"failed to download Node.js archive {filename}: {exc}")

    actual_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        fail("downloaded Node.js archive failed SHA256 verification")

    extract_dir = work_dir / "extract"
    extract_dir.mkdir()
    extract_root = extract_dir.resolve()
    try:
        with tarfile.open(archive_path, "r:xz") as archive:
            for member in archive.getmembers():
                member_path = (extract_root / member.name).resolve()
                if extract_root != member_path and extract_root not in member_path.parents:
                    fail(f"unsafe path in Node.js archive: {member.name}")
            archive.extractall(extract_root, filter="data")
    except Exception as exc:
        fail(f"failed to extract Node.js archive: {exc}")

    extracted_dirs = [path for path in extract_dir.iterdir() if path.is_dir()]
    if len(extracted_dirs) != 1:
        fail("Node.js archive did not contain exactly one top-level directory")

    if node_dir.exists():
        shutil.rmtree(node_dir)
    shutil.move(str(extracted_dirs[0]), str(node_dir))
finally:
    shutil.rmtree(work_dir, ignore_errors=True)

if not (node_dir / "bin" / "npm").is_file():
    fail("Node.js bootstrap completed but npm was not installed")
PY

  chmod +x \
    "$node_dir/bin/node" \
    "$node_dir/lib/node_modules/npm/bin/npm-cli.js" \
    "$node_dir/lib/node_modules/npm/bin/npx-cli.js" \
    2>/dev/null || true
  export PATH="$node_dir/bin:$PATH"
}

if [[ "${BASELINE_SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  VENV_DIR="${BASELINE_VENV_DIR:-$BASELINE_RUNTIME_DIR/baseline-venv}"
  if python3 -m venv "$VENV_DIR" >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    python -m pip install \
      --disable-pip-version-check \
      --no-input \
      --quiet \
      -r "$SUBMISSION_DIR/requirements.txt"
  else
    python3 -m pip install \
      --disable-pip-version-check \
      --no-input \
      --quiet \
      --user \
      -r "$SUBMISSION_DIR/requirements.txt" || \
    python3 -m pip install \
      --disable-pip-version-check \
      --no-input \
      --quiet \
      --break-system-packages \
      -r "$SUBMISSION_DIR/requirements.txt"
  fi
fi

# -- Choose agent backend ------------------------------------------------
# ANTHROPIC_API_KEY takes priority (hackathon LiteLLM proxy).
# Falls back to Gemini for local testing if no Claude key.
# ------------------------------------------------------------------------

if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "Using Claude agent backend (via LiteLLM proxy)" >&2

  CLAUDE_BIN="${CLAUDE_CLI_BIN:-claude}"
  if ! command -v "$CLAUDE_BIN" >/dev/null 2>&1; then
    if [[ "${BASELINE_INSTALL_CLAUDE:-1}" == "1" ]]; then
      if ! command -v npm >/dev/null 2>&1; then
        install_node_with_npm
      fi
      export NPM_CONFIG_PREFIX="${NPM_CONFIG_PREFIX:-$BASELINE_RUNTIME_DIR/npm-global}"
      mkdir -p "$NPM_CONFIG_PREFIX"
      export PATH="$NPM_CONFIG_PREFIX/bin:$PATH"
      npm install -g @anthropic-ai/claude-code
      CLAUDE_BIN="${CLAUDE_CLI_BIN:-claude}"
    else
      echo "claude CLI not found; set BASELINE_INSTALL_CLAUDE=1 or provide CLAUDE_CLI_BIN" >&2
      exit 127
    fi
  fi

  export CLAUDE_CLI_BIN="$CLAUDE_BIN"

  python3 "$SUBMISSION_DIR/scripts/run_claude_agent.py" \
    --task "$TASK_DIR" \
    --output "$OUTPUT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --submission-dir "$SUBMISSION_DIR"

elif [[ -n "${LLM_API_KEY:-}${OPENAI_API_KEY:-}" ]]; then
  echo "Using LLM agent backend (OpenAI-compatible via run_gemini_agent.py)" >&2
  python3 "$SUBMISSION_DIR/scripts/run_gemini_agent.py" \
    --task "$TASK_DIR" \
    --output "$OUTPUT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --submission-dir "$SUBMISSION_DIR"
