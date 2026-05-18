#!/bin/bash
set -e

# -- Hackathon LiteLLM credentials ----------------------------------------
export ANTHROPIC_API_KEY="sk-7NRx1SUb0hjL7-13RgBgZA"
export ANTHROPIC_BASE_URL="https://hackathon.tiptreesystems.com/litellm"
# -------------------------------------------------------------------------

TASK_DIR=""
OUTPUT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --task)       TASK_DIR="$2"; shift 2 ;;
    --output)     OUTPUT="$2"; shift 2 ;;
    --output-dir) shift 2 ;;
    *) shift ;;
  esac
done

echo "=== Julia's Agent — Science of AI/ML ==="
echo "Task: $TASK_DIR"

pip install --quiet --upgrade pip
pip install --quiet anthropic numpy pandas Pillow scikit-learn tensorflow-cpu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$SCRIPT_DIR/agent.py" --task_dir "$TASK_DIR" --output "$OUTPUT"
