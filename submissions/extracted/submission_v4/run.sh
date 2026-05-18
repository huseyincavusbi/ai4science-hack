#!/bin/bash
set -eu

# -- Hackathon LiteLLM credentials ----------------------------------------
export LLM_API_KEY="sk-7NRx1SUb0hjL7-13RgBgZA"
export LLM_BASE_URL="https://hackathon.tiptreesystems.com/litellm"
export LLM_MODEL="bedrock-claude-opus"
export LLM_MAX_TURNS="60"
# -------------------------------------------------------------------------

TASK_DIR=""
OUTPUT=""
OUTPUT_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --task)       TASK_DIR="$2"; shift 2 ;;
    --output)     OUTPUT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) shift ;;
  esac
done

pip install -q anthropic numpy pandas scikit-learn torch torchvision 2>&1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/src/agent.py" \
  --task "$TASK_DIR" \
  --output "$OUTPUT" \
  --output-dir "$OUTPUT_DIR" \
  --submission-dir "$SCRIPT_DIR"
