#!/bin/sh
set -eu

TASK_DIR=""
OUTPUT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --task)
      TASK_DIR="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --output-dir)
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  PYTHON_BIN=python3
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
"$PYTHON_BIN" "$SCRIPT_DIR/src/autoresearch_baseline.py" "$TASK_DIR" "$OUTPUT"
