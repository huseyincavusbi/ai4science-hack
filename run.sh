#!/bin/bash
# ============================================================
# AI4Science Hackathon — Science of AI/ML Track
# Codabench entry point. Called as: bash run.sh
# Expects: task files in current directory
# Produces: predictions.csv in current directory
# ============================================================
 
set -e  # exit on first error
 
TASK_DIR="${1:-.}"  # default to current directory if no arg passed
 
echo "============================================"
echo " AI4Science Agent — Science of AI/ML Track"
echo " Task dir: $TASK_DIR"
echo "============================================"
 
# ── Install dependencies ──────────────────────────────────────────────────────
echo "[setup] Installing dependencies..."
 
pip install --quiet --upgrade pip
pip install --quiet \
    anthropic \
    numpy \
    pandas \
    Pillow \
    scikit-learn \
    tensorflow-cpu
 
echo "[setup] Dependencies installed."
 
# ── Run agent ─────────────────────────────────────────────────────────────────
echo "[agent] Starting agent..."
 
python agent.py --task_dir "$TASK_DIR"
 
# ── Verify output ─────────────────────────────────────────────────────────────
if [ -f "$TASK_DIR/predictions.csv" ]; then
    echo "[done] predictions.csv produced successfully."
    echo "[done] Row count: $(wc -l < "$TASK_DIR/predictions.csv")"
else
    echo "[error] predictions.csv not found. Agent failed."
    exit 1
fi
