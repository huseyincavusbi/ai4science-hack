# AI4Science Hackathon — Science of AI/ML Track

Autonomous agent for domain generalization tasks. Diagnoses spurious correlations from metadata, selects a robust strategy, generates and executes training code, and produces `predictions.csv`.

## Structure

```
├── run.sh                  # Codabench entry point
├── agent.py                # Agent scaffold (LLM loop + code execution)
├── prompts/
│   └── system_prompt.txt   # Agent reasoning instructions
└── README.md
```

## How it works

1. `run.sh` installs dependencies and calls `agent.py`
2. `agent.py` reads `task.json` + `metadata.json`, builds a user message, and calls Claude
3. Claude diagnoses the spurious correlation, selects a strategy, and writes a Python training script
4. The script is executed; if it fails, the error is fed back to Claude for a retry (up to 3 attempts)
5. `predictions.csv` is verified and written to the task directory

## Strategy selection

| Scenario | Strategy |
|---|---|
| Color/texture spurious feature (image data) | Strip spurious channel, train on grayscale |
| Unknown spurious feature (tabular) | Invariant Risk Minimization (IRM) |
| Dominant environment correlation | Environment reweighting |
| No detectable spurious correlation | Naive ERM (fallback) |

## Setup

```bash
export ANTHROPIC_API_KEY=your_key_here
pip install anthropic numpy pandas Pillow scikit-learn tensorflow-cpu
```

## Local test run

```bash
bash run.sh /path/to/task_directory
```
