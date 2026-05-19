import csv
import json
import math
import sys
from pathlib import Path


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def solve_linear_system(a, b):
    n = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            aug[col][col] += 1e-6
            pivot = col
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        if abs(scale) < 1e-12:
            continue
        for j in range(col, n + 1):
            aug[col][j] /= scale
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def main():
    task_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    meta = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    target = meta.get("target", {}).get("name")
    train = read_csv(task_dir / meta.get("input_files", {}).get("train", "data/train.csv"))
    test = read_csv(task_dir / meta.get("input_files", {}).get("test", "data/test.csv"))

    if target and target in train[0]:
        y_source = target
    elif "label" in train[0]:
        y_source = "label"
    elif "outcome" in train[0]:
        y_source = "outcome"
    else:
        y_source = None

    test_columns = set(test[0].keys()) if test else set()
    feature_names = [
        name for name in train[0].keys()
        if name in test_columns and name not in {"id", target, y_source}
    ]
    feature_names = [
        name for name in feature_names
        if all(to_float(row.get(name)) is not None for row in train[:20])
    ]
    y = [float(row[y_source]) for row in train] if y_source else [0.0 for _ in train]
    means = {}
    scales = {}
    for name in feature_names:
        values = [float(row[name]) for row in train]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
        means[name] = mean
        scales[name] = math.sqrt(var) or 1.0

    def features(row):
        vals = [1.0]
        for name in feature_names:
            vals.append((float(row[name]) - means[name]) / scales[name])
        return vals

    x = [features(row) for row in train]
    p = len(x[0])
    xtx = [[0.0 for _ in range(p)] for _ in range(p)]
    xty = [0.0 for _ in range(p)]
    for row_x, row_y in zip(x, y):
        for i in range(p):
            xty[i] += row_x[i] * row_y
            for j in range(p):
                xtx[i][j] += row_x[i] * row_x[j]
    for i in range(1, p):
        xtx[i][i] += 0.2
    weights = solve_linear_system(xtx, xty)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prediction"])
        writer.writeheader()
        for row in test:
            pred = sum(w * v for w, v in zip(weights, features(row)))
            writer.writerow({"id": row["id"], "prediction": f"{pred:.8f}"})


if __name__ == "__main__":
    main()
