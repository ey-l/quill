import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
dataset = "crop"
truth_dir = project_root / "truth" / dataset

dfs = []
for i in range(10):
    path = truth_dir / f"{dataset}.multi_attrs.DecisionTreeClassifier.worker{i}.csv"
    if path.exists():
        dfs.append(pd.read_csv(path))
    else:
        print(f"Warning: missing {path}")

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv(truth_dir / f"{dataset}.multi_attrs.DecisionTreeClassifier.csv", index=False)
print("Merged all worker outputs into final CSV.")
