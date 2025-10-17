import sys, os, json, itertools, time
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Dict

sys.path.append("./")
from scripts.framework.discretizers import *
from scripts.framework.SearchSpace import *
from scripts.framework.utils import *
from scripts.framework.framework_utils import *
from scripts.framework.TestSearchSpace import TestSearchSpace

def explainable_modeling_multi_attrs(data, y_col, attrs_bins:Dict) -> float:
    for attr, bins in attrs_bins.items():
        data[attr + '.binned'] = pd.cut(data[attr], bins, labels=False).astype('float64')
        data = data.dropna(subset=[attr + '.binned'])

    X_cols = [col for col in data.columns if col != y_col and col not in list(attrs_bins.keys())]
    X = data[X_cols]
    y = data[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=0, max_depth=5)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

def main(worker_id: int, total_workers: int):
    project_root = Path(__file__).resolve().parents[1]
    dataset = 'bank'  # Change this to your dataset name
    exp_config = json.load(open(project_root / 'data' / dataset / f'{dataset}.json'))
    attributes = list(exp_config['attributes'].keys())
    y_col = exp_config['target']
    raw_data = load_raw_data(str(project_root), exp_config)

    space_dict = {attr: TestSearchSpace(pd.read_csv(project_root / 'data' / dataset / 'scored_attributes' / f'{attr}.csv'), attr)
                  for attr in attributes}

    all_candidates = [list(c) for c in itertools.product(*[space_dict[attr].candidates for attr in attributes])]
    total = len(all_candidates)
    slice_size = total // total_workers
    start = worker_id * slice_size
    end = total if worker_id == total_workers - 1 else (worker_id + 1) * slice_size
    my_candidates = all_candidates[start:end]

    print(f"Worker {worker_id}: processing {len(my_candidates)} candidates (indices {start} to {end})")

    f_cols = [f"{attr}_ID" for attr in attributes] + [f"{attr}_bins" for attr in attributes] + ['utility', 'l2_norm', 'KLDiv', 'gpt_semantics']
    f_data = []

    for i, strategy in enumerate(my_candidates):
        data = raw_data.copy()
        attrs_bins = {}
        IDs, bins_ls = [], []
        l2_norm, KLDiv, gpt_semantics = 0, 0, 0

        for j, attr in enumerate(attributes):
            candidate = strategy[j]
            attrs_bins[attr] = candidate.bins
            IDs.append(candidate.ID)
            bins_ls.append(candidate.bins)
            l2_norm += candidate.l2_norm
            KLDiv += candidate.KLDiv
            gpt_semantics += candidate.gpt_semantics

        utility = explainable_modeling_multi_attrs(data, y_col, attrs_bins)
        f_data.append(IDs + bins_ls + [utility, l2_norm/len(attributes), KLDiv/len(attributes), gpt_semantics/len(attributes)])
        print(f"Worker {worker_id}: {i+1}/{len(my_candidates)} done")

    df = pd.DataFrame(f_data, columns=f_cols)
    output_dir = project_root / "truth" / dataset
    # If the directory does not exist, create it
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = project_root / "truth" / dataset / f"{dataset}.multi_attrs.DecisionTreeClassifier.worker{worker_id}.csv"
    df.to_csv(output_path, index=False)
    print(f"Worker {worker_id}: saved results to {output_path}")

if __name__ == '__main__':
    worker_id = int(sys.argv[1])
    total_workers = int(sys.argv[2])
    main(worker_id, total_workers)