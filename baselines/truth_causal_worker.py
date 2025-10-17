import sys
from pathlib import Path
import json
sys.path.append("./")
from scripts.framework.TestSearchSpace import TestSearchSpace
from scripts.framework.SearchSpace import *
from scripts.framework.framework_utils import *
import argparse
from scripts.framework.causal import compute_causal_effect,warper_get_thresholds,data_causal_score


def main(dataset:str, worker_id: int, total_workers: int,use_case: str):
    project_root = Path(__file__).resolve().parents[1]
    exp_config = json.load(open(project_root / 'data' / dataset / f'{dataset}-causal.json'))
    treatment = exp_config['treatment']
    attributes = list(exp_config['attributes'].keys())
    outcome = exp_config['target']
    attributes= [attr for attr in attributes if attr != treatment] #the confounding attributes
    raw_data = load_raw_data(str(project_root), exp_config)
    space_dict = {attr: TestSearchSpace(pd.read_csv(project_root / 'data' / dataset / 'scored_attributes' / f'{attr}.csv'), attr)
                  for attr in attributes}

    all_candidates = [list(c) for c in itertools.product(*[space_dict[attr].candidates for attr in attributes])]
    #all_candidates= all_candidates[:3]
    total = len(all_candidates)
    slice_size = total // total_workers
    start = worker_id * slice_size
    end = total if worker_id == total_workers - 1 else (worker_id + 1) * slice_size
    my_candidates = all_candidates[start:end]

    print(f"Worker {worker_id}: processing {len(my_candidates)} candidates (indices {start} to {end})")

    f_cols = [f"{attr}_ID" for attr in attributes] + [f"{attr}_bins" for attr in attributes] + ['utility', 'l2_norm', 'KLDiv', 'gpt_semantics']
    f_data = []
    if use_case == "known":
        unbinned_pre_utility= compute_causal_effect(raw_data, treatment, outcome, confounders=attributes) #this is the "real" ATE
    else: # use_case == "unknown":
        data = raw_data.copy()
        lower_tau, higher_tau,unbinned_correlations= warper_get_thresholds(data,epsilon="Thirds")
        unbinned_pre_utility = [lower_tau , higher_tau,unbinned_correlations]
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
        binned_data = apply_binning(data, attrs_bins)
        utility = data_causal_score(binned_data, outcome, treatment,attributes,use_case,unbinned_pre_utility)
        f_data.append(IDs + bins_ls + [utility, l2_norm/len(attributes), KLDiv/len(attributes), gpt_semantics/len(attributes)])
        print(f"Worker {worker_id}: {i+1}/{len(my_candidates)} done")
    print(f"Worker {worker_id}: finished processing candidates")
    df = pd.DataFrame(f_data, columns=f_cols)
    print(f"Results: {df} and the utility is {df['utility']}")

    output_dir = project_root / "truth" / dataset
    # If the directory does not exist, create it
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = project_root / "truth" / dataset / f"{dataset}.multi_attrs.causal.worker{worker_id}.csv"
    df.to_csv(output_path, index=False)
    print(f"Worker {worker_id}: saved results to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run causal utility worker")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--worker_id', type=int, required=True, help='ID of the current worker')
    parser.add_argument('--total_workers', type=int, required=True, help='Total number of workers')
    parser.add_argument('--use_case', type=str, required=True, help='known or unknown ATE')
    args = parser.parse_args()
    main(args.dataset, args.worker_id, args.total_workers, args.use_case)


