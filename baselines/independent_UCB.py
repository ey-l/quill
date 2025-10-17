import sys
import os
import argparse
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
import json
import random

sys.path.append("./")
# from scripts.framework.discretizers import *
from scripts.framework.SearchSpace import *
# from scripts.framework.utils import *
from scripts.framework.UCB import *
from scripts.framework.framework_utils import *
from baselines.exhaustive_modeling import explainable_modeling_multi_attrs
from scripts.framework.TestSearchSpace import TestSearchSpace


def get_semantic_score(candidate, metric: SyntaxWarning):
    if metric == 'l2_norm':
        return candidate.l2_norm
    elif metric == 'KLDiv':
        return candidate.KLDiv
    elif metric == 'gpt_semantics':
        return candidate.gpt_semantics
    else:
        raise ValueError(f"Unknown semantic metric: {metric}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pima')
    parser.add_argument('--use_case', type=str, default='modeling', choices=['modeling', 'imputation', 'causal_inference'])
    parser.add_argument('--imputer', type=str, default='KNN')
    parser.add_argument('--missing_data_fraction', type=float, default=0.3)
    parser.add_argument('--semantic_metric', type=str, default='gpt_semantics')
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--t', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=2)
    parser.add_argument("--causal_use_case", type=str, default='unknown', choices=['known','unknown'], help="Causal inference use case (e.g., 'known', 'unknown')")

    args = parser.parse_args()

    # ppath = os.path.abspath(__file__).rsplit('/', 2)[0]
    ppath = sys.path[0] + '/../'
    dataset = args.dataset
    use_case = args.use_case
    imputer = args.imputer
    missing_data_fraction = args.missing_data_fraction
    semantic_metric = args.semantic_metric
    rounds = args.rounds
    t = args.t
    alpha = args.alpha

    if use_case == 'causal_inference' :
        exp_config = json.load(open(os.path.join(ppath, 'data', dataset, f'{dataset}-causal.json')))
    else: exp_config = json.load(open(os.path.join(ppath, 'data', dataset, f'{dataset}.json')))
    attributes = exp_config['attributes'].keys()
    gold_standard = exp_config['attributes']
    y_col = exp_config['target']
    raw_data = load_raw_data(ppath, exp_config, use_case=use_case, missing_data_fraction=missing_data_fraction)
    

    if use_case == 'modeling':
        utility_function = 'DecisionTreeClassifier'
    elif use_case == 'imputation':
        utility_function = f'{imputer}Imputer.{missing_data_fraction}'
    elif use_case == 'causal_inference':
        causal_case = args.causal_use_case
        utility_function = 'causal'
        treatment = exp_config['treatment']
        confounders = [attr for attr in attributes if attr != treatment]
        unbinned_utility_args = dict(causal_case=causal_case, data=raw_data,
                                     known_args=(treatment, y_col, confounders) if causal_case == "known" else None)
        unbinned_pre_utility = get_unbbined_data_causal_utility(**unbinned_utility_args)

    gt_data = pd.read_csv(os.path.join(ppath, 'truth', f'{dataset}.multi_attrs.{utility_function}.{semantic_metric}.csv'))
    if args.semantic_metric == 'gpt_semantics':
        gt_data[args.semantic_metric] = gt_data[args.semantic_metric] / 4  # Normalize GPT score to [0, 1]
    gt_points = np.array([gt_data[semantic_metric].values, gt_data['utility'].values]).T
    print("Ground truth points:", gt_points)
    gd = GD(gt_points)
    igd = IGD(gt_points)

    date_today = datetime.today().strftime('%Y_%m_%d_%H_%M')
    result_dst_dir = os.path.join(ppath, 'testresults', f"ind_UCB.{utility_function}.{semantic_metric}")
    os.makedirs(result_dst_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dst_dir, "cached_results"), exist_ok=True)
    columns = ['n_samples', 'alpha', 'p', 't', 'GD mean', 'GD median', 'GD std',
               'IGD mean', 'IGD median', 'IGD std', 'AHD mean', 'AHD median', 'AHD std']
    f_results = pd.DataFrame(columns=columns)

    search_space = {}
    for attr in attributes:
        data = pd.read_csv(os.path.join(ppath, 'data', dataset, 'scored_attributes', f'{attr}.csv'))
        search_space[attr] = TestSearchSpace(data, attr, decimal_place=8)

    t_dict = {}
    p_list = [0.2, 0.5, 0.8, 1.0]
    cached_results = {p: {} for p in p_list}
    for p in p_list:
        gd_list = []
        igd_list = []
        ahd_list = []
        numbers_samples = []
        for n_round in range(rounds):
            n_strategies_tried = 0
            est_partition_dicts = {attr: [] for attr in attributes}

            for attr in attributes:
                data_i = raw_data.copy()
                cached_ID = random.randrange(100000, 1000000)
                method = UCB(gt_data=gt_data, data=data_i, alpha=alpha, p=p, y_col=y_col,
                            search_space=search_space[attr], gold_standard=gold_standard,
                            semantic_metric=semantic_metric, use_case=use_case, t=t,
                            treatment_attr=treatment if use_case == 'causal_inference' else None,
                            confounders=confounders if use_case == 'causal_inference' else None,
                            causal_case=args.causal_use_case,
                            result_dst_dir=result_dst_dir, debug=False)
                results, temp_cached_results = method.run(cached_ID=cached_ID, n_runs=1)
                n_strategies_tried += len(search_space[attr].candidates) * p
                t_dict[attr] = method.t
                est_partition_dicts[attr] = temp_cached_results[0]['estimated_partitions']

            candidates_tuples = list(itertools.product(*[est_partition_dicts[attr] for attr in attributes]))
            partition_dicts = []
            for candidate_tuple in candidates_tuples:
                semantics = 0
                partition_dict = {"Partition": {}}
                for i, attr in enumerate(attributes):
                    candidate = candidate_tuple[i]
                    semantics += candidate["__semantic__"]
                    partition_dict["Partition"][attr] = candidate[attr]
                semantics /= len(candidate_tuple)
                data_i = raw_data.copy()
                if use_case == 'modeling':
                    utility = explainable_modeling_multi_attrs(data_i, y_col, partition_dict["Partition"])
                elif use_case == 'imputation':
                    utility = data_imputation_multi_attrs(data_i, y_col, partition_dict["Partition"])
                elif use_case == 'causal_inference':
                    utility = causal_inference_utility(data_i, y_col, partition_dict["Partition"],
                                                       confounders, treatment, causal_case,
                                                       unbinned_pre_utility, after_bin_flag=False)
                partition_dict["Semantic"] = semantics
                partition_dict["Utility"] = utility
                partition_dicts.append(partition_dict)

            semantic_scores = [point["Semantic"] for point in partition_dicts]
            utility_scores = [point["Utility"] for point in partition_dicts]
            datapoints = np.array([semantic_scores, utility_scores])
            lst = compute_pareto_front(datapoints)
            partition_dicts = [partition_dicts[i] for i in lst]
            if semantic_metric == 'gpt_semantics':
                for partition in partition_dicts:
                    partition['Semantic'] = partition['Semantic'] / 4  # Normalize GPT score to [0, 1]
            estimated_points = np.array([[partition["Semantic"], partition["Utility"]] for partition in partition_dicts])
            estimated_points = np.unique(estimated_points, axis=0)
            print("Estimated:", estimated_points)
            gd_i = gd(estimated_points)
            igd_i = igd(estimated_points)
            ahd_i = Evaluator.average_hausdorff_distance(gd_i, igd_i, mode='max')
            gd_list.append(gd_i)
            igd_list.append(igd_i)
            ahd_list.append(ahd_i)
            numbers_samples.append(n_strategies_tried)
            cached_results[p][n_round] = {
                'alpha': args.alpha,
                't': t_dict,
                'estimated_P': estimated_points.tolist(),
                'estimated_partitions': partition_dicts,
                'gd': gd_i,
                'igd': igd_i,
                'ahd': ahd_i,
                'n_strategies_tried': n_strategies_tried
            }

        # Print the mean, median, std of the gd and igd
        print(f"GD mean: {np.mean(gd_list)}; median: {np.median(gd_list)}; std: {np.std(gd_list)}")
        print(f"IGD mean: {np.mean(igd_list)}; median: {np.median(igd_list)}; std: {np.std(igd_list)}")
        print(f"AHD mean: {np.mean(ahd_list)}; median: {np.median(ahd_list)}; std: {np.std(ahd_list)}")
        print(f"Number of strategies tried: {np.mean(numbers_samples)}; std: {np.std(numbers_samples)}")
        
        result = {
            'n_samples': int(np.mean(numbers_samples)),
            'alpha': alpha,
            'p': p,
            't': t_dict,
            'GD mean': np.mean(gd_list),
            'GD median': np.median(gd_list),
            'GD std': np.std(gd_list),
            'IGD mean': np.mean(igd_list),
            'IGD median': np.median(igd_list),
            'IGD std': np.std(igd_list),
            'AHD mean': np.mean(ahd_list),
            'AHD median': np.median(ahd_list),
            'AHD std': np.std(ahd_list)
        }
        f_results = pd.concat([f_results, pd.DataFrame([result])], ignore_index=True)

    # Write the cached results to a JSON file
    cached_results_path = os.path.join(result_dst_dir, "cached_results", f"{dataset}.{date_today}.json")
    with open(cached_results_path, 'w') as f:
        json.dump(cached_results, f, indent=4)
    print(f"Cached detailed results saved to {cached_results_path}")
    f_results.to_csv(os.path.join(result_dst_dir, f"{dataset}.{date_today}.csv"), index=False)
    print(f"Results saved to {os.path.join(result_dst_dir, f'{dataset}.{date_today}.csv')}")

if __name__ == '__main__':
    main()
