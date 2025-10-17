import sys
import os
import json
import random
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.append("./")
from scripts.framework.discretizers import *
from scripts.framework.SearchSpace import *
from scripts.framework.utils import *
from scripts.framework.UCB import *
from scripts.framework.MCC import *
from scripts.framework.framework_utils import *
import argparse
from scripts.framework.TestSearchSpace import TestSearchSpace
# python3 baselines/random_sampling.py --dataset lalonde --use_case "causal_inference" --causal_use_case known
from baselines.exhaustive_modeling import explainable_modeling_multi_attrs

def get_semantic_score(candidate, metric):
    if metric == 'l2_norm':
        return candidate.l2_norm
    elif metric == 'KLDiv':
        return candidate.KLDiv
    elif metric == 'gpt_semantics':
        return candidate.gpt_semantics / 4  # Normalize GPT score to [0, 1]
    else:
        raise ValueError(f"Unknown semantic metric: {metric}")
        

def main():
    ppath = sys.path[0] + '/../'     #project_root = Path(__file__).resolve().parents[1]
    # dataset = 'pima'
    # use_case = 'imputation'  # 'modeling', 'imputation'
    # semantic_metric = 'gpt_semantics'
    # imputer = 'KNN'  # 'KNN', 'Simple', 'Iterative'
    # missing_data_fraction = 0.3  # fraction of data to be used for imputation
    # rounds = 10
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'diabetes')")
    parser.add_argument("--use_case", type=str, required=True, help="Task name (e.g., 'causal inference')",choices=['modeling', 'imputation','causal_inference', 'visualization'])
    parser.add_argument("--semantic_metric", type=str, default='gpt_semantics', help="Semantic metric to use (e.g., 'gpt_semantics')")
    parser.add_argument("--imputer", type=str, default='KNN', help="Imputer to use (e.g., 'KNN', 'Simple', 'Iterative')")
    parser.add_argument("--missing_data_fraction", type=float, default=0.3, help="Fraction of data to be used for imputation")
    parser.add_argument("--rounds", type=int, default=20, help="Number of rounds for random sampling")
    parser.add_argument("--causal_use_case", type=str, default='unknown', choices=['known','unknown'], help="Causal inference use case (e.g., 'known', 'unknown')")
    parser.add_argument("--budget_list", type=int, nargs='+', default=[100, 200, 400, 1000, 2000], help="List of budget values to evaluate")

    args = parser.parse_args()

    dataset = args.dataset
    use_case = args.use_case
    semantic_metric = args.semantic_metric
    rounds = args.rounds
    if use_case == 'imputation':
        imputer = args.imputer
        missing_data_fraction = args.missing_data_fraction
        utility_function = f'{imputer}Imputer.{missing_data_fraction}'
    if use_case == 'modeling':
        utility_function = 'DecisionTreeClassifier'
    # read json file
    if use_case == 'causal_inference' :
        exp_config = json.load(open(os.path.join(ppath, 'data', dataset, f'{dataset}-causal.json')))
    else: exp_config = json.load(open(os.path.join(ppath, 'data', dataset, f'{dataset}.json')))
    
    if use_case == "visualization":
        utility_function = 'spearman'
        attributes = [exp_config['attribute_viz']]
        y_col = exp_config['target_viz']
    else:
        attributes = exp_config['attributes'].keys()
        y_col = exp_config['target']

    load_data_args= {
        'project_root': ppath,
        'exp_config': exp_config,
        'use_case': use_case,
        'missing_data_fraction': missing_data_fraction if use_case == 'imputation' else None,
    }
    raw_data = load_raw_data(**load_data_args)

    if use_case == 'causal_inference':
        causal_case = args.causal_use_case
        utility_function = 'causal'
        treatment = exp_config['treatment']
        confounders = [attr for attr in attributes if attr != treatment]
        unbinned_utility_args = dict(causal_case=causal_case, data=raw_data, known_args=(treatment, y_col, confounders) if causal_case == "known" else None)
        unbinned_pre_utility = get_unbbined_data_causal_utility(**unbinned_utility_args)

    # Load ground truth pareto front
    if use_case == 'visualization':
        gt_data = pd.read_csv(os.path.join(ppath, 'truth', 'single_attrs', f'{args.dataset}.{attributes[0]}.{utility_function}.csv'))
    else:
        gt_data = pd.read_csv(os.path.join(ppath, 'truth', f'{args.dataset}.multi_attrs.{utility_function}.{args.semantic_metric}.csv'))
    datapoints = [np.array(gt_data[args.semantic_metric].values), np.array(gt_data['utility'].values)]
    lst = compute_pareto_front(datapoints)
    gt_data['Estimated'] = 0
    gt_data.loc[lst, 'Estimated'] = 1
    gt_data = gt_data[gt_data['Estimated'] == 1].drop_duplicates(subset=[args.semantic_metric, 'utility'])
    gt_points = np.array([gt_data[args.semantic_metric].values / 4, gt_data['utility'].values]).T
    print(f"Ground truth points: {gt_points}")
    gd = GD(gt_points)
    igd = IGD(gt_points)

    date_today = datetime.today().strftime('%Y_%m_%d_%H_%M')
    result_dst_dir = os.path.join(ppath, 'testresults', f"random.{utility_function}.{args.semantic_metric}")
    os.makedirs(result_dst_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dst_dir, 'cached_results'), exist_ok=True)

    columns = ['n_samples', 'GD mean', 'GD median', 'GD std',
               'IGD mean', 'IGD median', 'IGD std', 'AHD mean', 'AHD median', 'AHD std']
    f_results = pd.DataFrame(columns=columns)

    space_list = []
    for attr in attributes:
        data = pd.read_csv(os.path.join(ppath, 'data', args.dataset, 'scored_attributes', f'{attr}.csv'))
        ss = TestSearchSpace(data, attr)
        print(f'Attribute: {attr}', len(ss.candidates))
        space_list.append(ss)
    strategy_space = StrategySpace(space_list, gpt_threshold=0)
    search_space = strategy_space
    #budget = int(len(cluster_assignments) * p)
    #budget = 471 # 583 for pima, 471 for titanic

    cached_results = {budget: {} for budget in args.budget_list}
    for budget in args.budget_list:
        budget = min(budget, len(search_space.candidates))
        random_gd = []
        random_igd = []
        random_ahd = []
        gd= GD(gt_points)
        igd = IGD(gt_points)
        for round in range(rounds):
            print(f"Round {round + 1}/{rounds}")
            sampled_partitions = random.sample(search_space.candidates, budget)

            for strategy in sampled_partitions:
                data_i = raw_data.copy()
                partition_dict = {}
                for partition in strategy.partition_list:
                    partition_dict[partition.attribute] = partition.bins
                if use_case == 'modeling':
                    strategy.utility = explainable_modeling_using_strategy(data_i, y_col, strategy)
                elif use_case == 'imputation':
                    strategy.utility = data_imputation_multi_attrs(data_i, y_col, partition_dict)
                elif use_case == 'causal_inference':
                    strategy.utility = causal_inference_utility(data_i, y_col, partition_dict,
                                                                        confounders, treatment,
                                                                        causal_case, unbinned_pre_utility,
                                                                        after_bin_flag=False)
                elif use_case == 'visualization':
                    strategy.utility = visualization_spearmanr(data_i, y_col, partition_dict)

            explored_points, random_pareto_points, est_points_df = get_pareto_front(sampled_partitions, semantic_metric)
            est_points_df = est_points_df[est_points_df["Pareto"] == 1]
            # remove duplicates the pareto points based on Semantic and Utility
            est_points_df = est_points_df.drop_duplicates(subset=['Semantic', 'Utility'])
            if semantic_metric == 'gpt_semantics':
                est_points_df['Semantic'] = est_points_df['Semantic'] / 4  # Normalize GPT score to [0, 1]
            random_points = [np.array(est_points_df['Semantic'].values), np.array(est_points_df['Utility'].values)]
            
            #random_distances.append(average_distance(np.array(points).T, random_pareto_points))
            estimated_points = np.array(random_points).T
            print(estimated_points)
            gd_i = gd(estimated_points)
            igd_i = igd(estimated_points)
            ahd_i = Evaluator.average_hausdorff_distance(gd_i, igd_i, mode='max')
            random_gd.append(gd_i)
            random_igd.append(igd_i)
            random_ahd.append(ahd_i)
            if budget not in cached_results:
                cached_results[budget] = {}
            cached_results[budget][round] = {
                'estimated_P': estimated_points.tolist(),
                'gd': gd_i,
                'igd': igd_i,
                'ahd': ahd_i
            }

        result = {
            'n_samples': budget,
            'GD mean': np.mean(random_gd), 'GD median': np.median(random_gd), 'GD std': np.std(random_gd),
            'IGD mean': np.mean(random_igd), 'IGD median': np.median(random_igd), 'IGD std': np.std(random_igd),
            'AHD mean': np.mean(random_ahd), 'AHD median': np.median(random_ahd), 'AHD std': np.std(random_ahd)
        }
        f_results = pd.concat([f_results, pd.DataFrame([result])], ignore_index=True)

    # Write the cached results to a JSON file
    cached_results_path = os.path.join(result_dst_dir, "cached_results", f"{dataset}.{date_today}.json")
    with open(cached_results_path, 'w') as f:
        json.dump(cached_results, f, indent=4)
    print(f"Cached detailed results saved to {cached_results_path}")
    result_path = os.path.join(result_dst_dir, f"{args.dataset}.{date_today}.csv")
    f_results.to_csv(result_path, index=False)
    print(f"Results saved to {result_path}")

if __name__ == '__main__':
    main()
