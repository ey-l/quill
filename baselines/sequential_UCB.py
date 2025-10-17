import sys
import os
import json
import itertools
import random
from datetime import datetime
import numpy as np
import pandas as pd
import argparse

sys.path.append("./")
from scripts.framework.TestSearchSpace import TestSearchSpace
# from scripts.framework.utils import *
from scripts.framework.UCB import *
from scripts.framework.framework_utils import *
from baselines.exhaustive_modeling import explainable_modeling_multi_attrs

SEMANTICS = ['l2_norm', 'KLDiv', 'gpt_semantics']

def get_semantic_score(candidate, metric):
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
    parser.add_argument('--use_case', type=str, choices=['modeling', 'imputation','causal_inference'], default='imputation')
    parser.add_argument('--imputer', type=str, default='KNN')
    parser.add_argument('--semantic_metric', type=str, choices=SEMANTICS, default='gpt_semantics')
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--t', type=float, default=0.0, help='Threshold for clustering. 0.0 means auto clustering')
    parser.add_argument('--alpha', type=float, default=2, help='Exploration parameter for UCB; does not affect quality performance of the method')
    parser.add_argument('--missing_data_fraction', type=float, default=0.3, help='Fraction of missing data for imputation use case')
    parser.add_argument("--causal_use_case", type=str, default='unknown', choices=['known','unknown'], help="Causal inference use case (e.g., 'known', 'unknown')")

    args = parser.parse_args()
    alpha = args.alpha
    dataset = args.dataset
    t = args.t
    semantic_metric = args.semantic_metric
    use_case = args.use_case
    if use_case == 'imputation':
        imputer = args.imputer



    ppath = sys.path[0] + '/../'     #project_root = Path(__file__).resolve().parents[1] # ppath = os.path.abspath(__file__).rsplit('/', 2)[0]
    if use_case == 'causal_inference' :
        exp_config = json.load(open(os.path.join(ppath, 'data', dataset, f'{dataset}-causal.json')))
    else: exp_config = json.load(open(os.path.join(ppath, 'data', dataset, f'{dataset}.json')))
    attributes = exp_config['attributes'].keys()
    gold_standard = exp_config['attributes']
    y_col = exp_config['target']
    raw_data = load_raw_data(ppath, exp_config, use_case=args.use_case, missing_data_fraction=0.3)
    if use_case == 'causal_inference':
        causal_case = args.causal_use_case
        treatment = exp_config['treatment']
        confounders = [attr for attr in attributes if attr != treatment]
        unbinned_utility_args = dict(causal_case=causal_case, data=raw_data,
                                     known_args=(treatment, y_col, confounders) if causal_case == "known" else None)
        unbinned_pre_utility = get_unbbined_data_causal_utility(**unbinned_utility_args)
    if __name__ == '__main__':
        utility_function = 'DecisionTreeClassifier' if args.use_case == 'modeling' else f'{args.imputer}Imputer.{args.missing_data_fraction}' if args.use_case == 'imputation' else "causal"
    gt_data = pd.read_csv(os.path.join(ppath, 'truth', f'{args.dataset}.multi_attrs.{utility_function}.{semantic_metric}.csv'))
    gt_df = gt_data.copy()
    datapoints = [np.array(gt_df[args.semantic_metric].values), np.array(gt_df['utility'].values)]
    lst = compute_pareto_front(datapoints)
    gt_df['Estimated'] = 0
    gt_df.loc[lst, 'Estimated'] = 1
    gt_df['Explored'] = 1
    gt_df = gt_df[gt_df['Estimated'] == 1].drop_duplicates(subset=[args.semantic_metric, 'utility'])
    if args.semantic_metric == 'gpt_semantics':
        gt_df[args.semantic_metric] = gt_df[args.semantic_metric] / 4  # Normalize GPT score to [0, 1]
    gt_points = np.array([gt_df[args.semantic_metric].values, gt_df['utility'].values]).T
    gd = GD(gt_points)
    igd = IGD(gt_points)

    date_today = datetime.today().strftime('%Y_%m_%d_%H_%M')
    result_dst_dir = os.path.join(ppath, 'testresults', f"seq_UCB.{utility_function}.{args.semantic_metric}")
    os.makedirs(result_dst_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dst_dir, 'cached_results'), exist_ok=True)

    columns = ['n_samples', 'alpha', 'p', 't', 'GD mean', 'GD median', 'GD std',
               'IGD mean', 'IGD median', 'IGD std', 'AHD mean', 'AHD median', 'AHD std']
    f_results = pd.DataFrame(columns=columns)

    search_space = {}
    for attr in attributes:
        data = pd.read_csv(os.path.join(ppath, 'data', args.dataset, 'scored_attributes', f'{attr}.csv'))
        search_space[attr] = TestSearchSpace(data, attr, decimal_place=8)

    permutations = list(itertools.permutations(attributes))
    t_dict = {}

    p_list = [0.2, 0.5, 0.8, 1.0]
    cached_results = {p: {} for p in p_list}
    for p in p_list:
        gd_list, igd_list, ahd_list, numbers_samples = [], [], [], []
        for n_round in range(args.rounds):
            # (Same inner loop structure as original — omitted here for brevity)
            # Fill in est_partitions, compute metrics, append to *_list
            n_strategies_tried = 0
            est_partitions = []
            for permutation in permutations:
            
                est_partition_dicts = []
                processed_attributes = []
                for n, attr in enumerate(permutation):
                    data_i = raw_data.copy()
                    print("====== Attribute:", attr, "======")
                    if len(processed_attributes) == 0:
                        cached_ID = random.randrange(100000, 1000000)
                        method = UCB(
                            gt_data=gt_data,
                            data=data_i,
                            alpha=alpha,
                            p=p,
                            y_col=y_col,
                            search_space=search_space[attr],
                            gold_standard=gold_standard,
                            semantic_metric=semantic_metric,
                            use_case=use_case,
                            t=t,
                            treatment_attr=treatment if use_case == 'causal_inference' else None,
                            confounders=confounders if use_case == 'causal_inference' else None,
                            causal_case=args.causal_use_case,
                            result_dst_dir=result_dst_dir,
                            debug=False
                        )
                        results, temp_cached_results = method.run(cached_ID=cached_ID, n_runs=1)
                        n_strategies_tried += len(search_space[attr].candidates) * p
                        t_dict[attr] = method.t
                        # Estimated Pareto points only
                        est_pareto_partitions = temp_cached_results[0]['estimated_partitions']
                        for est_partition in est_pareto_partitions:
                            est_partition_dicts.append({attr: est_partition})
                        processed_attributes.append(attr)
                    else:
                        new_partition_dicts = []
                        for partition_dict in est_partition_dicts:
                            data_i = raw_data.copy()
                            for attr2, partition in partition_dict.items():
                                data_i[attr2 + '.binned'] = pd.cut(data_i[attr2], bins=partition[attr2], labels=False)
                                data_i[attr2 + '.binned'] = data_i[attr2 + '.binned'].astype('float64')
                                data_i = data_i.dropna(subset=[attr2 + '.binned'])
                            cached_ID = random.randrange(100000, 1000000)
                            method = UCB(
                                gt_data=gt_data,
                                data=data_i,
                                alpha=alpha,
                                p=p,
                                y_col=y_col,
                                search_space=search_space[attr],
                                gold_standard=gold_standard,
                                semantic_metric=semantic_metric,
                                use_case=use_case,
                                t=t,
                                treatment_attr=treatment if use_case == 'causal_inference' else None,
                                confounders=confounders if use_case == 'causal_inference' else None,
                                causal_case=args.causal_use_case,
                                result_dst_dir=result_dst_dir,
                                debug=False
                            )
                            results, temp_cached_results = method.run(cached_ID=cached_ID, n_runs=1)
                            n_strategies_tried += len(search_space[attr].candidates) * p
                            t_dict[attr] = method.t
                            # Estimated Pareto points only
                            est_pareto_partitions = temp_cached_results[0]['estimated_partitions']
                            for est_partition in est_pareto_partitions:
                                new_partition_dict = partition_dict.copy()
                                new_partition_dict[attr] = est_partition
                                new_partition_dicts.append(new_partition_dict)
                        processed_attributes.append(attr)
                        est_partition_dicts = new_partition_dicts
                # extend the est_partitions with the new partition dicts
                est_partitions.extend(est_partition_dicts)
                
            # Exhausitve search for the pareto curve partitions
            utility_scores = []
            semantic_scores = []
            new_partition_dicts = []
            for partition_dict in est_partitions:
                new_partition_dict = {}
                semantic_score = 0
                for attr, partition in partition_dict.items():
                    new_partition_dict[attr] = partition[attr]
                    semantic_score += partition['__semantic__']
                semantic_score /= len(new_partition_dict)
                data_i = raw_data.copy()
                if use_case == 'modeling':
                    utility_score = explainable_modeling_multi_attrs(data_i, y_col, new_partition_dict)
                elif use_case == 'imputation':
                    utility_score = data_imputation_multi_attrs(data_i, y_col, new_partition_dict, imputer=imputer)
                elif use_case == 'causal_inference':
                    utility_score = causal_inference_utility(data_i, y_col, new_partition_dict,confounders, treatment,
                                                            causal_case, unbinned_pre_utility, after_bin_flag=False)
                #print(f"Partition: {new_partition_dict}, Semantic: {semantic_score}, Utility: {utility_score}")
                new_partition_dicts.append({'Partition': new_partition_dict, 'Semantic': semantic_score, 'Utility': utility_score, 'Explored': 1, 'Estimated': 0})
                utility_scores.append(utility_score)
                semantic_scores.append(semantic_score)
            
            datapoints = np.array([np.array(semantic_scores), np.array(utility_scores)])
            lst = compute_pareto_front(datapoints)
            #est_partitions = [new_partition_dicts[i] for i in lst]
            est_partitions = []
            # This is for demo purposes, we will set the Estimated flag to 1 for the estimated partitions
            if args.semantic_metric == 'gpt_semantics':
                for partition in new_partition_dicts:
                    partition['Semantic'] = partition['Semantic'] / 4  # Normalize GPT score to [0, 1]
            for i in lst:
                new_partition_dicts[i]['Estimated'] = 1
                est_partitions.append(new_partition_dicts[i])
            estimated_points = np.array([np.array([point["Semantic"] for point in est_partitions]),
                                np.array([point["Utility"] for point in est_partitions])])
            estimated_points = np.array(estimated_points).T
            estimated_points = np.unique(estimated_points, axis=0)
            print("Final Estimated of This Run:", estimated_points)
            gd_i = gd(estimated_points)
            igd_i = igd(estimated_points)
            ahd_i = Evaluator.average_hausdorff_distance(gd_i, igd_i, mode='max')
            gd_list.append(gd_i)
            igd_list.append(igd_i)
            ahd_list.append(ahd_i)
            print(f"Round {n_round + 1}:")
            print(f"GD: {gd_i}")
            print(f"IGD: {igd_i}")
            print(f"AHD: {ahd_i}")
            numbers_samples.append(n_strategies_tried)
            print(new_partition_dicts)
            cached_results[p][n_round] = {
                'alpha': args.alpha,
                't': t_dict,
                'estimated_P': estimated_points.tolist(),
                'estimated_partitions': est_partitions,
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
            'alpha': args.alpha,
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
            'AHD std': np.std(ahd_list),
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
