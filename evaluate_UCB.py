"""
Evaluate UCB
"""
import argparse
import random
from datetime import datetime
from scripts.framework.UCB import *
import warnings
warnings.filterwarnings('ignore')
from scripts.framework.TestSearchSpace import TestSearchSpace

def main():
    project_root = os.path.abspath(__file__).rsplit('/', 1)[0]  # Get the project root directory
    print("Project root:", project_root)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=project_root, help="Project root.")
    parser.add_argument('--dataset', type=str, default='titanic', help='Dataset name')
    parser.add_argument('--use_case', type=str, default='visualization', choices=['visualization', 'causal_inference'], help='Use case for the evaluation')
    parser.add_argument('--imputer', type=str, default='KNN', choices=['', 'KNN', 'Simple', 'Iterative'], help='Imputer to use for the imputation use case')
    parser.add_argument('--semantic_metric', type=str, default='l2_norm', help='Semantic metric to use')
    parser.add_argument('--num_iterations', type=int, default=10, help='Number of iterations for the evaluation')
    parser.add_argument('--alpha_list', type=int, nargs='+', default=[2], help='Exploration parameter for UCB')
    parser.add_argument('--p_list', type=float, nargs='+', default=[100, 200, 400, 1000, 2000], help='p values for UCB') # [10, 20, 30, 40, 50, 60] for viz
    #parser.add_argument('--t_list', type=float, nargs='+', default=[0.3, 0.4, 0.5, 0.6, 0.7], help='t values for clustering')
    parser.add_argument('--missing_data_fraction', type=float, default=0.3, help='Fraction of missing data for imputation use case')
    parser.add_argument("--causal_use_case", type=str, default=None, choices=['known','unknown'], help="Causal inference use case (e.g., 'known', 'unknown')")
    args = parser.parse_args()

    project_root = args.project_root
    dataset = args.dataset
    use_case = args.use_case
    imputer = args.imputer
    semantic_metric = args.semantic_metric
    num_iterations = args.num_iterations
    alpha_list = args.alpha_list
    p_list = args.p_list
    #t_list = args.t_list
    missing_data_fraction = args.missing_data_fraction
    print("Current project_root:", project_root)

    if use_case == 'causal_inference' :
        exp_config = json.load(open(os.path.join(project_root, 'data', dataset, f'{dataset}-causal.json')))
    else: exp_config = json.load(open(os.path.join(project_root, 'data', dataset, f'{dataset}.json')))
    attributes = list(exp_config['attributes'].keys())
    gold_standard=exp_config['attributes']
    y_col = exp_config['target']
    # Load raw data
    raw_data = load_raw_data(project_root, exp_config, use_case=use_case, missing_data_fraction=missing_data_fraction)
    
    # Create search space
    search_space = {}
    for attr in attributes:
        attr_df = pd.read_csv(os.path.join(project_root, 'data', dataset, 'scored_attributes', f'{attr}.csv'))
        search_space[attr] = TestSearchSpace(attr_df, attribute=attr)

    # Load ground truth data
    # TODO: depending on the benchmark, 'anova' sometimes
    if use_case == 'modeling':
        utility_function = 'DecisionTreeClassifier'  # or 'RandomForestClassifier'
    elif use_case == 'imputation':
        utility_function = f'{imputer}Imputer.{missing_data_fraction}'  # or 'KNNImputer', 'SimpleImputer'
    elif use_case == 'causal_inference':
        utility_function = 'causal'
        treatment = exp_config['treatment']
        confounders = [attr for attr in attributes if attr != treatment]
        unbinned_utility_args = dict(causal_case=args.causal_use_case, data=raw_data,
                                     known_args=(treatment, y_col, confounders) if args.causal_use_case == "known" else None)
        unbinned_pre_utility = get_unbbined_data_causal_utility(**unbinned_utility_args)

    gt_filepath = os.path.join(project_root, 'truth', f'{dataset}.multi_attrs.{utility_function}.{semantic_metric}.csv')
    if not os.path.exists(gt_filepath):
        raise FileNotFoundError(f"Ground truth data not found at {gt_filepath}. Please ensure the file exists.")
    gt_data = pd.read_csv(gt_filepath)

    utility_function = 'spearman'

    # Prep to record results
    date_today = datetime.today().strftime('%Y_%m_%d_%H_%M')
    result_dst_dir = os.path.join(project_root, 'testresults', f"UCB.{utility_function}.{semantic_metric}")
    os.makedirs(result_dst_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dst_dir, "cached_results"), exist_ok=True)

    for attr in attributes:
        # Load ground truth data for the attribute
        print(f"Processing attribute: {attr}")
        gt_filepath = os.path.join(project_root, 'truth', 'single_attrs', f'{dataset}.{attr}.{utility_function}.csv')
        if not os.path.exists(gt_filepath):
            raise FileNotFoundError(f"Ground truth data not found at {gt_filepath}. Please ensure the file exists.")
        gt_data = pd.read_csv(gt_filepath)
        # Prepare to record results
        columns = ['cached_ID', 't', 'GD mean', 'GD median', 'GD std',
               'IGD mean', 'IGD median', 'IGD std', 'AHD mean', 'AHD median', 'AHD std']
        f_results = pd.DataFrame(columns=columns)

        for alpha in alpha_list:
            for p in p_list:
                #for t in t_list:
                    print(f"Running UCB with alpha={alpha}, p={p}")
                    cached_ID = random.randrange(100000, 1000000)
                    method = UCB(
                        gt_data=gt_data,
                        data=raw_data,
                        alpha=alpha,
                        n_samples=p,
                        #p=p,
                        y_col=y_col,
                        search_space=search_space[attr],
                        gold_standard=gold_standard,
                        semantic_metric=semantic_metric,
                        use_case=use_case,
                        t=0.4,
                        treatment_attr=treatment if use_case == 'causal_inference' else None,
                        confounders=confounders if use_case == 'causal_inference' else None,
                        causal_case=args.causal_use_case,
                        result_dst_dir=result_dst_dir,
                        debug=False
                    )
                    results, cached_results = method.run(cached_ID=cached_ID, n_runs=num_iterations)
                    f_results = pd.concat([f_results, pd.DataFrame([results])], ignore_index=True)
                    print(results)
                    # Write the cached results to a JSON file
                    cached_results_path = os.path.join(result_dst_dir, "cached_results", f"{dataset}.{attr}.{date_today}.{cached_ID}.json")
                    with open(cached_results_path, 'w') as f:
                        json.dump(cached_results, f, indent=4)
                    print(f"Cached detailed results saved to {cached_results_path}")
        # Save the results to a CSV file
        f_results.to_csv(os.path.join(result_dst_dir, f"{dataset}.{attr}.{date_today}.csv"), index=False)
        print(f"Results saved to {os.path.join(result_dst_dir, f'{dataset}.{attr}.{date_today}.csv')}")
        #break

if __name__ == "__main__":
    main()  # Adjust parameters as needed