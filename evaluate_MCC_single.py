"""
Evaluate MCC
"""
import argparse
import random
from datetime import datetime
from scripts.framework.MCC import *
from scripts.framework.TestSearchSpace import TestSearchSpace
import warnings
warnings.filterwarnings('ignore')

def main():
    project_root = os.path.abspath(__file__).rsplit('/', 1)[0]  # Get the project root directory
    print("Project root:", project_root)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=project_root, help="Project root.")
    parser.add_argument('--dataset', type=str, default='titanic', help='Dataset name')
    parser.add_argument('--use_case', type=str, default='visualization', choices=['visualization'], help='Use case for the evaluation')
    parser.add_argument('--imputer', type=str, default='KNN', choices=['', 'KNN', 'Simple', 'Iterative'], help='Imputer to use for the imputation use case')
    parser.add_argument('--semantic_metric', type=str, default='l2_norm', choices=['l2_norm', 'KLDiv', 'gpt_semantics'], help='Semantic metric to use')
    parser.add_argument('--update_mode', type=str, default='final-only', choices=['final-only', 'every-step'], help='Update mode for MCC')
    parser.add_argument('--num_iterations', type=int, default=20, help='Number of iterations for the evaluation')
    parser.add_argument('--training_cutoffs', type=int, nargs='+', default=[10, 20, 30, 40, 50, 60], help='Budget for the evaluation')
    parser.add_argument('--epsilon_list', type=float, nargs='+', default=[0.4, 0.5, 0.6], help='Epsilon values for the evaluation')
    parser.add_argument('--t', type=float, default=0, help='t value for clustering')
    parser.add_argument('--missing_data_fraction', type=float, default=0.3, help='Fraction of missing data for imputation use case')
    parser.add_argument('--epsilon_schedule', type=bool, default=True, help='Epsilon schedule for MCC')
    parser.add_argument('--ablation_variant', type=str, default="", choices=["", "dbscan", "no_tuning", "binned_values"], help='Ablation variant for MCC')
    args = parser.parse_args()

    project_root = args.project_root
    dataset = args.dataset
    use_case = args.use_case
    imputer = args.imputer
    semantic_metric = args.semantic_metric
    update_mode = args.update_mode
    num_iterations = args.num_iterations
    training_cutoffs = args.training_cutoffs
    epsilon_list = args.epsilon_list
    t = args.t
    ablation_variant = args.ablation_variant
    if ablation_variant == "dbscan":
        dbscan = True
    else: dbscan = False
    if ablation_variant == "binned_values":
        binned_values = True
    else: binned_values = False
    missing_data_fraction = args.missing_data_fraction
    epsilon_schedule = args.epsilon_schedule
    if epsilon_schedule: epsilon_list = [0] # Fixed schedule anyway
    print("Current project_root:", project_root)

    exp_config = json.load(open(os.path.join(project_root, 'data', dataset, f'{dataset}.json')))
    #attributes = list(exp_config['attributes'].keys())
    attributes = [exp_config['attribute_viz']]  # For single attribute evaluation
    gold_standard=exp_config['attributes']
    y_col = exp_config['target_viz']
    # Load raw data
    raw_data = load_raw_data(project_root, exp_config, use_case=use_case, missing_data_fraction=missing_data_fraction)
    
    # Create search space
    search_space = {}
    for attr in attributes:
        attr_df = pd.read_csv(os.path.join(project_root, 'data', dataset, 'scored_attributes', f'{attr}.csv'))
        search_space[attr] = TestSearchSpace(attr_df, attribute=attr)

    # Load ground truth data
    # TODO: depending on the benchmark, 'anova' sometimes
    utility_function = 'spearman'

    # Prep to record results
    date_today = datetime.today().strftime('%Y_%m_%d_%H_%M')
    result_dst_dir = os.path.join(project_root, 'testresults', f"MCC{ablation_variant}.{utility_function}.{semantic_metric}")
    os.makedirs(result_dst_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dst_dir, "cached_results"), exist_ok=True)

    
    for attr in attributes:
        # Load ground truth data for the attribute
        print(f"Processing attribute: {attr}")
        gt_filepath = os.path.join(project_root, 'truth', 'single_attrs', f'{dataset}.{attr}.{utility_function}.csv')
        if not os.path.exists(gt_filepath):
            raise FileNotFoundError(f"Ground truth data not found at {gt_filepath}. Please ensure the file exists.")
        gt_data = pd.read_csv(gt_filepath)
        
        columns = ['cached_ID', 'update_mode','n_train_episodes', 'max_steps', 'epsilon', 't', 'GD mean', 'GD median', 'GD std',
               'IGD mean', 'IGD median', 'IGD std', 'AHD mean', 'AHD median', 'AHD std']
        f_results = pd.DataFrame(columns=columns)
        
        for epsilon in epsilon_list:
            #for t in t_list:
                cached_ID = random.randrange(100000, 1000000)
                mcc = MCC(
                    gt_data=gt_data,
                    data=raw_data,
                    epsilon=epsilon,
                    epsilon_schedule=epsilon_schedule,
                    training_cutoffs=training_cutoffs,
                    y_col=y_col,
                    search_space={attr: search_space[attr]},
                    gold_standard=gold_standard,
                    semantic_metric=semantic_metric,
                    use_case=use_case,
                    imputer=imputer,
                    update_mode=update_mode,
                    t=t,
                    result_dst_dir=result_dst_dir,
                    debug=False,
                    dbscan=dbscan,
                    binned_values=binned_values
                )
                results, cached_results = mcc.run(cached_ID=cached_ID, n_runs=num_iterations)
                f_results = pd.concat([f_results, pd.DataFrame(results)], ignore_index=True)
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