"""
This is a comparable method to multi-attribute UCB and Q-learning
"""
import sys
import os
from datetime import datetime
sys.path.append("./")
from scripts.framework.discretizers import *
from scripts.framework.TestSearchSpace import *
from scripts.framework.utils import *
from scripts.framework.UCB import *
from scripts.framework.MCC import *
from scripts.framework.framework_utils import *
from baselines.exhaustive_modeling import explainable_modeling_multi_attrs
SEMANTICS = ['l2_norm', 'KLDiv', 'gpt_semantics']

def get_semantic_score(candidate, metric:SyntaxWarning):
    """
    Get the semantic score of a candidate
    :param candidate: Candidate object
    :param metric: Semantic metric
    :return: Semantic score
    """
    if metric == 'l2_norm':
        return candidate.l2_norm
    elif metric == 'KLDiv':
        return candidate.KLDiv
    elif metric == 'gpt_semantics':
        return candidate.gpt_semantics
    else:
        raise ValueError(f"Unknown semantic metric: {metric}")

if __name__ == '__main__':
    ppath = sys.path[0] + '/../'
    dataset = 'titanic'
    use_case = 'modeling'  # 'modeling', 'imputation'
    imputer = 'KNN'  # 'KNN', 'Simple', 'Iterative'
    missing_data_fraction = 0.3 # fraction of data to be used for imputation
    semantic_metric = 'gpt_semantics' #'KLDiv', 'gpt_semantics'
    
    # read json file
    exp_config = json.load(open(os.path.join(ppath, 'data', dataset, f'{dataset}.json')))
    attributes = exp_config['attributes'].keys()
    y_col = exp_config['target']
    raw_data = pd.read_csv(os.path.join(ppath, exp_config['data_path']))
    raw_data = raw_data[exp_config['features'] + [exp_config['target']]]
    raw_data = raw_data.dropna(subset=exp_config['features'] + [exp_config['target']])
    if use_case == 'imputation':
        for attr in attributes:
            data = raw_data.copy()
            data[attr + '.gt'] = data[attr]
            nans = raw_data.sample(frac=missing_data_fraction, random_state=42)
            data.loc[raw_data.index.isin(nans.index),attr] = np.nan
            raw_data = data.copy()

    # Load ground truth pareto front
    if use_case == 'modeling':
        utility_function = 'DecisionTreeClassifier'  # or 'RandomForestClassifier'
    else: # use_case == 'imputation'
        utility_function = f'{imputer}Imputer.{missing_data_fraction}'  # or 'KNNImputer', 'SimpleImputer'
    gt_df = pd.read_csv(os.path.join(ppath, 'truth', f'{dataset}.multi_attrs.{utility_function}.{semantic_metric}.csv'))
    datapoints = [np.array(gt_df[semantic_metric].values), np.array(gt_df['utility'].values)]
    lst = compute_pareto_front(datapoints)
    gt_df["Estimated"] = 0
    gt_df.loc[lst, "Estimated"] = 1
    gt_df["Explored"] = 1
    gt_df = gt_df[gt_df['Estimated'] == 1]
    gt_df = gt_df.drop_duplicates(subset=[semantic_metric, 'utility']) # remove duplicates
    gt_points = [np.array(gt_df[semantic_metric].values), np.array(gt_df['utility'].values)]
    gt_points = np.array(gt_points).T
    print("Ground truth points:", gt_points)
    gd= GD(gt_points)
    igd = IGD(gt_points)

    # Make a new folder to save the results
    date_today = datetime.today().strftime('%Y_%m_%d')
    dst = os.path.join(ppath, 'testresults', f"{dataset}.{date_today}.test_pipeline_example")
    dst_folder = os.path.join(dst, use_case)
    dst_fig_folder = os.path.join(dst, use_case, 'figs')
    dst_cluster_folder = os.path.join(dst, use_case, 'figs', 'clusters')
    if not os.path.exists(dst):
        os.mkdir(dst)
        os.mkdir(dst_folder)
        os.mkdir(dst_fig_folder)
        os.mkdir(dst_cluster_folder)
    elif not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
        os.mkdir(dst_fig_folder)
        os.mkdir(dst_cluster_folder)

    search_space = {}
    data_dfs = {}
    for attr in attributes:
        data = pd.read_csv(os.path.join(ppath, 'data', dataset, 'scored_attributes', f'{attr}.csv'))
        data_dfs[attr] = data
        search_space[attr] = TestSearchSpace(data, attr, decimal_place=1)
    
    # Generate all permutations
    gd_list = []
    igd_list = []
    ahd_list = []
    n_strategies_tried = 0
    permutations = list(itertools.permutations(attributes))
    for permutation in permutations:
        permutation = list(permutation)
        print(f"Permutation: {permutation}")
        partition_dicts = []
        for n, attr in enumerate(permutation):
            if len(partition_dicts) == 0:
                for candidate in search_space[attr].candidates:
                    data_i = raw_data.copy()
                    if use_case == 'modeling':
                        utility_score = explainable_modeling_multi_attrs(data_i, y_col, {attr: candidate.bins})
                    else: 
                        utility_score = data_imputation_multi_attrs(data_i, y_col, {attr: candidate.bins})
                    partition_dicts.append({"Partition": {attr: candidate.bins}, "Semantic": get_semantic_score(candidate, semantic_metric), "Utility": utility_score})
            else:
                new_partition_dicts = []
                for partition_dict in partition_dicts:
                    for candidate in search_space[attr].candidates:
                        new_partition_dict = {}
                        new_partition_dict["Partition"] = partition_dict["Partition"].copy()
                        new_partition_dict["Partition"][attr] = candidate.bins
                        new_partition_dict["Semantic"] = (partition_dict["Semantic"] * n + get_semantic_score(candidate, semantic_metric)) / (n + 1)
                        data_i = raw_data.copy()
                        #print("Partition:", new_partition_dict["Partition"])
                        if use_case == 'modeling':
                            utility_score = explainable_modeling_multi_attrs(data_i, y_col, new_partition_dict["Partition"])
                        else: 
                            utility_score = data_imputation_multi_attrs(data_i, y_col, new_partition_dict["Partition"])
                        #print("Utility:", utility_score)
                        #print("Semantic:", new_partition_dict["Semantic"], "; Last semantic score:", partition_dict["Semantic"], "; ", get_semantic_score(candidate, semantic_metric), n)
                        new_partition_dict["Utility"] = utility_score
                        new_partition_dicts.append(new_partition_dict)
                        n_strategies_tried += 1
                partition_dicts = new_partition_dicts
            # Get the pareto front after each attribute
            semantic_scores = [p["Semantic"] for p in partition_dicts]
            utility_scores = [p["Utility"] for p in partition_dicts]
            datapoints = np.array([semantic_scores, utility_scores])
            lst = compute_pareto_front(datapoints)
            partition_dicts = [partition_dicts[i] for i in lst]
        
        semantic_scores = [p["Semantic"] for p in partition_dicts]
        utility_scores = [p["Utility"] for p in partition_dicts]
        estimated_points = np.array([semantic_scores, utility_scores])
        estimated_points = np.array(estimated_points).T
        estimated_points = np.unique(estimated_points, axis=0)
        print("Estimated:", estimated_points)
        gd_i = gd(estimated_points)
        igd_i = igd(estimated_points)
        ahd_i = Evaluator.average_hausdorff_distance(gd_i, igd_i, mode='max')
        gd_list.append(gd_i)
        igd_list.append(igd_i)
        ahd_list.append(ahd_i)
        print(f"GD: {gd_i}")
        print(f"IGD: {igd_i}")
        print(f"AHD: {ahd_i}")
    
    # Print the mean, median, std of the gd and igd
    print(f"GD mean: {np.mean(gd_list)}; median: {np.median(gd_list)}; std: {np.std(gd_list)}")
    print(f"IGD mean: {np.mean(igd_list)}; median: {np.median(igd_list)}; std: {np.std(igd_list)}")
    print(f"AHD mean: {np.mean(ahd_list)}; median: {np.median(ahd_list)}; std: {np.std(ahd_list)}")
    # best permutation
    best_idx = np.argmin(ahd_list)
    best_permutation = permutations[best_idx]
    print(f"Best permutation: {best_permutation} with GD: {gd_list[best_idx]}, IGD: {igd_list[best_idx]}, AHD: {ahd_list[best_idx]}")
    print(f"Number of strategies tried: {n_strategies_tried}")