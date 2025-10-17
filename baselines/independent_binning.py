"""
This is a comparable method to multi-attribute UCB and Q-learning
"""
import sys
import os
from datetime import datetime
sys.path.append("./")
from scripts.framework.discretizers import *
from scripts.framework.SearchSpace import *
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
    ppath = sys.path[0] + '/../../'
    dataset = 'titanic'
    use_case = 'modeling'  # 'modeling', 'imputation'
    semantic_metric = 'gpt_semantics' #'KLDiv', 'gpt_semantics'
    imputer = 'KNN'  # 'KNN', 'Simple', 'Iterative'
    missing_data_fraction = 0.3 # fraction of data to be used for imputation
    
    # read json file
    exp_config = json.load(open(os.path.join(ppath, 'code', 'configs', f'{dataset}.json')))
    attributes = exp_config['attributes'].keys()
    y_col = exp_config['target']
    raw_data = pd.read_csv(os.path.join(ppath, exp_config['data_path']))
    raw_data = raw_data[exp_config['features'] + [exp_config['target']]]
    raw_data = raw_data.dropna(subset=exp_config['features'] + [exp_config['target']])
    if use_case == 'imputation':
        for attr in attributes:
            data = raw_data.copy()
            data[attr + '.gt'] = data[attr]
            nans = raw_data.sample(frac=0.3, random_state=42)
            data.loc[raw_data.index.isin(nans.index),attr] = np.nan
            raw_data = data.copy()

    # Load ground truth pareto front
    if use_case == 'modeling':
        utility_function = 'DecisionTreeClassifier'
    else:
        utility_function = f'{imputer}Imputer.{missing_data_fraction}'

    gt_data = pd.read_csv(os.path.join(ppath, 'truth', f'{dataset}.multi_attrs.{utility_function}.{semantic_metric}.csv'))
    gt_df = gt_data.copy()
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
    
    for attr in attributes:
        for candidate in search_space[attr].candidates:
            data_i = raw_data.copy()
            if use_case == 'modeling':
                utility_score = explainable_modeling_multi_attrs(data_i, y_col, {candidate.attribute: candidate.bins})
            elif use_case == 'imputation':
                utility_score = data_imputation_multi_attrs(data_i, y_col, {candidate.attribute: candidate.bins})
            candidate.utility = utility_score
        # Get the pareto front after each attribute
        datapoints = [np.array([get_semantic_score(c, semantic_metric) for c in search_space[attr].candidates]),
                      np.array([c.utility for c in search_space[attr].candidates])]
        lst = compute_pareto_front(datapoints)
        selected_candidates = [search_space[attr].candidates[i] for i in lst]
        print(f"Selected candidates for {attr}: {len(selected_candidates)}")
        search_space[attr].candidates = selected_candidates
    
    # Get the pareto front after all attributes
    # 1. Get the tuple of (candidate1, candidate2, candidate3) from search_space
    candidates_tuples = list(itertools.product(*[search_space[attr].candidates for attr in attributes]))
    print(f"Total number of candidates: {len(candidates_tuples)}")
    # 2. Compute the utility and semantic score for each candidate tuple
    partition_dicts = []
    for candidate_tuple in candidates_tuples:
        utility = 0
        semantics = 0
        partition_dict = {"Partition": {}}
        for candidate in candidate_tuple:
            semantics += get_semantic_score(candidate, semantic_metric)
            partition_dict["Partition"][candidate.attribute] = candidate.bins
        # Compute the utility based on the use case
        semantics = semantics / len(candidate_tuple)
        data_i = raw_data.copy()
        if use_case == 'modeling':
            utility = explainable_modeling_multi_attrs(data_i, y_col, partition_dict["Partition"])
        elif use_case == 'imputation':
            utility = data_imputation_multi_attrs(data_i, y_col, partition_dict["Partition"])
        partition_dict["Semantic"] = semantics
        partition_dict["Utility"] = utility
        partition_dicts.append(partition_dict)
    # 3. Get the pareto front from the computed candidates
    semantic_scores = [p["Semantic"] for p in partition_dicts]
    utility_scores = [p["Utility"] for p in partition_dicts]
    datapoints = np.array([semantic_scores, utility_scores])
    lst = compute_pareto_front(datapoints)
    partition_dicts = [partition_dicts[i] for i in lst]
    estimated_points = np.array([[p["Semantic"] for p in partition_dicts],
                                 [p["Utility"] for p in partition_dicts]]).T
    estimated_points = np.unique(estimated_points, axis=0)
    print("Estimated:", estimated_points)
    gd_i = gd(estimated_points)
    igd_i = igd(estimated_points)
    ahd_i = Evaluator.average_hausdorff_distance(gd_i, igd_i, mode='max')
    print(f"GD: {gd_i}")
    print(f"IGD: {igd_i}")
    print(f"AHD: {ahd_i}")