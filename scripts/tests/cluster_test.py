"""
This script is to test q-learning with multiple attributes.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.framework.discretizers import *
from scripts.framework.SearchSpace import *
from scripts.framework.utils import *
from scripts.framework.UCB import *
from scripts.framework.framework_utils import *
from multi_attribute_qlearning import *

if __name__ == '__main__':
    ppath = sys.path[0] + '/../../'
    dataset = 'pima'
    use_case = 'imputation'
    semantic_metric = 'gpt_semantics'
    # Read json file
    exp_config = json.load(open(os.path.join(ppath, 'code', 'configs', f'{dataset}.json')))
    attributes = list(exp_config['attributes'].keys())
    
    # Load ground truth pareto front
    gt_df = pd.read_csv(os.path.join(ppath, 'truth', f'{dataset}.multi_attrs_exhaustive_search.{use_case}.csv'))
    datapoints = [np.array(gt_df[semantic_metric].values), np.array(gt_df['utility'].values)]
    lst = compute_pareto_front(datapoints)
    gt_df["Estimated"] = 0
    gt_df.loc[lst, "Estimated"] = 1
    gt_df["Explored"] = 1
    gt_df = gt_df[gt_df['Estimated'] == 1]
    gt_df = gt_df.drop_duplicates(subset=[semantic_metric, 'utility']) # remove duplicates
    gt_points = [np.array(gt_df[semantic_metric].values), np.array(gt_df['utility'].values)]
    gt_points = np.array(gt_points).T
    print(f"Ground truth points: {gt_points}")
    gd= GD(gt_points)
    igd = IGD(gt_points)
    
    # Load experiment data
    space_dict = {}
    for attr in attributes:
        data = pd.read_csv(os.path.join(ppath, 'data', dataset, 'scored_attributes', f'{attr}.csv'))
        # Filter data by gpt_prompt > 0
        ss = TestSearchSpace(data)
        space_dict[attr] = ss
    
    y_col = exp_config['target']
    raw_data = pd.read_csv(os.path.join(ppath, exp_config['data_path']))
    raw_data = raw_data[exp_config['features'] + [exp_config['target']]]
    raw_data = raw_data.dropna(subset=exp_config['features'] + [exp_config['target']])
    for attr in attributes:
        data = raw_data.copy()
        data[attr + '.gt'] = data[attr]
        nans = raw_data.sample(frac=0.3, random_state=42)
        data.loc[raw_data.index.isin(nans.index),attr] = np.nan
        raw_data = data.copy()
    #cluster_params = {'t': 0.5, 'criterion': 'distance'} # modeling configuration
    cluster_params = {'t': 0.4, 'criterion': 'distance'} # titanic 0.7 imputation configuration
    cluster_assignments = get_cluster_assignments_multi_attr(list(space_dict.values()), cluster_params)
    candidate_action_mappping, attribute_action_mapping = get_candidate_action_mapping(list(space_dict.values()), cluster_assignments, attributes)
    #cluster_assignments = [[1,2],[1,2,3]]

    gd_list = []
    igd_list = []
    for round in range(3):
        sampled_partition_list = []
        for i in range(len(cluster_assignments)):
            sampled_indices = []
            cluster_assignment = cluster_assignments[i]
            for c in np.unique(cluster_assignment):
                sampled_indices.append(np.random.choice(np.where(cluster_assignment == c)[0], 1, replace=False)[0])
            if 0 not in sampled_indices:
                sampled_indices.append(0)
            samples = [space_dict[attributes[i]].candidates[j] for j in sampled_indices]
            sampled_partition_list.append(samples)
        
        utility_list = []
        semantics_list = []
        # Get all combinations of sampled partitions
        p_sets = list(itertools.product(*sampled_partition_list))
        print(f"Total number of candidates: {len(p_sets)}")
        for i in range(len(p_sets)):
            data_i = raw_data.copy()
            p_sets[i] = list(p_sets[i])
            attrs_bins = [p_sets[i][j].bins for j in range(len(p_sets[i]))]
            pset_dict = dict(zip(attributes, attrs_bins))
            #acc = explainable_modeling_partition_dict(data_i, y_col, pset_dict)
            acc = data_imputation_multi_attrs(data_i, y_col, pset_dict)
            sem = np.sum([p_sets[i][j].gpt_semantics for j in range(len(p_sets[i]))]) / len(p_sets[i])
            utility_list.append(acc)
            semantics_list.append(sem)
        
        datapoints = [np.array(semantics_list), np.array(utility_list)]
        lst = compute_pareto_front(datapoints)
        estimated = [p_sets[i] for i in lst]
        est_points = np.array([semantics_list, utility_list]).T
        lst = list(lst)
        print(est_points)
        print(lst)
        est_points = est_points[lst]
        est_points = np.unique(est_points, axis=0)
        print(f"Estimated Pareto front: {est_points}")
        #print(f"Estimated: {estimated}")
        # Compute distances
        gd_i = gd(est_points)
        igd_i = igd(est_points)
        gd_list.append(gd_i)
        igd_list.append(igd_i)
        print(f"GD: {gd_i:.2f}, IGD: {igd_i:.2f}")
    
    # Print the mean, median and std of the dist
    print("GD: ", np.mean(gd_list), np.median(gd_list), np.std(gd_list))
    print("IGD: ", np.mean(igd_list), np.median(igd_list), np.std(igd_list))