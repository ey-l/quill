import sys
import os
import random
import numpy as np
from scripts.framework.MCC import *

class Cluster:
    def __init__(self, clusterID):
        self.count = 0
        self.value = 0.0
        self.clusterID = clusterID
        self.points = []
        self.unexplored_start_index = 0

class UCB:
    def __init__(self, gt_data, alpha=2, n_samples=None, p=None, data=None, y_col=None, search_space=None, gold_standard=None, t:float=0.0, semantic_metric:str="gpt_semantics", use_case:str="modeling", imputer='KNN', result_dst_dir:str=None, debug=False, treatment_attr=None,confounders=None,causal_case=None) -> None:
        """
        Initialize the UCB algorithm with the given alpha value.
        :param alpha: The exploration parameter.
        """
        self.gt_data = gt_data
        self.alpha = alpha
        self.n_samples = n_samples
        self.p = p
        self.data = data
        self.y_col = y_col
        self.search_space = search_space
        self.attribute = search_space.attribute if search_space else None
        self.gold_standard = gold_standard
        self.t = t
        self.cluster_params = {'t': t, 'criterion': 'distance'}
        self.use_case = use_case
        self.imputer = imputer
        self.semantic_metric = semantic_metric
        self.result_dst_dir = result_dst_dir
        self.debug = debug

        if self.t == 0.0:
            self.t = find_optimal_t(self.search_space)

        self.evaluator = None
        if self.gt_data is not None:
            self.evaluator = Evaluator(gt_data, semantic_metric, use_case)
        # System attributes
        self.explored_nodes = None
        self.fully_explored_clusters = None
        self.cluster_assignments = None
        self.num_clusters = 0
        self.clusters = None

        if self.use_case == 'causal_inference':
            if not all([treatment_attr,confounders,causal_case]):
                raise ValueError("Treatment,confounders and causal_case must be provided for causal inference evaluation.")
            self.treatment_attr = treatment_attr
            self.confounders = confounders
            self.causal_case = causal_case
            unbinned_utility_args= dict(causal_case=causal_case, data=data, known_args=(self.treatment_attr, self.y_col,self.confounders) if causal_case == "known" else None)
            self.unbinned_pre_utility = get_unbbined_data_causal_utility(**unbinned_utility_args)
        
    def initialize(self):
        self.clusters = [Cluster(i) for i in range(self.num_clusters)]
        
        for i, c in enumerate(self.cluster_assignments):
            # List of Partitions
            self.clusters[c].points.append(self.search_space.candidates[i])
        
        # Shuffle the points in each cluster now 
        # So later we explore in order
        for c in self.clusters:
            random.shuffle(c.points)
            c.count += 1
            data_i = self.data.copy()
            candidate = c.points[c.unexplored_start_index]
            if self.use_case == 'modeling':
                utility = explainable_modeling_one_attr(data_i, self.y_col, self.attribute, candidate)
            elif self.use_case == 'imputation':
                utility = data_imputation_one_attr(data_i, self.y_col, self.attribute, candidate, imputer=self.imputer)
            elif self.use_case == 'visualization':
                utility = visualization_one_attr_spearmanr(data_i, self.y_col, self.attribute, candidate)
            elif self.use_case == 'causal_inference':
                utility = causal_inference_utility(data_i, self.y_col, {self.attribute: candidate.bins},self.confounders, self.treatment_attr, self.causal_case,self.unbinned_pre_utility)
            else:
                raise ValueError(f"Unknown use case: {self.use_case}")
            candidate.set_utility(utility) # Set the utility score
            c.value += candidate.utility + candidate.get_semantic_score(self.semantic_metric)
            self.explored_nodes.append(c.points[c.unexplored_start_index])
            c.unexplored_start_index += 1 # Advance the unexplored start index
    
    def _select_cluster(self):
        # Select the cluster with the highest value
        max_value = -1
        selected_cluster = None
        for c in self.clusters:
            # Scale the exploratory right hand term
            ucb = c.value / c.count + self.alpha * np.sqrt(2 * np.log(len(self.explored_nodes)) / c.count)
            #print(f"Cluster {c.clusterID}: value={c.value}, count={c.count}, UCB={ucb}")
            if ucb > max_value:
                max_value = ucb
                selected_cluster = c
        return selected_cluster
    
    def explore(self):
        while True:
            selected_cluster = self._select_cluster()
            if selected_cluster is None:
                #print("No more clusters to explore.")
                return None
            if selected_cluster.unexplored_start_index < len(selected_cluster.points): break
            self.fully_explored_clusters.append(selected_cluster)
            self.clusters.remove(selected_cluster)

        #datapoint = get_points([selected_cluster.points[selected_cluster.unexplored_start_index]], self.semantic_metric)
        #selected_cluster.value += datapoint[0][0] + datapoint[1][0]
        candidate = selected_cluster.points[selected_cluster.unexplored_start_index]
        data_i = self.data.copy()
        #candidate.utility = explainable_modeling_using_strategy(data_i, self.y_col, candidate)
        if self.use_case == 'modeling':
            utility = explainable_modeling_one_attr(data_i, self.y_col, self.attribute, candidate)
        elif self.use_case == 'imputation':
            utility = data_imputation_one_attr(data_i, self.y_col, self.attribute, candidate)
        elif self.use_case == 'visualization':
            utility = visualization_one_attr_spearmanr(data_i, self.y_col, self.attribute, candidate)
        elif self.use_case == 'causal_inference':
            utility = causal_inference_utility(data_i, self.y_col, {self.attribute: candidate.bins},self.confounders, self.treatment_attr, self.causal_case,self.unbinned_pre_utility)
        else:
            raise ValueError(f"Unknown use case: {self.use_case}")
        candidate.set_utility(utility) # Set the utility score
        selected_cluster.value += candidate.utility + candidate.get_semantic_score(self.semantic_metric)
        
        selected_cluster.count += 1
        self.explored_nodes.append(selected_cluster.points[selected_cluster.unexplored_start_index])
        selected_cluster.unexplored_start_index += 1
        return selected_cluster
    
    def explore_point(self, point):
        # Explore a specific point
        data_i = self.data.copy()
        #point.utility = explainable_modeling_using_strategy(data_i, self.y_col, point)
        if self.use_case == 'modeling':
            utility = explainable_modeling_one_attr(data_i, self.y_col, self.attribute, point)
        elif self.use_case == 'imputation':
            utility = data_imputation_one_attr(data_i, self.y_col, self.attribute, point)
        elif self.use_case == 'visualization':
            utility = visualization_one_attr_spearmanr(data_i, self.y_col, self.attribute, point)
        elif self.use_case == 'causal_inference':
            # TODO: implement causal inference for one attribute
            utility = causal_inference_utility(data_i, self.y_col, {self.attribute: point.bins},self.confounders, self.treatment_attr, self.causal_case,self.unbinned_pre_utility)
            #raise NotImplementedError("Causal inference for one attribute is not implemented yet.")
        else:
            raise ValueError(f"Unknown use case: {self.use_case}")
        point.set_utility(utility) # Set the utility score
        self.explored_nodes.append(point)
        return point
    
    def get_estimated_P(self, estimated_partitions):
        # Find the pareto front
        semantic_scores = [p.get_semantic_score(self.semantic_metric) for p in estimated_partitions]
        utility_scores = [p.utility for p in estimated_partitions]
        datapoints = np.array([semantic_scores, utility_scores])
        lst = compute_pareto_front(datapoints)
        est_partitions = [estimated_partitions[i] for i in lst]
        est_points = np.array([[p.get_semantic_score(self.semantic_metric) for p in est_partitions],
                               [p.utility for p in est_partitions]])
        est_points = np.array(est_points).T
        est_points = np.unique(est_points, axis=0)
        # Turn the np array in each partition into a list for saving the results to a JSON file
        for p in est_partitions:
            p.bins = p.bins.tolist() if isinstance(p.bins, np.ndarray) else p.bins
        return est_points, est_partitions

    def run(self, cached_ID, n_runs=20):
        """
        Run the UCB algorithm.
        """
        cached_results = {}
        gd_list = []
        igd_list = []
        ahd_list = []
        for round in range(n_runs):
            start_time = time.time()
            self.explored_nodes = [] # Reset explored nodes for each run
            self.fully_explored_clusters = [] # Reset fully explored clusters for each run
            self.cluster_assignments = linkage_distributions(self.search_space, self.cluster_params)
            self.num_clusters = len(np.unique(self.cluster_assignments))
            
            # Set n_samples based on len(self.cluster_assignments)
            #n_samples_left = int(len(self.cluster_assignments) * self.p)
            if self.n_samples is None and self.p is not None:
                self.n_samples = int(len(self.cluster_assignments) * self.p)
            elif self.p is None and self.n_samples is not None:
                self.p = self.n_samples / int(len(self.cluster_assignments))
            elif self.n_samples is None and self.p is None:
                raise ValueError("Either n_samples or p must be provided.")
            
            n_samples_left = self.n_samples

            print(f"Round {round+1}/{n_runs}")
            if self.n_samples < self.num_clusters:
                estimated_indices = random_sampling_clusters_robust(self.cluster_assignments, {'p': self.p})
                estimated_partitions = [self.search_space.candidates[i] for i in estimated_indices]
            else:
                self.initialize()
                n_samples_left = n_samples_left - self.num_clusters
                for _ in range(n_samples_left):
                    self.explore()
                estimated_partitions = self.explored_nodes
                estimated_indices = [p.ID for p in estimated_partitions]
            
            if self.search_space.candidates[0].ID not in estimated_indices:
                # Explore search_space.candidates[0]
                point = self.explore_point(self.search_space.candidates[0])
                estimated_partitions.append(point)
            
            est_P, partitions = self.get_estimated_P(estimated_partitions)
            print(f"Estimated Pareto front: {est_P}")
            end_time = time.time()
            # Extract bins from estimated partitions
            est_partitions = []
            for p in partitions:
                p_dict = {self.attribute: p.bins.tolist() if isinstance(p.bins, np.ndarray) else p.bins,
                          "__semantic__": p.get_semantic_score(self.semantic_metric),
                          "__utility__": p.utility,}
                est_partitions.append(p_dict)

            # Compute distances
            n_llm_calls = self.n_samples
            if self.evaluator is not None:
                gd_value = self.evaluator.gd(est_P)
                igd_value = self.evaluator.igd(est_P)
                ahd_value = Evaluator.average_hausdorff_distance(gd_value, igd_value, mode='max')
                print(f"GD: {gd_value:.2f}, IGD: {igd_value:.2f}, AHD: {ahd_value:.2f}")
                gd_list.append(gd_value)
                igd_list.append(igd_value)
                ahd_list.append(ahd_value)
            else:
                gd_value = 'No ground truth data provided.'
                igd_value = 'No ground truth data provided.'
                ahd_value = 'No ground truth data provided.'
                print("No ground truth data provided, skipping distance calculations.")
            
            cached_results[round] = {
                'n_samples': self.n_samples, # budget, n_utility_calls
                'alpha': self.alpha, 
                'p': self.p, 
                't': self.t,
                'estimated_P': est_P.tolist(),
                'estimated_partitions': est_partitions,
                #'mean_reward': mean_reward,
                #'std_reward': std_reward,
                'gd_value': gd_value,
                'igd_value': igd_value,
                'ahd_value': ahd_value,
                'n_llm_calls': n_llm_calls,
                'time_taken': end_time - start_time,
            }
            
        if self.evaluator is not None:
            # Print the mean, median and std of the distances
            print("GD: ", np.mean(gd_list), np.median(gd_list), np.std(gd_list))
            print("IGD: ", np.mean(igd_list), np.median(igd_list), np.std(igd_list))
            print("HD: ", np.mean(ahd_list), np.median(ahd_list), np.std(ahd_list))
            return {
                'cached_ID': cached_ID,
                'n_samples': self.n_samples,
                'alpha': self.alpha, 
                'p': self.p, 
                't': self.t,
                "GD mean": np.mean(gd_list),
                "GD median": np.median(gd_list),
                "GD std": np.std(gd_list),
                "IGD mean": np.mean(igd_list),
                "IGD median": np.median(igd_list),
                "IGD std": np.std(igd_list),
                "AHD mean": np.mean(ahd_list),
                "AHD median": np.median(ahd_list),
                "AHD std": np.std(ahd_list)
            }, cached_results
        else:
            print("No ground truth data provided, skipping distance calculations.")
            return {
                'cached_ID': cached_ID,
                'n_samples': self.n_samples,
                'alpha': self.alpha, 
                'p': self.p, 
                't': self.t,
                'GD mean': 'No ground truth data provided.',
                'GD median': 'No ground truth data provided.',
                'GD std': 'No ground truth data provided.',
                'IGD mean': 'No ground truth data provided.',
                'IGD median': 'No ground truth data provided.',
                'IGD std': 'No ground truth data provided.',
                'AHD mean': 'No ground truth data provided.',
                'AHD median': 'No ground truth data provided.',
                'AHD std': 'No ground truth data provided.'
            }, cached_results

def UCB_estimate(gt_data, alpha, search_space, strategy_space, clustering, semantic_metric='l2_norm', clustering_params:Dict={}, sampling_params:Dict={}, use_case=None, if_runtime_stats=True, gold_standard=None, data=None, y_col=None) -> List:
    """
    Estimate the Pareto front using the UCB algorithm.
    When budget is less than the number of clusters, we use random sampling.
    """
    cluster_assignments = clustering(search_space, clustering_params)

    p = sampling_params['p']
    budget = int(len(cluster_assignments) * p)
    if budget < len(np.unique(cluster_assignments)):
        sampled_indices = random_sampling_clusters_robust(cluster_assignments, sampling_params)
        sampled_partitions = [strategy_space.candidates[i] for i in sampled_indices]
    else:
        ucb = UCB(gt_data=gt_data, alpha=alpha, data=data, y_col=y_col, p=p, search_space=search_space, semantic_metric=semantic_metric, use_case='modeling', result_dst_dir=None, debug=False, gold_standard=gold_standard)
        ucb.cluster_assignments = cluster_assignments
        ucb.initialize()
        budget = budget - len(np.unique(cluster_assignments))
        for _ in range(budget):
            ucb.explore()
        sampled_partitions = ucb.explored_nodes
        sampled_indices = [p.ID for p in sampled_partitions]

    if strategy_space.candidates[0].ID not in sampled_indices:
        # Explore search_space.candidates[0]
        strategy_space.candidates[0] = ucb.explore_point(strategy_space.candidates[0])
        sampled_partitions.append(strategy_space.candidates[0])
    #explored_points, pareto_points, est_points_df = get_pareto_front(sampled_partitions, semantic_metric)
    datapoints = get_points(sampled_partitions, semantic_metric)
    lst = compute_pareto_front(datapoints)
    est_pareto_partitions = [sampled_partitions[i] for i in lst]

    return est_pareto_partitions

if __name__ == '__main__':
    ppath = sys.path[0] + '/../../'
    dataset = 'pima'
    use_case = 'modeling'
    # read json file
    exp_config = json.load(open(os.path.join(ppath, 'code', 'configs', f'{dataset}.json')))
    attributes = exp_config['attributes'].keys()

    for attr in attributes:
        f_quality = []
        f_runtime = []
        # load experiment data
        if attr == "Glucose":
            data = pd.read_csv(os.path.join(ppath, 'experiment_data', dataset, use_case, f'{attr}.csv'))
            ss = TestSearchSpace(data)
            break
    
    semantic_metric = 'l2_norm'
    search_space = ss
    datapoints, gt_pareto_points, points_df = get_pareto_front(ss.candidates, semantic_metric)

    parameters = {'t': 0.5, 'criterion': 'distance'}
    t = parameters['t']
    criterion = parameters['criterion']
    X = np.array([p.distribution for p in search_space.candidates])
    X = pairwise_distance(X, metric=wasserstein_distance)
    Z = linkage(X, method='ward')
    #fig = plt.figure(figsize=(25, 10))
    #dn = dendrogram(Z, color_threshold=t)
    agg_clusters = fcluster(Z, t=t, criterion=criterion)
    agg_clusters = [x-1 for x in agg_clusters] # 0-indexing
    #print(Z)

    avg_distance_results = []
    for round in range(1):
        # Create dendrogram
        ucb = UCB()
        ucb.initialize(agg_clusters, search_space, semantic_metric)
        for i in range(20):
            print(f"Round {i}")
            print(ucb.explore().clusterID)

        explored_nodes = ucb.explored_nodes
        explored_points, est_pareto_points, _ = get_pareto_front(explored_nodes, semantic_metric)
        print(est_pareto_points)
        average_distance = average_distance(gt_pareto_points, est_pareto_points, debug=True)
        avg_distance_results.append(average_distance)

        # Sort the points for plotting
        gt_pareto_points = sorted(gt_pareto_points, key=lambda x: x[0])
        est_pareto_points = sorted(est_pareto_points, key=lambda x: x[0])
        # Plot the Pareto front
        gt_pareto_points = np.array(gt_pareto_points)
        est_pareto_points = np.array(est_pareto_points)
        #datapoints = np.array(datapoints)
        # Set size of the plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(explored_points[0], explored_points[1], c='gray', label='Explored Points', marker='x',)
        ax.plot(gt_pareto_points[:, 0], gt_pareto_points[:, 1], '+-', c='red', label='Ground Truth')
        ax.plot(est_pareto_points[:, 0], est_pareto_points[:, 1], 'x-', c='green', label='Estimated')
        ax.legend(bbox_to_anchor=(1, 1),ncol=3)
        ax.set_xlabel('Semantic Distance', fontsize=14)
        ax.set_ylabel('Utility', fontsize=14)
        ax.set_title('Pareto Curve Estimated vs. Ground-Truth', fontsize=14)

        fig.savefig(os.path.join(ppath, 'code', 'plots', f'UCB_{attr}_{round}.png'), bbox_inches='tight')
    
    # plot the average distance as a boxplot
    fig, ax = plt.subplots()
    ax.boxplot(avg_distance_results)
    ax.set_xlabel('UCB')
    ax.set_ylabel('Average Distance')
    fig.savefig(os.path.join(ppath, 'code', 'plots', f'UCB_{attr}_boxplot.png'), bbox_inches='tight')