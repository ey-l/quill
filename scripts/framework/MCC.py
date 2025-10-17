"""
Monte Carlo for multi-attribute explainable modeling
"""
# import numpy as np
# import time
import random
from collections import defaultdict
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

#from import_packages import *
#from scripts.framework.discretizers import *
# from scripts.framework.TestSearchSpace import *
from scripts.framework.framework_utils import *
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
#from gpt_grader import evaluate_groupings_gpt

class Evaluator:
    def __init__(self, gt_data, semantic_metric, use_case):
        self.gt_data = gt_data
        self.semantic_metric = semantic_metric
        self.use_case = use_case
        self.gt_points = None
        self.gd = None
        self.igd = None

        datapoints = [np.array(self.gt_data[semantic_metric].values), np.array(self.gt_data['utility'].values)]
        lst = compute_pareto_front(datapoints)
        self.gt_data["Estimated"] = 0
        self.gt_data.loc[lst, "Estimated"] = 1
        self.gt_data["Explored"] = 1
        self.gt_data = self.gt_data[self.gt_data['Estimated'] == 1]
        self.gt_data = self.gt_data.drop_duplicates(subset=[semantic_metric, 'utility']) # remove duplicates
        if semantic_metric == 'gpt_semantics':
            self.gt_data[semantic_metric] = self.gt_data[semantic_metric] / 4
        self.gt_points = [np.array(self.gt_data[semantic_metric].values), np.array(self.gt_data['utility'].values)]
        self.gt_points = np.array(self.gt_points).T
        print(f"Ground truth points: {self.gt_points}")
        self.gd= GD(self.gt_points)
        self.igd = IGD(self.gt_points)
    
    def average_hausdorff_distance(gd_value, igd_value, mode='max'):
        """
        Compute the average Hausdorff distance between two sets of points.
        :param gd_value: float, GD value
        :param igd_value: float, IGD value
        :param mode: str, 'max' or 'average', determines how to combine GD and IGD values
        :return: float, average Hausdorff distance
        """
        if mode == 'max':
            return max(gd_value, igd_value)
        else: return (gd_value + igd_value) / 2
    
    def get_num_of_llm_calls(sampled_partitions):
        """
        Get the number of LLM calls made during the training.
        Each partition only requires one LLM call from the semantic grader, so we can count the unique partitions for each action.
        :param sampled_partitions: dict, mapping of actions to sampled partitions
        :return: int, number of LLM calls
        """
        llm_calls = 0
        for action, partitions in sampled_partitions.items():
            unique_partitions = set(partitions)
            llm_calls += len(unique_partitions)
        return llm_calls

class MCC:
    def __init__(self, 
                 lut=None, 
                 gt_data=None, 
                 epsilon=0.5, 
                 epsilon_schedule=False, 
                 training_cutoffs=[200, 400], 
                 n_eval_episodes=2000, 
                 data=None, 
                 y_col=None, 
                 search_space=None, 
                 gold_standard=None, 
                 t:float=0, 
                 semantic_metric:str="gpt_semantics", 
                 use_case:str="modeling", 
                 imputer="KNN", 
                 update_mode:str="final-only", 
                 result_dst_dir:str=None, 
                 debug=False,
                 treatment_attr=None,
                 confounders=None,
                 causal_case=None, 
                 dbscan=False, 
                 binned_values=False) -> None:
        """
        Initialize the MCC algorithm with the given epsilon value.
        :param gt_data: Ground truth dataframe for evaluation.
        :param epsilon: The exploration parameter.
        :param N: Number of top greedy clusters to consider in evaluation.
        :param n_training_episodes: Number of training episodes.
        :param use_case: The use case to optimize for. One of: 'modeling', 'imputation', 'causal'.

        """
        self.lut = lut
        self.epsilon = epsilon
        self.training_cutoffs = training_cutoffs
        self.n_eval_episodes = n_eval_episodes
        self.data = data
        self.y_col = y_col
        self.t = {}
        #self.cluster_params = {'t': t, 'criterion': 'distance'}
        self.search_space = search_space
        self.gold_standard = gold_standard
        self.max_steps = len(list(self.search_space.keys())) # Max steps is the number of attributes in the search space
        self.update_mode = update_mode  # 'final-only' or 'every-step'
        self.result_dst_dir = result_dst_dir
        self.debug = debug
        self.semantic_metric = semantic_metric
        self.use_case = use_case
        self.imputer = imputer  # Imputation method, can be 'KNN', 'Iterative', or 'Simple'
        self.epsilon_schedule = epsilon_schedule # If True, epsilon will be decayed over time
        self.dbscan = dbscan # If True, use DBSCAN for clustering
        self.binned_values = binned_values # If True, use binned values for imputation evaluation

        if gt_data is not None:
            self.evaluator = Evaluator(gt_data, semantic_metric, use_case)
        else: self.evaluator = None
        
        if t == 0:
            # If no t values are provided, automatically tune t values based on the search space.
            if dbscan:
                for attr, space in self.search_space.items():
                    t, _ = find_optimal_dbscan_params(space, coverage_weight=0.30, verbose=True)
                    self.t[attr] = t
            else:
                for attr, space in self.search_space.items():
                    t = find_optimal_t(space, self.binned_values)
                    self.t[attr] = t
        # If t is a dict, use the provided t values for each attribute
        else:
            if dbscan: 
                self.t = {attr:{'eps':self.t,'min_samples':2} for attr in self.search_space.keys()}
            else: self.t = {attr:self.t for attr in self.search_space.keys()}

        # System attributes
        self.cluster_assignments = None
        self.cluster_action_mappping = None
        self.attribute_action_mapping = None
        self.Qtable = None
        self.sampled_cluster_partitions = None

        if self.use_case == 'causal_inference':
            if not all([treatment_attr,confounders,causal_case]):
                raise ValueError("Treatment,confounders and causal_case must be provided for causal inference evaluation.")
            self.treatment_attr = treatment_attr
            self.confounders = confounders
            self.causal_case = causal_case
            unbinned_utility_args= dict(causal_case=causal_case, data=data, known_args=(self.treatment_attr, self.y_col,self.confounders) if causal_case == "known" else None)
            self.unbinned_pre_utility = get_unbbined_data_causal_utility(**unbinned_utility_args)

    def explainable_modeling(self, data, y_col, attr:str, bins) -> Tuple[float, pd.DataFrame]:
        """
        Wrapper function to model the data using an explainable model
        :param data: DataFrame
        :param attr: str
        :param bins: List
        :return: float
        """
        data[attr + '.binned'] = pd.cut(data[attr], bins, labels=False)
        data[attr + '.binned'] = data[attr + '.binned'].astype('float64')
        data = data.dropna(subset=[attr + '.binned'])

        # Evaluate the explainable model
        X_cols = [col for col in data.columns if col != y_col and col != attr]
        X = data[X_cols]
        y = data[y_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a decision tree model
        model = DecisionTreeClassifier(random_state=0, max_depth=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_accuracy = accuracy_score(y_test, y_pred)
        
        return model_accuracy, data

    def data_imputation(self, data, y_col, attr:str, bins, imputer) -> Tuple[float, pd.DataFrame]:
        """
        Wrapper function to impute missing values in a dataset
        :param data: DataFrame
        :param attr: str
        :param bins: List
        :return: float
        """
        # Bin attr column, with nan values
        data[attr + '.binned'] = pd.cut(data[attr], bins=bins, labels=bins[1:])
        
        # Impute the missing values using KNN
        X_cols = [col for col in data.columns if col != y_col and col != attr and col != attr + '.gt']
        X = data[X_cols]
        idx = X.columns.get_loc(attr + '.binned')
        if imputer == 'KNN':
            imputer = KNNImputer(n_neighbors=len(bins)-1)
            X_imputed = imputer.fit_transform(X)
        elif imputer == 'Iterative':
            imputer = IterativeImputer(max_iter=3, random_state=0)
            X_imputed = imputer.fit_transform(X)
        elif imputer == 'Simple':
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
        else:
            raise ValueError(f"Unknown imputer: {imputer}")
        
        # Bin imputed values
        data_imputed = np.round(X_imputed[:, idx])
        data[attr+'.imputed'] = data_imputed
        data[attr + '.final'] = pd.cut(data[attr+'.imputed'], bins=bins, labels=bins[1:])
        data[attr + '.final'] = data[attr + '.final'].astype('float64')
        value_final = np.array(data[attr + '.final'].values)
        value_final[np.isnan(value_final)] = -1
        value_final = np.round(value_final)

        # Evaluate data imputation
        data[attr + '.gt'] = pd.cut(data[attr + '.gt'], bins=bins, labels=bins[1:])
        data[attr + '.gt'] = data[attr + '.gt'].astype('float64')
        value_gt = np.array(data[attr + '.gt'].values)
        value_gt[np.isnan(value_gt)] = -1
        value_gt = np.round(value_gt)
        impute_accuracy = accuracy_score(value_gt, value_final)

        return impute_accuracy, data

    def get_cluster_assignments_multi_attr(self):
        """
        Get cluster assignments for multi-attribute data.
        :return: List of clusters
        """
        assignments_list = []
        for attr, space in self.search_space.items():
            if self.dbscan:
                cluster_params = {'eps': self.t[attr]['eps'], 'min_samples': self.t[attr]['min_samples']} if attr in self.t else {'eps': 0.5, 'min_samples': 2}
                assignments = DBSCAN_distributions(space, cluster_params)
            else: 
                cluster_params = {'t': self.t.get(attr, 0.5), 'criterion': 'distance'} if attr in self.t else {'t': 0.5, 'criterion': 'distance'}
                if self.binned_values:
                    assignments = linkage_binned_values(space, cluster_params, binned_values=True)
                else: assignments = linkage_distributions(space, cluster_params)
            
            assignments_list.append(assignments)
            if self.debug: print("There are ", len(np.unique(assignments)), " unique clusters")
        return assignments_list
    
    def get_candidate_action_mapping(self):
        """
        :param space_list: List of PartitionSearchSpace
        :param cluster_assignments: List of clusters
        :return: List of candidate actions
        :return: Dictionary of attribute cluster mapping
        """
        candidate_action_mapping = []
        attribute_cluster_mapping = {}
        count = 0
        attributes = list(self.search_space.keys())
        space_list = list(self.search_space.values())
        for i, space in enumerate(space_list):
            cluster_indices = []
            candidate_indices = []
            for cluster in np.unique(self.cluster_assignments[i]):  
                candidate_indices = np.where(self.cluster_assignments[i] == cluster)[0]
                candidate_action_mapping.append(candidate_indices)
                cluster_indices.append(count)
                count += 1
            attribute_cluster_mapping[attributes[i]] = cluster_indices
        if self.debug: 
            print("Candidate action mapping: ", candidate_action_mapping)
            print("Attribute cluster mapping: ", attribute_cluster_mapping)
        return candidate_action_mapping, attribute_cluster_mapping
    
    def initialize_Q(self):
        """
        Initialize the Q table.
        The Q table is initialized with zeros.
        The state space is the number of unique clusters for each attribute plus two for the initial and end state.
        The action space is the number of unique clusters for each attribute plus two for the initial and end state.
        :return: Q table
        """
        # Add a dummy state and action for the initial state and end state
        state_space = sum([len(np.unique(assignments)) for assignments in self.cluster_assignments]) + 2
        action_space = sum([len(np.unique(assignments)) for assignments in self.cluster_assignments]) + 2
        if self.debug: print("There are ", state_space, " possible states; ", action_space, " possible actions")
        Qtable = np.zeros((state_space, action_space))
        return Qtable
    
    def mask_Qtable(self):
        """
        Mask the Q table to only allow actions that are possible in the current state.
        Only states that are for the same attribute are not allowed.
        """
        curr_attr = 0
        num_unique_clusters = len(np.unique(self.cluster_assignments[curr_attr]))
        curr_attr_start_index = 1
        curr_attr_end_index = num_unique_clusters + 1
        for state in range(1, len(self.Qtable)):
            if state == curr_attr_end_index:
                if self.debug:
                    print("State: ", state)
                    print("curr_attr: ", curr_attr)
                    print("curr_attr_start_index: ", curr_attr_start_index)
                curr_attr += 1
                if curr_attr < len(self.cluster_assignments):
                    num_unique_clusters = len(np.unique(self.cluster_assignments[curr_attr]))
                    curr_attr_start_index = curr_attr_end_index
                    curr_attr_end_index += num_unique_clusters
                else: curr_attr_start_index = curr_attr_end_index
            self.Qtable[state][curr_attr_start_index:curr_attr_end_index] = -np.inf
        
        self.Qtable[1:,-1] = 0
        self.Qtable[:, 0] = -np.inf
        self.Qtable[0, -1] = -np.inf # Have to bin something
        self.Qtable[-1,:-1] = -np.inf
        if self.debug: print("Qtable after masking: \n", self.Qtable)
        return self.Qtable
    
    def epsilon_greedy_policy(self, state, episode_n, step_count=None, attributes_ep=None):
        random_int = random.uniform(0,1)
        possible_actions = self.Qtable[state].copy()  # Get the possible actions for the current state
        #print("State: ", state, "Possible actions before filtering: ", possible_actions)
        if step_count < self.max_steps:
            # If we are not at the last step, we never choose the end action
            possible_actions[-1] = -np.inf  # The last action is the "do nothing" action, which we don't want to choose if we are not at the last step
        # Exclude actions that are not possible in the current state
        if attributes_ep is not None:
            for attr in attributes_ep:
                #print("Attribute: ", attr)
                for a in self.attribute_action_mapping[attr]:
                    possible_actions[a+1] = -np.inf
        #print("Possible actions: ", possible_actions)
        
        if self.epsilon_schedule:
            # Epsilon decay schedule
            epsilon_0 = 0.7
            epsilon_min = 0.5
            k = np.log(epsilon_0 / epsilon_min) / self.n_training_episodes
            epsilon_t = lambda t: max(epsilon_min, epsilon_0 * np.exp(-k * t))
            epsilon = epsilon_t(episode_n)
        else: 
            # Constant epsilon
            epsilon = self.epsilon
        
        if random_int > epsilon:
            action = np.argmax(possible_actions)
        else:
            # Implement random action, but only from the possible actions
            # Only pick actions that are not -inf
            possible_actions = np.where(possible_actions != -np.inf)[0]
            action = np.random.choice(possible_actions)
        return action

    def select_ucb_arm(self, t=2):
        bonuses = np.sqrt( (2*np.log(t+1)) / np.maximum(1, self.counts) )
        return np.argmax(self.Qtable + bonuses)

    def step_func(self, processed_data, action, attribute=None, training=True):
        # Find what attribute and cluster the action corresponds to, from the action mapping
        
        if attribute is None:
            for attr, cluster_indices in self.attribute_action_mapping.items():
                if action-1 in cluster_indices:
                    attribute = attr
                    break
        
        if action == len(self.cluster_action_mappping) + 1:
            return len(self.cluster_action_mappping) + 1, 1, True, processed_data, None, None, None
        
        if training:
            cluster = self.cluster_action_mappping[action-1]
            partition_index = np.random.choice(cluster)
        else:
            if action-1 not in self.sampled_cluster_partitions:
                #raise ValueError(f"Action {action-1} not in sampled_cluster_partitions")
                return action, 0, True, processed_data, None, None, None
            sampled_partition_indecies = self.sampled_cluster_partitions[action-1]
            partition_index = np.random.choice(sampled_partition_indecies)
        #print(f"Action: {action}, Attribute: {attribute}, Partition index: {partition_index}")
        partition = self.search_space[attribute].candidates[partition_index]
        bins = partition.bins
        if self.update_mode == "every-step":
            if self.use_case == 'modeling': accuracy, processed_data = self.explainable_modeling(processed_data, self.y_col, attribute, bins)
            elif self.use_case == 'imputation': accuracy, processed_data = self.data_imputation(processed_data, self.y_col, attribute, bins, self.imputer)
            elif self.use_case == 'causal_inference': accuracy, processed_data = causal_inference_utility(processed_data, self.y_col, {attribute:bins}, self.confounders, self.treatment_attr, self.causal_case,self.unbinned_pre_utility,after_bin_flag=True)
            else: raise ValueError(f"Use case {self.use_case} not supported. Use 'modeling' or 'imputation'.")
        elif self.update_mode == "final-only":
            # For final-only mode, we don't compute the accuracy here, we do it at the end of the episode
            accuracy = 0
        
        # Calculate the reward
        #grade = get_semantic_grade(attribute, bins)
        semantic = partition.get_semantic_score(self.semantic_metric)
        #if self.semantic_metric == 'gpt_semantics':
            # Divide by 4 to normalize the grade
            # This is because the GPT grader returns a score between 0 and 4
            # We want to normalize it to be between 0 and 1
            #semantic = partition.gpt_semantics/4
        #print("Attribute: ", attribute, "Grade: ", semantic)
        reward = (accuracy + semantic) / 2
        #print("Reward: ", reward)

        return action, reward, False, processed_data, partition, partition_index, attribute

    def _evaluate_partition(self, attribute, arm, part_idx):
        """
        Helper for single-attribute bandit case.
        Computes reward, handles bookkeeping, and returns reward value.
        """
        processed_data = self.data.copy()

        # concrete partition & bins
        partition   = self.search_space[attribute].candidates[part_idx]
        partition_dict = {attribute: partition.bins}

        # utility & semantic
        util = visualization_spearmanr(processed_data, self.y_col, partition_dict)
        sem  = partition.get_semantic_score(self.semantic_metric)
        #if self.semantic_metric == 'gpt_semantics':
        #    sem /= 4

        reward = (util + sem) / 2

        # store for later Pareto analysis
        partition_dict["__semantic__"] = sem
        partition_dict["__utility__"]  = util
        self.train_partitions.append(partition_dict)

        return reward

    def train_single(self):
        """
        Single-attribute training.
        Implements vanilla UCB1 with
        •  one mandatory “initialisation pull’’ of every arm,
        •  book-keeping to avoid re-sampling the same partition,
        •  automatic skip of arms whose partitions are exhausted.
        """
        # ------------------------------------------------------------
        # 0.  Sanity: single attribute only
        # ------------------------------------------------------------
        assert len(self.search_space) == 1, \
            "train_single should be called only when exactly one attribute exists"

        attribute = list(self.search_space.keys())[0]
        k         = len(set(self.cluster_assignments[0]))      # # clusters = # arms
        T         = self.n_training_episodes                   # total pulls / budget

        # Q-values (mean rewards) and pull-counts n_i
        self.Qtable  = np.zeros(k)
        self.counts  = np.zeros(k, dtype=int)

        # For every arm store the list of partition indices that are still unused
        remaining_partitions = {
            arm: list(self.cluster_action_mappping[arm]) for arm in range(k)
        }
        # For statistics / LLM-call count
        self.sampled_cluster_partitions = {}

        # ------------------------------------------------------------
        # 1.  INITIALISATION — pull each arm once
        # ------------------------------------------------------------
        for arm in range(min(k, T + 1)):
            if not remaining_partitions[arm]:                # defensive
                continue
            part_idx = random.choice(remaining_partitions[arm])
            remaining_partitions[arm].remove(part_idx)

            reward = self._evaluate_partition(attribute, arm, part_idx)
            self.counts[arm]  = 1
            self.Qtable[arm]  = reward
            #print(f"Initial pull for arm {arm} (partition {part_idx}): reward = {reward}")
            self.sampled_cluster_partitions.setdefault(arm, []).append(part_idx)

        # ------------------------------------------------------------
        # 2.  Main UCB loop  (T − k remaining pulls)
        # ------------------------------------------------------------
        for t in range(k + 1, T + 1):

            # --- build list of arms that still have unseen partitions
            viable_arms = [a for a,pool in remaining_partitions.items() if pool]

            if not viable_arms:               # all clusters exhausted
                break

            # --- UCB1 scores only over viable arms
            bonuses = np.sqrt((2 * np.log(t)) / np.maximum(1, self.counts))
            ucb_scores = {a: self.Qtable[a] + bonuses[a] for a in viable_arms}

            # choose arm with highest score
            arm = max(ucb_scores, key=ucb_scores.get)

            # --- sample a partition not seen before -----------------
            part_idx = random.choice(remaining_partitions[arm])
            remaining_partitions[arm].remove(part_idx)

            # if cluster is now empty mark it non-viable next round
            reward = self._evaluate_partition(attribute, arm, part_idx)

            # --- incremental-mean update ----------------------------
            self.counts[arm] += 1
            n = self.counts[arm]
            self.Qtable[arm] += (reward - self.Qtable[arm]) / n
            #print(f"Pull for arm {arm} (partition {part_idx}): reward = {reward}, counts = {self.counts[arm]}, Q-value = {self.Qtable[arm]}")

            self.sampled_cluster_partitions.setdefault(arm, []).append(part_idx)

        
        # No episodic returns dictionary needed in bandit mode
        return self.Qtable, self.sampled_cluster_partitions, self.train_partitions

    def train(self):
        """
        Train the algorithm.

        •  If only one attribute is present, the routine collapses to a classic
        UCB1 loop (MCC ≡ UCB): each training episode is a single arm-pull and
        Q-values are updated online.

        •  Otherwise it executes the original episodic MCC training.
        """

        if len(self.search_space) == 1:
            # Single-attribute case
            return self.train_single()
        
        # ------------------------------------------------------------------
        # === Multi-attribute case  →  original MCC logic ==================
        # ------------------------------------------------------------------
        for episode_n in range(self.n_training_episodes):
            processed_data = self.data.copy()
            episode = [] # save (state, action, reward) tuples for the episode
            # Reset the environment
            state = 0
            step = 0
            done = False
            visited = set()
            visited.add(state)
            partition_dict = {}
            total_semantic_score = 0

            for step in range(self.max_steps):
    
                action = self.epsilon_greedy_policy(state, episode_n, step_count=step, attributes_ep=list(partition_dict.keys()))
                #print("Action: ", action)
                
                new_state, reward, done, processed_data, partition, sampled_partition_index, attribute = self.step_func(processed_data, action, training=True)
                episode.append((state, action, reward))

                # Enforce no repeated states
                if new_state in visited: break
                visited.add(new_state)

                # Update the sampled cluster partitions
                if action-1 not in self.sampled_cluster_partitions:
                    self.sampled_cluster_partitions[action-1] = []
                self.sampled_cluster_partitions[action-1].append(sampled_partition_index)
                # Update the partition dictionary
                if attribute is not None:
                    partition_dict[attribute] = partition.bins
                    total_semantic_score += partition.get_semantic_score(self.semantic_metric)

                # If done, finish the episode
                if done or len(partition_dict) == self.max_steps:
                    #print("Episode ", episode, " finished after ", step, " steps") 
                    break

                # Our state is the new state
                state = new_state
            
            if self.update_mode == "every-step":
                G = 0
                gamma = 1  # Discount factor, can be tuned
                # Update the Q table
                for t in range(len(episode)-1, -1, -1):
                    state_t, action_t, reward_t = episode[t]
                    G = reward_t + gamma * G
                    self.returns[(state_t, action_t)].append(G)
                    self.Qtable[state_t][action_t] = np.mean(self.returns[(state_t, action_t)])
            
            elif self.update_mode == "final-only":
                # Update the Q table only at the end of the episode
                if self.use_case == 'imputation':
                    final_reward = data_imputation_multi_attrs(processed_data, self.y_col, partition_dict, self.imputer, self.lut)
                elif self.use_case=='modeling':
                    final_reward = explainable_modeling_multi_attrs(processed_data, self.y_col, partition_dict, self.lut)
                elif self.use_case=='causal_inference':
                    final_reward = causal_inference_utility(processed_data, self.y_col, partition_dict, self.confounders, self.treatment_attr, self.causal_case,self.unbinned_pre_utility)
                elif self.use_case == 'visualization':
                    final_reward = visualization_spearmanr(processed_data, self.y_col, partition_dict)
                else: raise ValueError(f"Use case {self.use_case} not supported. Use 'modeling', 'imputation', or 'visualization'.")
                total_semantic_score = total_semantic_score / self.max_steps  # Average semantic score over all attributes in the episode
                partition_dict["__semantic__"] = total_semantic_score
                partition_dict["__utility__"] = final_reward
                self.train_partitions.append(partition_dict)
                #if self.semantic_metric == 'gpt_semantics':
                    # Divide by 4 to normalize the grade
                    # This is because the GPT grader returns a score between 0 and 4
                    # We want to normalize it to be between 0 and 1
                    #total_semantic_score /= 4
                final_reward = (final_reward + total_semantic_score) / 2
                for state_t, action_t, _ in episode:
                    self.returns[(state_t, action_t)].append(final_reward)
                    self.Qtable[state_t][action_t] = np.mean(self.returns[(state_t, action_t)])
        
        if "__utility__" not in self.gold_standard:
            if self.use_case == 'imputation':
                self.gold_standard["__utility__"] = data_imputation_multi_attrs(self.data.copy(), self.y_col, self.gold_standard, self.imputer, self.lut)
            elif self.use_case == 'modeling':
                self.gold_standard["__utility__"] = explainable_modeling_multi_attrs(self.data.copy(), self.y_col, self.gold_standard, self.lut)
            elif self.use_case == 'causal_inference':
                self.gold_standard["__utility__"] = causal_inference_utility(self.data.copy(), self.y_col, self.gold_standard, self.confounders, self.treatment_attr, self.causal_case,self.unbinned_pre_utility)
            elif self.use_case == 'visualization':
                self.gold_standard["__utility__"] = visualization_spearmanr(self.data.copy(), self.y_col, self.gold_standard)
            else: raise ValueError(f"Use case {self.use_case} not supported. Use 'modeling', 'imputation', or 'causal'.")

            if self.semantic_metric == 'gpt_semantics':
                self.gold_standard["__semantic__"] = 4  # Assuming the gold standard has a semantic score of 4
            else: self.gold_standard["__semantic__"] = 1
        self.train_partitions.append(self.gold_standard)
        return self.Qtable, self.sampled_cluster_partitions, self.train_partitions

    def get_estimated_P(self):
        # Find the pareto front
        semantic_scores = [p["__semantic__"] for p in self.train_partitions]
        utility_scores = [p["__utility__"] for p in self.train_partitions]
        datapoints = np.array([semantic_scores, utility_scores])
        lst = compute_pareto_front(datapoints)
        est_partitions = [self.train_partitions[i] for i in lst]
        if self.semantic_metric == 'gpt_semantics':
            for p in est_partitions:
                p["__semantic__"] = p["__semantic__"] / 4
        # Get the unique points in the estimated pareto front
        est_points = np.array([np.array([p["__semantic__"] for p in est_partitions]),
                               np.array([p["__utility__"] for p in est_partitions])])
        est_points = np.array(est_points).T
        est_points = np.unique(est_points, axis=0)
        # Turn the ny array in each partition to a list
        for p in est_partitions:
            for attr in p.keys():
                if isinstance(p[attr], np.ndarray):
                    p[attr] = p[attr].tolist()
        return est_points, est_partitions

    def run(self, cached_ID, n_runs=20):
        """
        Run the MCC algorithm.
        """
        # Initialize the lists to store the results
        cached_results = {cutoff: {} for cutoff in self.training_cutoffs}
        gd_list = {cutoff: [] for cutoff in self.training_cutoffs}
        igd_list = {cutoff: [] for cutoff in self.training_cutoffs}
        ahd_list = {cutoff: [] for cutoff in self.training_cutoffs}
        
        for round in range(n_runs):
            start_time = time.time()
            self.cluster_assignments = self.get_cluster_assignments_multi_attr()
            self.cluster_action_mappping, self.attribute_action_mapping = self.get_candidate_action_mapping()
            self.Qtable = self.initialize_Q()
            self.Qtable = self.mask_Qtable()
            print(f"Round {round+1}/{n_runs}")
            self.sampled_cluster_partitions = {}
            self.train_partitions = []
            # Returns are stored for averaging
            self.returns = defaultdict(list)

            for cutoff_i, training_cutoff in enumerate(self.training_cutoffs):
                if len(self.search_space) == 1:
                    self.n_training_episodes = training_cutoff
                else:
                    self.n_training_episodes = training_cutoff - self.training_cutoffs[cutoff_i-1] if cutoff_i > 0 else training_cutoff
                self.train()
                print("Training completed.")
                est_P, est_partitions = self.get_estimated_P()
                print(f"Estimated Pareto front: {est_P}")
                end_time = time.time()

                # Compute distances
                n_llm_calls = Evaluator.get_num_of_llm_calls(self.sampled_cluster_partitions)
                if self.evaluator is not None:
                    gd_value = self.evaluator.gd(est_P)
                    igd_value = self.evaluator.igd(est_P)
                    ahd_value = Evaluator.average_hausdorff_distance(gd_value, igd_value, mode='max')
                    gd_list[training_cutoff].append(gd_value)
                    igd_list[training_cutoff].append(igd_value)
                    ahd_list[training_cutoff].append(ahd_value)
                    print(f"GD: {gd_value:.2f}, IGD: {igd_value:.2f}, AHD: {ahd_value:.2f}")
                else:
                    gd_value = 'No ground truth data provided.'
                    igd_value = 'No ground truth data provided.'
                    ahd_value = 'No ground truth data provided.'
                    print("No ground truth data provided, skipping distance calculations.")
            
                cached_results[training_cutoff][round] = {
                    'update_mode': self.update_mode,
                    'n_train_episodes': training_cutoff, # budget, n_utility_calls
                    'max_steps': self.max_steps, 
                    'epsilon': self.epsilon, 
                    'epsilon_schedule': self.epsilon_schedule,
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
        
        result = []
        for cutoff in self.training_cutoffs:
            if self.evaluator is not None:
                
                # Print the mean, median and std of the distances
                print("GD: ", np.mean(gd_list[cutoff]), np.median(gd_list[cutoff]), np.std(gd_list[cutoff]))
                print("IGD: ", np.mean(igd_list[cutoff]), np.median(igd_list[cutoff]), np.std(igd_list[cutoff]))
                print("HD: ", np.mean(ahd_list[cutoff]), np.median(ahd_list[cutoff]), np.std(ahd_list[cutoff]))
                result.append({
                    'cached_ID': cached_ID,
                    'update_mode': self.update_mode,
                    'n_train_episodes': cutoff,  # budget, n_utility_calls
                    'max_steps': self.max_steps, 
                    'epsilon': self.epsilon, 
                    'epsilon_schedule': self.epsilon_schedule,
                    't': self.t,
                    "GD mean": np.mean(gd_list[cutoff]),
                    "GD median": np.median(gd_list[cutoff]),
                    "GD std": np.std(gd_list[cutoff]),
                    "IGD mean": np.mean(igd_list[cutoff]),
                    "IGD median": np.median(igd_list[cutoff]),
                    "IGD std": np.std(igd_list[cutoff]),
                    "AHD mean": np.mean(ahd_list[cutoff]),
                    "AHD median": np.median(ahd_list[cutoff]),
                    "AHD std": np.std(ahd_list[cutoff])
                })
            else:
                print("No ground truth data provided, skipping distance calculations.")
                result.append({
                    'cached_ID': cached_ID,
                    'update_mode': self.update_mode,
                    'n_train_episodes': cutoff,  # budget, n_utility_calls
                    'max_steps': self.max_steps, 
                    'epsilon': self.epsilon, 
                    'epsilon_schedule': self.epsilon_schedule,
                    't': self.t,
                    #'estimated_P': est_P.tolist(),
                    #'estimated_partitions': est_partitions,
                    "GD mean": 'No ground truth data provided.',
                    "GD median": 'No ground truth data provided.',
                    "GD std": 'No ground truth data provided.',
                    "IGD mean": 'No ground truth data provided.',
                    "IGD median": 'No ground truth data provided.',
                    "IGD std": 'No ground truth data provided.',
                    "AHD mean": 'No ground truth data provided.',
                    "AHD median": 'No ground truth data provided.',
                    "AHD std": 'No ground truth data provided.',
                    #'n_llm_calls': Evaluator.get_num_of_llm_calls(self.sampled_cluster_partitions),
                })
        return result, cached_results