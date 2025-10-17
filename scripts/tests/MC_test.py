"""
Monte Carlo for multi-attribute explainable modeling
"""

import sys
import os
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.framework.discretizers import *
from scripts.framework.SearchSpace import *
from scripts.framework.utils import *
from scripts.framework.UCB import *
from scripts.framework.framework_utils import *
from grader.gpt_grader import evaluate_groupings_gpt

def explainable_modeling_partition_dict(data, y_col, partition_dict) -> float:
    """
    Wrapper function to model the data using an explainable model
    ***** Note: This function is only used in demo_data_modeling_case.ipynb for now *****
    :param data: DataFrame
    :param y_col: str
    :param partition_dict: Dict[str, Partition]
    :return: float
    """
    start_time = time.time()
    
    for attr, partition in partition_dict.items():
        #print(f"Discretizing {attr}...")
        data[attr + '.binned'] = pd.cut(data[attr], bins=partition, labels=False)
        data[attr + '.binned'] = data[attr + '.binned'].astype('float64')
        data = data.dropna(subset=[attr + '.binned'])
    
    
    # Evaluate the explainable model
    X_cols = [col for col in data.columns if col != y_col and col not in list(partition_dict.keys())]
    X = data[X_cols]
    y = data[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a decision tree model
    model = DecisionTreeClassifier(random_state=0, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)

    return model_accuracy

def get_cluster_assignments_multi_attr(space_list, parameters) -> List:
    """
    :param space_list: List of PartitionSearchSpace
    :return: List of clusters
    """
    assignments_list = []
    for space in space_list:
        assignments = linkage_distributions(space, parameters)
        assignments_list.append(assignments)
        print("There are ", len(np.unique(assignments)), " unique clusters")
    return assignments_list

def get_candidate_action_mapping(space_list, cluster_assignments, attributes):
    """
    :param space_list: List of PartitionSearchSpace
    :param cluster_assignments: List of clusters
    :return: List of candidate actions
    :return: Dictionary of attribute cluster mapping
    """
    candidate_action_mapping = []
    attribute_cluster_mapping = {}
    count = 0
    for i, space in enumerate(space_list):
        cluster_indices = []
        candidate_indices = []
        for cluster in np.unique(cluster_assignments[i]):  
            candidate_indices = np.where(cluster_assignments[i] == cluster)[0]
            candidate_action_mapping.append(candidate_indices)
            cluster_indices.append(count)
            count += 1
        attribute_cluster_mapping[attributes[i]] = cluster_indices
    print("Candidate action mapping: ", candidate_action_mapping)
    print("Attribute cluster mapping: ", attribute_cluster_mapping)
    return candidate_action_mapping, attribute_cluster_mapping

def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable

def mask_Qtable(Qtable, cluster_assignments, debug=True):
    """
      Mask the Q table to only allow actions that are possible in the current state.
      Only states that are for the same attribute are not allowed.
    """
    curr_attr = 0
    num_unique_clusters = len(np.unique(cluster_assignments[curr_attr]))
    curr_attr_start_index = 1
    curr_attr_end_index = num_unique_clusters + 1
    for state in range(1, len(Qtable)):
        if state == curr_attr_end_index:
            if debug:
                print("State: ", state)
                print("curr_attr: ", curr_attr)
                print("curr_attr_start_index: ", curr_attr_start_index)
            curr_attr += 1
            if curr_attr < len(cluster_assignments):
                num_unique_clusters = len(np.unique(cluster_assignments[curr_attr]))
                curr_attr_start_index = curr_attr_end_index
                curr_attr_end_index += num_unique_clusters
            else: curr_attr_start_index = curr_attr_end_index
        Qtable[state][curr_attr_start_index:curr_attr_end_index] = -np.inf
        #Qtable[state][0:curr_attr_end_index] = -np.inf
    
    Qtable[1:,-1] = 0
    Qtable[:, 0] = -np.inf
    Qtable[0, -1] = -np.inf # Have to bin something
    Qtable[-1,:-1] = -np.inf
    if debug: print("Qtable after masking: \n", Qtable)
    return Qtable

def epsilon_greedy_policy(Qtable, state, epsilon):
    random_int = random.uniform(0,1)
    if random_int > epsilon:
        action = np.argmax(Qtable[state])
    else:
        #action = env.action_space.sample()
        # TODO: Implement random action, but only from the possible actions
        # Only pick actions that are not -inf
        possible_actions = np.where(Qtable[state] != -np.inf)[0]
        action = np.random.choice(possible_actions)
    return action

def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[state])
    return action

def greedy_top_N_policy(Qtable, state, attributes_ep, attribute_action_mapping, N=3):
    """
    Select the top 3 actions with the highest Q values
    """
    top_actions_index = np.where(Qtable[state] != -np.inf)[0]  # Get the indices of the actions that are not -inf
    #print("Top actions index: ", top_actions_index)
    for attr in attributes_ep:
        if attr in attribute_action_mapping:
            # If the attribute is already in the episode, we don't want to select actions for it
            # This is to avoid selecting actions for the same attribute multiple times
            top_actions_index = [a for a in top_actions_index if a-1 not in attribute_action_mapping[attr]] # Exclude actions that are for the already-chosen-for attribute
    if len(attributes_ep) < 3: top_actions_index = top_actions_index[:-1] # Exclude the last action which is the "do nothing" action
    #print("Top actions index after filtering: ", top_actions_index)
    if len(top_actions_index) > N:
        top_actions = np.argsort(Qtable[state][top_actions_index])[-N:] 
    else: top_actions = np.argsort(Qtable[state][top_actions_index])[-len(top_actions_index):] # Get the indices of the top 3 actions
    top_actions_index = np.array(top_actions_index)  # Convert to numpy array for indexing
    top_actions = top_actions_index[top_actions]
    #print("Top actions values: ", Qtable[state][top_actions_index])
    #print("Top actions: ", top_actions)
    action = np.random.choice(top_actions)  # Randomly select one of the top 3 actions
    return action

def explainable_modeling_multi_attrs(data, y_col, attr:str, bins) -> float:
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

def get_semantic_grade(attr:str, bins) -> float:
    """
    Wrapper function to calculate the semantic grade of the bins
    :param attr: str
    :param bins: List
    :return: float
    """
    input = [(attr, create_intervals(bins), (0, float('inf')))]
    print("Input: ", input)
    results = evaluate_groupings_gpt(input)
    # Print results for each test case in the group
    for (feature, grouping, feature_range), (grade, explanation, reference_count, reference_links) in zip(input, results):
        return int(grade) / 4 # 4 is the maximum grade

def step_func(data, semantic_metric, action, cluster_action_mappping, attribute_action_mapping, space_dict, y_col, training=True, sampled_cluster_partitions=None):
    # Find what attribute and cluster the action corresponds to, from the action mapping
    attribute = None
    for attr, cluster_indices in attribute_action_mapping.items():
        if action-1 in cluster_indices:
            attribute = attr
            break
    
    if action == len(cluster_action_mappping) + 1:
        return len(cluster_action_mappping) + 1, 1, True, data, None, None, None
    
    if training:
        cluster = cluster_action_mappping[action-1]
        partition_index = np.random.choice(cluster)
    else:
        if action-1 not in sampled_cluster_partitions:
            #raise ValueError(f"Action {action-1} not in sampled_cluster_partitions")
            return action, 0, True, data, None, None, None
        sampled_partition_indecies = sampled_cluster_partitions[action-1]
        partition_index = np.random.choice(sampled_partition_indecies)
    partition = space_dict[attribute].candidates[partition_index]
    bins = partition.bins
    accuracy, data = explainable_modeling_multi_attrs(data, y_col, attribute, bins)
    
    # Calculate the reward
    #grade = get_semantic_grade(attribute, bins)
    if semantic_metric == 'l2_norm':
        semantic = partition.l2_norm
    elif semantic_metric == 'KLDiv':
        semantic = partition.KLDiv
    elif semantic_metric == 'gpt_semantics':
        # Divide by 4 to normalize the grade
        semantic = partition.gpt_semantics / 4
    #print("Attribute: ", attribute, "Grade: ", semantic)
    reward = (accuracy + semantic) / 2
    #print("Reward: ", reward)

    return action, reward, False, data, partition, partition_index, attribute

def train(data, semantic_metric, n_training_episodes, epsilon, max_steps, Qtable, candidate_action_mappping, attribute_action_mapping, space_dict, y_col):

    sampled_cluster_partitions = {}
    # Returns are stored for averaging
    returns = defaultdict(list)

    for episode_n in range(n_training_episodes):
        processed_data = data.copy()
        episode = [] # save (state, action, reward) tuples for the episode
        # Reset the environment
        state = 0
        step = 0
        done = False
        visited = set()
        visited.add(state)

        # repeat
        for step in range(max_steps):
   
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            #print("Action: ", action)
            step += 1
            
            new_state, reward, done, processed_data, _, sampled_partition_index, _ = step_func(processed_data, semantic_metric, action, candidate_action_mappping, attribute_action_mapping, space_dict, y_col)
            episode.append((state, action, reward))

            # Enforce no repeated states
            if new_state in visited: break
            visited.add(new_state)

            # Update the sampled cluster partitions
            if action-1 not in sampled_cluster_partitions:
                sampled_cluster_partitions[action-1] = []
            sampled_cluster_partitions[action-1].append(sampled_partition_index)

            # If done, finish the episode
            if done or step == max_steps:
                #print("Episode ", episode, " finished after ", step, " steps") 
                break

            # Our state is the new state
            state = new_state
        
        G = 0
        gamma = 1  # Discount factor, can be tuned
        # Update the Q table
        for t in range(len(episode)-1, -1, -1):
            state_t, action_t, reward_t = episode[t]
            G = reward_t + gamma * G
            returns[(state_t, action_t)].append(G)
            Qtable[state_t][action_t] = np.mean(returns[(state_t, action_t)])
    
    return Qtable, sampled_cluster_partitions

def evaluate_agent(data, semantic_metric, max_steps, n_eval_episodes, Q, candidate_action_mappping, attribute_action_mapping, space_dict, y_col, sampled_cluster_partitions):

    episode_rewards = []
    strategies = []
    attempts = 0
    while len(strategies) < n_eval_episodes and attempts < 10 * n_eval_episodes:
        attempts += 1
        episode_steps = {} # steps in the episode, used to reconstruct the strategy
        attributes_ep = [] # attributes used in the episode so far
        processed_data = data.copy()
        state = 0
        step = 0
        done = False
        total_rewards_ep = 0
        visited = set()
        visited.add(state)
   
        for step in range(max_steps):
            # Take the action (index) that have the maximum reward
            #action = np.argmax(Q[state][:])
            action = greedy_top_N_policy(Q, state, attributes_ep, attribute_action_mapping, N=3)
            #print("State: ", state, "Action: ", action)
            new_state, reward, done, processed_data, partition, _, attribute = step_func(processed_data, semantic_metric, action, candidate_action_mappping, attribute_action_mapping, space_dict, y_col, False, sampled_cluster_partitions)
            if attribute is not None: 
                attributes_ep.append(attribute)
                episode_steps[attribute] = partition
                total_rewards_ep += reward
       
            if done or new_state in visited: break
            
            visited.add(new_state)
            state = new_state
        
        # Only add the episode (valid only) if it has the maximum number of steps
        if len(attributes_ep) == max_steps:
            episode_rewards.append(total_rewards_ep)
            strategies.append(episode_steps)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward, strategies

def episode_to_strategy(episode, space_dict, cluster_action_mappping, attribute_action_mapping):
    """
    Turn an episode into a strategy
    :param episode: List
    :param space_dict: Dict
    :param cluster_action_mappping: List
    :param attribute_action_mapping: Dict
    :return: List[Partition]
    """
    strategy = []
    for action in episode:
        space = None
        for attr, cluster_indices in attribute_action_mapping.items():
            if action-1 in cluster_indices:
                space = space_dict[attr]
                break
        if space is None: continue

        cluster = cluster_action_mappping[action-1]
        partition_index = np.random.choice(cluster)
        partition = space.candidates[partition_index]
        strategy.append(partition)
    return strategy

def evaluate_strategies(strategies, y_col, raw_data, semantic_metric):
    partition_dicts = []
    attributes = None
    for s in strategies:
        semantic_score = 0
        p_dict = {}
        if attributes is None: 
            attributes = list(s.keys())
            # sort alphabetically
            attributes.sort()
        for attr in s.keys():
            partition = s[attr]
            if semantic_metric == 'l2_norm':
                semantic_score += partition.l2_norm
            elif semantic_metric == 'KLDiv':
                semantic_score += partition.KLDiv
            elif semantic_metric == 'gpt_semantics':
                # Divide by 4 to normalize the grade
                semantic_score += partition.gpt_semantics
            p_dict[attr] = partition.bins
        #print(p_dict)
        p_dict = dict(sorted(p_dict.items()))  # Sort the dictionary by keys (attributes alphabetically)
        partition_dicts.append({"Partition": p_dict, "Semantic": semantic_score / 3})
    
    partition_dicts.append({"Partition": {'Age': np.array([0, 18, 35, 45, 65, 100]), 'BMI': np.array([0, 18.5, 25, 30, 68]), 'Glucose': np.array([0, 140, 200])}, "Semantic": 4})
    
    #raw_data = raw_data[attributes + [y_col]]
    for partition_dict in partition_dicts:
        data_i = raw_data.copy()
        utility_score = explainable_modeling_partition_dict(data_i, y_col, partition_dict["Partition"])
        partition_dict["Utility"] = utility_score
        bins_list = []
        for attr in attributes:
            bins_list.append(partition_dict["Partition"][attr])
        partition_dict["Partition"] = bins_list
        #print(partition_dict)
    
    # turn partition_dicts into a dataframe
    partition_df = pd.DataFrame(partition_dicts, index=None)

    # Find the pareto front
    datapoints = [np.array(partition_df['Semantic'].values), np.array(partition_df['Utility'].values)]
    lst = compute_pareto_front(datapoints)
    partition_df["Estimated"] = 0
    partition_df.loc[lst, "Estimated"] = 1
    partition_df["Explored"] = 1
    original_len = len(partition_df)
    # Remove duplicates
    partition_df = partition_df.drop_duplicates(subset=['Partition'])
    print(f"Original number of partitions: {original_len}, Number of unique partitions: {len(partition_df)}")
    return partition_df

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

if __name__ == '__main__':
    ppath = sys.path[0] + '/../../'
    dataset = 'titanic'
    use_case = 'modeling'
    semantic_metric = 'gpt_semantics'
    # Read json file
    exp_config = json.load(open(os.path.join(ppath, 'code', 'configs', f'{dataset}.json')))
    attributes = list(exp_config['attributes'].keys())
    
    # Load experiment data
    space_dict = {}
    for attr in attributes:
        data = pd.read_csv(os.path.join(ppath, 'exp_data_grader', dataset, use_case, f'{attr}.csv'))
        # Filter data by gpt_prompt > 0
        ss = TestSearchSpace(data)
        space_dict[attr] = ss
    
    y_col = exp_config['target']
    raw_data = pd.read_csv(os.path.join(ppath, exp_config['data_path']))
    raw_data = raw_data[exp_config['features'] + [exp_config['target']]]
    raw_data = raw_data.dropna(subset=exp_config['features'] + [exp_config['target']])
    cluster_params = {'t': 0.5, 'criterion': 'distance'}
    cluster_assignments = get_cluster_assignments_multi_attr(list(space_dict.values()), cluster_params)
    candidate_action_mappping, attribute_action_mapping = get_candidate_action_mapping(list(space_dict.values()), cluster_assignments, attributes)
    #cluster_assignments = [[1,2],[1,2,3]]
    
    # Load ground truth pareto front
    gt_df = pd.read_csv(os.path.join(ppath, 'testresults', f'{dataset}.multi_attrs_exhaustive_search.{use_case}.csv'))
    datapoints = [np.array(gt_df[semantic_metric].values), np.array(gt_df['utility'].values)]
    lst = compute_pareto_front(datapoints)
    gt_df["Estimated"] = 0
    gt_df.loc[lst, "Estimated"] = 1
    gt_df["Explored"] = 1
    gt_df = gt_df[gt_df['Estimated'] == 1]
    gt_df = gt_df.drop_duplicates(subset=[semantic_metric, 'utility']) # remove duplicates
    gt_points = [np.array(gt_df[semantic_metric].values), np.array(gt_df['utility'].values)]
    gt_points = np.array(gt_points).T
    gd= GD(gt_points)
    igd = IGD(gt_points)

    # Logging
    columns = ['n_train_episodes', 'max_steps', 'epsilon', 'GD', 'IGD', 'AHD']
    f_results = pd.DataFrame(columns=columns)
    training_episodes = [500, 1000, 2000, 5000, 10000]
    epsilons = [0.7, 0.5, 0.3, 0.1]

    for n_training_episodes in training_episodes:
        for epsilon in epsilons:
            print(f"Training episodes: {n_training_episodes}, Epsilon: {epsilon}")
            # Run the Monte Carlo control
            
            # MCC
            # Training parameters
            #n_training_episodes = 2000
            
            # Evaluation parameters
            if n_training_episodes < 2000:
                n_eval_episodes = n_training_episodes
            else: n_eval_episodes = 2000

            # Environment parameters
            max_steps = 3
            
            # Exploration parameters
            #epsilon = 0.5 # high to encourage exploration.

            gd_list = []
            igd_list = []
            hd_list = []
            for round in range(10):
                # Initialize Q table
                state_space = sum([len(np.unique(assignments)) for assignments in cluster_assignments]) + 2
                #print("There are ", state_space, " possible states")
                action_space = sum([len(np.unique(assignments)) for assignments in cluster_assignments]) + 2
                #print("There are ", action_space, " possible actions")
                Qtable = initialize_q_table(state_space, action_space)

                # Train Q table
                Qtable = mask_Qtable(Qtable, cluster_assignments, debug=True)
                Qtable, sampled_cluster_partitions = train(raw_data, semantic_metric, n_training_episodes, epsilon, max_steps, Qtable, candidate_action_mappping, attribute_action_mapping, space_dict, y_col)
                print("Qtable after training: \n", Qtable)
                
                # Evaluate Q table
                mean_reward, std_reward, strategies = evaluate_agent(raw_data, semantic_metric, max_steps, n_eval_episodes, Qtable, candidate_action_mappping, attribute_action_mapping, space_dict, y_col, sampled_cluster_partitions)
                print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                #for i, strategy in enumerate(strategies):
                #    print(f"Strategy {i+1}: {strategy}")
                Qtable_df = pd.DataFrame(Qtable)
                Qtable_df.to_csv(os.path.join(ppath, 'scratch', f"{dataset}.{use_case}.{semantic_metric}.Qtable.csv"))

                # Evaluate strategies
                partition_df = evaluate_strategies(strategies, y_col, raw_data, semantic_metric)
                partition_df = partition_df[partition_df['Estimated'] == 1]
                #partition_df.to_csv(os.path.join(ppath, 'scratch', f"{dataset}.{use_case}.{semantic_metric}.strategies.csv"))
                est_points = [np.array(partition_df['Semantic'].values), np.array(partition_df['Utility'].values)]
                est_points = np.array(est_points).T
                est_points = np.unique(est_points, axis=0)
                print("Estimated points: ", est_points)
                # Compute distances
                gd_i = gd(est_points)
                igd_i = igd(est_points)
                hd_i = average_hausdorff_distance(gd_i, igd_i, mode='max')
                gd_list.append(gd_i)
                igd_list.append(igd_i)
                hd_list.append(hd_i)
                print(f"GD: {gd_i:.2f}, IGD: {igd_i:.2f}, AHD: {hd_i:.2f}")
            
            # Print the mean, median and std of the dist
            print("GD: ", np.mean(gd_list), np.median(gd_list), np.std(gd_list))
            print("IGD: ", np.mean(igd_list), np.median(igd_list), np.std(igd_list))
            print("AHD: ", np.mean(hd_list), np.median(hd_list), np.std(hd_list))
            # Save evaluation results
            eval_result = {
                'n_train_episodes': n_training_episodes,
                'max_steps': max_steps,
                'epsilon': epsilon,
                'GD': np.mean(gd_list),
                'IGD': np.mean(igd_list),
                'AHD': np.mean(hd_list)
            }
            f_results = pd.concat([f_results, pd.DataFrame([eval_result])], ignore_index=True)
    f_results.to_csv(os.path.join(ppath, 'scratch', f"{dataset}.{use_case}.{semantic_metric}.eval_results.csv"), index=False)
    #save_eval_results(eval_results, env_id)
    