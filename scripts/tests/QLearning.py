"""
This script is to test q-learning with multiple attributes.
"""
import sys
import os

ppath = sys.path[0] + '/../../'
sys.path.append(os.path.join(ppath, 'code'))
sys.path.append(os.path.join(ppath, 'grader'))
sys.path.append(os.path.join(ppath, 'code', 'framework'))

from import_packages import *
from discretizers import *
from SearchSpace import *
from utils import *
from framework_utils import *
from gpt_grader import evaluate_groupings_gpt

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

# MC control utils
def generate_episode(Qtable):
    pass

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
        return len(cluster_action_mappping) + 1, 1, True, data, None, None
    
    if training:
        cluster = cluster_action_mappping[action-1]
        partition_index = np.random.choice(cluster)
    else:
        sampled_partition_indecies = sampled_cluster_partitions[action-1]
        partition_index = np.random.choice(sampled_partition_indecies)
    partition = space_dict[attribute].candidates[partition_index]
    bins = partition.bins
    accuracy, data = explainable_modeling_multi_attrs(data, y_col, attribute, bins)
    
    ################################
    #### Normalize the accuracy ####
    ################################
    accuracy = (accuracy - 0.5) / 0.5
    ################################
    
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

    return action, reward, False, data, partition, partition_index

def train(data, semantic_metric, n_training_episodes, min_epsilon, max_epsilon, learning_rate, gamma, decay_rate, max_steps, Qtable, candidate_action_mappping, attribute_action_mapping, space_dict, y_col):

    sampled_cluster_partitions = {}

    for episode in range(n_training_episodes):
        processed_data = data.copy()
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        #epsilon = max(min_epsilon, epsilon * decay_rate)
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
            
            new_state, reward, done, processed_data, _, sampled_partition_index = step_func(processed_data, semantic_metric, action, candidate_action_mappping, attribute_action_mapping, space_dict, y_col)
            
            # Enforce no repeated states
            if new_state in visited: break
            visited.add(new_state)

            # Q-learning update
            Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])
            
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
    
    return Qtable, sampled_cluster_partitions

def evaluate_agent(data, semantic_metric, max_steps, n_eval_episodes, Q, candidate_action_mappping, attribute_action_mapping, space_dict, y_col, sampled_cluster_partitions):

    episode_rewards = []
    strategies = []
    attempts = 0
    while len(strategies) < n_eval_episodes and attempts < 10 * n_eval_episodes:
        attempts += 1
        episode_steps = []
        processed_data = data.copy()
        state = 0
        step = 0
        done = False
        total_rewards_ep = 0
        visited = set()
        visited.add(state)
   
        for step in range(max_steps):
            # Take the action (index) that have the maximum reward
            action = np.argmax(Q[state][:])
            new_state, reward, done, processed_data, partition, _ = step_func(processed_data, semantic_metric, action, candidate_action_mappping, attribute_action_mapping, space_dict, y_col, False, sampled_cluster_partitions)
            total_rewards_ep += reward
            episode_steps.append(partition)
       
            if done or new_state in visited: break
            
            visited.add(new_state)
            state = new_state
        
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

def evaluate_strategies(strategies, attributes, y_col, raw_data, semantic_metric):
    partition_dicts = []
    for s in strategies:
        semantic_score = 0
        p_dict = {}
        for i, partition in enumerate(s):
            if semantic_metric == 'l2_norm':
                semantic_score += partition.l2_norm
            elif semantic_metric == 'KLDiv':
                semantic_score += partition.KLDiv
            elif semantic_metric == 'gpt_semantics':
                # Divide by 4 to normalize the grade
                semantic_score += partition.gpt_semantics
            p_dict[attributes[i]] = partition.bins
        #print(p_dict)
        partition_dicts.append({"Partition": p_dict, "Semantic": semantic_score / 3})
    
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
    

if __name__ == '__main__':
    ppath = sys.path[0] + '/../../'
    dataset = 'pima'
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
    cluster_params = {'t': 0.7, 'criterion': 'distance'}
    cluster_assignments = get_cluster_assignments_multi_attr(list(space_dict.values()), cluster_params)
    candidate_action_mappping, attribute_action_mapping = get_candidate_action_mapping(list(space_dict.values()), cluster_assignments, attributes)
    #cluster_assignments = [[1,2],[1,2,3]]

    # Q learning 
    # Training parameters
    n_training_episodes = 2000
    learning_rate = 0.5 # to avoid overwriting good alternate paths too aggressively.

    # Evaluation parameters
    n_eval_episodes = 500

    # Environment parameters
    max_steps = 3
    gamma = 0.95 # high (e.g., 0.95–0.99) to value long-term rewards.

    # Exploration parameters
    max_epsilon = 1.0 
    min_epsilon = 0.3 # high to encourage exploration.
    decay_rate = 0.05 # low to encourage exploration.
    
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

    gd_list = []
    igd_list = []
    for round in range(10):
        # Initialize Q table
        state_space = sum([len(np.unique(assignments)) for assignments in cluster_assignments]) + 2
        #print("There are ", state_space, " possible states")
        action_space = sum([len(np.unique(assignments)) for assignments in cluster_assignments]) + 2
        #print("There are ", action_space, " possible actions")
        Qtable = initialize_q_table(state_space, action_space)

        # Train Q table
        Qtable = mask_Qtable(Qtable, cluster_assignments, debug=True)
        break
        Qtable, sampled_cluster_partitions = train(raw_data, semantic_metric, n_training_episodes, min_epsilon, max_epsilon, learning_rate, gamma, decay_rate, max_steps, Qtable, candidate_action_mappping, attribute_action_mapping, space_dict, y_col)
        print("Qtable after training: \n", Qtable)
        
        # Evaluate Q table
        mean_reward, std_reward, strategies = evaluate_agent(raw_data, semantic_metric, max_steps, n_eval_episodes, Qtable, candidate_action_mappping, attribute_action_mapping, space_dict, y_col, sampled_cluster_partitions)
        print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        #print("Strategies:", strategies)
        Qtable_df = pd.DataFrame(Qtable)
        Qtable_df.to_csv(os.path.join(ppath, 'scratch', f"{dataset}.{use_case}.{semantic_metric}.Qtable.csv"))

        # Evaluate strategies
        partition_df = evaluate_strategies(strategies, attributes, y_col, raw_data, semantic_metric)
        partition_df = partition_df[partition_df['Estimated'] == 1]
        #partition_df.to_csv(os.path.join(ppath, 'scratch', f"{dataset}.{use_case}.{semantic_metric}.strategies.csv"))
        est_points = [np.array(partition_df['Semantic'].values), np.array(partition_df['Utility'].values)]
        est_points = np.array(est_points).T
        est_points = np.unique(est_points, axis=0)
        print("Estimated points: ", est_points)
        # Compute distances
        gd_i = gd(est_points)
        igd_i = igd(est_points)
        gd_list.append(gd_i)
        igd_list.append(igd_i)
        print(f"GD: {gd_i:.2f}, IGD: {igd_i:.2f}")
    
    # Print the mean, median and std of the dist
    print("GD: ", np.mean(gd_list), np.median(gd_list), np.std(gd_list))
    print("IGD: ", np.mean(igd_list), np.median(igd_list), np.std(igd_list))
    # Save evaluation results
    #save_eval_results(eval_results, env_id)
    