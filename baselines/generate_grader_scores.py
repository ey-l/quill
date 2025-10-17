"""
This script uses GPT grader to generate semantic scores for attributes in the dataset.
"""
import sys
import os
import json
import time

sys.path.append("./")

from scripts.framework.discretizers import *
from scripts.framework.SearchSpace import *
from scripts.framework.utils import *
np.set_printoptions(threshold=sys.maxsize)

def visualization_one_attr(data, y_col, attr:str, partition:Partition) -> float:
    """
    Wrapper function to visualize the data using ANOVA
    """
    start_time = time.time()
    data = data[[attr, y_col]]
    data[attr] = pd.cut(data[attr], bins=partition.bins, labels=partition.bins[1:], include_lowest=True)
    data[attr] = data[attr].astype('float64')
    data = data.groupby(attr)[y_col]
    data = [group[1] for group in data]
    try: f, p = f_oneway(*data)
    except: f = 0
    partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
    #print(f'ANOVA: {f}')
    partition.utility = f
    return f

def visualization_one_attr_spearmanr(data, y_col, attr:str, partition:Partition) -> float:
    """
    Wrapper function to visualize the data using ANOVA
    """
    start_time = time.time()
    data = data[[attr, y_col]]
    data[attr] = pd.cut(data[attr], bins=partition.bins, labels=partition.bins[1:], include_lowest=True)
    data[attr] = data[attr].astype('float64')
    data = data[[attr, y_col]]
    data = data.groupby(attr).mean().reset_index()
    try: rho, _ = spearmanr(data[attr], data[y_col])
    except: rho = 0
    partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
    partition.utility = rho
    return rho

def explainable_modeling_one_attr(data, y_col, attr:str, partition:Partition) -> float:
    """
    Wrapper function to model the data using an explainable model
    :param data: DataFrame
    :param attr: str
    :param bins: List
    :return: float
    """
    start_time = time.time()
    bins = partition.bins
    data[attr + '.binned'] = partition.binned_values
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

    partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
    partition.utility = model_accuracy
    return model_accuracy

def explainable_modeling_multi_attrs(data, y_col, partition_dict: Dict[str, Partition]) -> float:
    """
    Wrapper function to model the data using an explainable model
    ***** Note: This function is only used in demo_data_modeling_case.ipynb for now *****
    ***** Need to be updated to support multiple attributes for the real workflow *****
    :param data: DataFrame
    :param y_col: str
    :param partition_dict: Dict[str, Partition]
    :return: float
    """
    start_time = time.time()
    
    for attr, partition in partition_dict.items():
        data[attr + '.binned'] = partition.binned_values
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

    #partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
    #partition.utility = model_accuracy
    return model_accuracy

def data_imputation_one_attr(data, y_col, attr:str, partition:Partition) -> float:
    """
    Wrapper function to impute missing values in a dataset
    :param data: DataFrame
    :param attr: str
    :param bins: List
    :return: float
    """
    start_time = time.time()
    bins = partition.bins
    # Bin attr column, with nan values
    data[attr + '.binned'] = pd.cut(data[attr], bins=bins, labels=bins[1:], include_lowest=True)
    
    # Impute the missing values using KNN
    X_cols = [col for col in data.columns if col != y_col and col != attr and col != attr + '.gt']
    X = data[X_cols]
    idx = X.columns.get_loc(attr + '.binned')
    imputer = KNNImputer(n_neighbors=len(bins)-1)
    X_imputed = imputer.fit_transform(X)
    
    # Bin imputed values
    data_imputed = np.round(X_imputed[:, idx])
    data[attr+'.imputed'] = data_imputed
    data[attr + '.final'] = pd.cut(data[attr+'.imputed'], bins=bins, labels=bins[1:], include_lowest=True)
    data[attr + '.final'] = data[attr + '.final'].astype('float64')
    value_final = np.array(data[attr + '.final'].values)
    value_final[np.isnan(value_final)] = -1 # Replace NaN with -1
    value_final = np.round(value_final)

    # Evaluate data imputation
    data[attr + '.gt'] = pd.cut(data[attr + '.gt'], bins=bins, labels=bins[1:], include_lowest=True)
    data[attr + '.gt'] = data[attr + '.gt'].astype('float64')
    value_gt = np.array(data[attr + '.gt'].values)
    value_gt[np.isnan(value_gt)] = -1 # Replace NaN with -1
    value_gt = np.round(value_gt)
    impute_accuracy = accuracy_score(value_gt, value_final)

    partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
    partition.utility = impute_accuracy
    return impute_accuracy

def get_data_imputation_search_space(raw_data, attr, target, gold_standard_bins, min_val, max_val, min_num_bins=2, max_num_bins=20, gpt_measure=True):
    # Randomly sample 30% of the data and replace the age values with NaN
    data = raw_data.copy()
    data[attr + '.gt'] = data[attr]
    nans = raw_data.sample(frac=0.3, random_state=42)
    data.loc[raw_data.index.isin(nans.index),attr] = np.nan

    # Define gold standard bins
    data_i = data.copy()
    data_i = data_i.dropna(subset=[attr, target])
    values = data_i[attr].values
    binned_values = pd.cut(values, bins=gold_standard_bins, labels=gold_standard_bins[1:], include_lowest=True)
    gold_standard = Partition(bins=gold_standard_bins, binned_values=binned_values, values=values, method='gold-standard', gold_standard=True, gpt_measure=gpt_measure)

    # Generate bins
    ss = PartitionSearchSpace(gpt_measure=gpt_measure, max_val=max_val, min_val=min_val)
    ss.prepare_candidates(data, attr, target, min_num_bins, max_num_bins, gold_standard)
    ss.standardize_semantics()

    # Exhausitve search for the pareto curve partitions
    for partition in ss.candidates:
        data_i = data.copy()
        data_imputation_one_attr(data_i, target, attr, partition)
    
    return ss

def get_explainable_modeling_search_space(raw_data, attr, target, gold_standard_bins, min_val, max_val, min_num_bins=2, max_num_bins=20, gpt_measure=True):
    data = raw_data.dropna(subset=[attr, target])
    data_i = data.copy()
    data_i = data_i.dropna(subset=[attr, target])
    values = data_i[attr].values
    binned_values = pd.cut(values, bins=gold_standard_bins, labels=gold_standard_bins[1:], include_lowest=True)
    gold_standard = Partition(bins=gold_standard_bins, binned_values=binned_values, values=values, method='gold-standard', gold_standard=True, gpt_measure=gpt_measure)

    # Generate bins
    ss = PartitionSearchSpace(gpt_measure=gpt_measure, max_val=max_val, min_val=min_val)
    ss.prepare_candidates(data, attr, target, min_num_bins, max_num_bins, gold_standard)
    ss.standardize_semantics()

    # Exhausitve search for the pareto curve partitions
    for partition in ss.candidates:
        data_i = data.copy()
        explainable_modeling_one_attr(data_i, target, attr, partition)
    
    return ss

def get_visualization_search_space(raw_data, attr, target, gold_standard_bins, min_val, max_val, min_num_bins=2, max_num_bins=20, gpt_measure=True):
    data = raw_data.dropna(subset=[attr, target])
    data_i = data.copy()
    data_i = data_i.dropna(subset=[attr, target])
    values = data_i[attr].values
    #binned_values = pd.cut(values, bins=gold_standard_bins, labels=gold_standard_bins[1:], include_lowest=True)
    gold_standard = Partition(bins=gold_standard_bins, values=values, method='gold-standard', gold_standard=True, gpt_measure=gpt_measure)

    # Generate bins
    ss = PartitionSearchSpace(gpt_measure=gpt_measure, max_val=max_val, min_val=min_val)
    ss.prepare_candidates(data, attr, target, min_num_bins, max_num_bins, gold_standard)
    ss.standardize_semantics()

    # Exhausitve search for the pareto curve partitions
    #for partition in ss.candidates:
    #    data_i = data.copy()
    #    visualization_one_attr_spearmanr(data_i, target, attr, partition)
    
    #ss.standardize_utility()
    
    return ss


if __name__ == '__main__':
    project_root = sys.path[0] + '/../'
    f_data_cols = ['ID', 'method', 'bins', 'distribution', 'kl_d', 'l2_norm', 'gpt_prompt']
    semantic_metrics = ['gpt_semantics', 'l2_norm', 'KLDiv']
    
    # Load the diabetes dataset
    use_case = 'visualization' #'imputation'
    gpt_measure = True
    dataset = 'lalonde' #'pima'

    # read json file
    exp_config = json.load(open(os.path.join(project_root, 'data', dataset, f'{dataset}.json')))
    target = exp_config['target']
    attributes = exp_config['attributes']
    raw_data = load_raw_data(project_root, exp_config)
    min_num_bins = exp_config['min_num_bins']
    max_num_bins = exp_config['max_num_bins']
    value_ranges = exp_config['value_ranges']

    dst_folder = os.path.join(project_root, 'data', dataset, 'scored_attributes')
    if not os.path.exists(dst_folder): 
        os.makedirs(dst_folder)
    else: print(f"Folder {dst_folder} already exists")
    
    for attr, gold_standard_bins in attributes.items():
        f_data = []
        min_val = value_ranges[attr][0]
        max_val = value_ranges[attr][1]
        if use_case == 'modeling':
            ss = get_explainable_modeling_search_space(raw_data, attr, target, gold_standard_bins, min_val, max_val, min_num_bins, max_num_bins, gpt_measure)
        elif use_case == 'imputation':
            ss = get_data_imputation_search_space(raw_data, attr, target, gold_standard_bins, min_val, max_val, min_num_bins, max_num_bins, gpt_measure)
        elif use_case == 'visualization':
            ss = get_visualization_search_space(raw_data, attr, target, gold_standard_bins, min_val, max_val, min_num_bins, max_num_bins, gpt_measure)
        else:
            raise ValueError("Invalid use case")
            
        
        for partition in ss.candidates:
            #binned_values = partition.binned_values.to_numpy()
            #print(a.tolist())
            #print(a)
            f_data.append([partition.ID, partition.method, np.array(partition.bins), partition.distribution, partition.KLDiv, partition.l2_norm, partition.gpt_semantics])
            #print(partition.binned_values.values)
        f_data_df = pd.DataFrame(f_data, columns=f_data_cols)
        f_data_df.to_csv(os.path.join(dst_folder, f'{attr}.csv'), index=False)
        #break
        