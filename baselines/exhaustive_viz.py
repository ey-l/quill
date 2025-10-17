"""
This script is to run exhaustive search on multiple attributes for modeling.
"""
import sys
import os
sys.path.append("./")
from scripts.framework.discretizers import *
from scripts.framework.TestSearchSpace import *
from scripts.framework.utils import *
from scripts.framework.framework_utils import *
import warnings
warnings.filterwarnings('ignore')

def visualization_spearmanr(data, y_col, attrs_bins:Dict) -> float:
    """
    Wrapper function to visualize the data using ANOVA
    """
    #start_time = time.time()
    attr = list(attrs_bins.keys())[0]  # Assuming we are only interested in the first attribute for visualization
    partition = attrs_bins[attr]

    data = data[[attr, y_col]]
    data[attr] = pd.cut(data[attr], bins=partition, labels=partition[1:], include_lowest=True)
    data[attr] = data[attr].astype('float64')

    data = data[[attr, y_col]]
    data = data.groupby(attr).mean().reset_index()
    try: interestingness, _ = spearmanr(data[attr], data[y_col])
    except: interestingness = 0
    
    #data = data.groupby(attr)[y_col]
    #data = [group[1] for group in data]
    #try: f, p = f_oneway(*data)
    #except: f, p = 0, 1
    #interestingness = 1 - p  # Convert p-value to interestingness
    # if interestingness is nan, set it to 0
    if not isinstance(interestingness, (int, float)):
        interestingness = 0
    if pd.isna(interestingness):
        interestingness = 0
    return interestingness

if __name__ == '__main__':
    project_root = sys.path[0] + '/../'
    dataset = 'bank'
    # Read json file
    exp_config = json.load(open(os.path.join(project_root, 'data', dataset, f'{dataset}.json')))
    attributes = list(exp_config['attributes'].keys())
    
    # Load experiment data
    space_dict = {}
    for attr in attributes:
        data = pd.read_csv(os.path.join(project_root, 'data', dataset, 'scored_attributes', f'{attr}.csv'))
        # Filter data for multi_attribute
        #data = data[data['multi_attribute'] == 1]
        ss = TestSearchSpace(data, attr)
        space_dict[attr] = ss
    
    y_col = exp_config['target_viz']
    raw_data = load_raw_data(project_root, exp_config, use_case='visualization')

    for attr, space in space_dict.items():
        print(f"Processing attribute: {attr}")
        f_cols = [f"{attr}_ID", f"{attr}_bins", 'utility', 'raw_utility', 'l2_norm', 'KLDiv', 'gpt_semantics']
        f_data = []
        for i, candidate in enumerate(space.candidates):
            data = raw_data.copy()
            start_time = time.time()
            utility = visualization_spearmanr(data, y_col, {attr: candidate.bins})
            #utility = data_imputation_partition_dict(data, y_col, attrs_bins)
            end_time = time.time()
            print(f"Time taken for candidate {i+1}: {end_time - start_time} seconds; Utility: {utility}.")
            f_data.append([candidate.ID, candidate.bins, abs(utility), utility, candidate.l2_norm, candidate.KLDiv, candidate.gpt_semantics])
            #break
        
        f_data = pd.DataFrame(f_data, columns=f_cols)
        f_data.to_csv(os.path.join(project_root, 'truth', 'single_attrs', f"{dataset}.{attr}.spearman.csv"), index=False)
        #break
    print("Exhaustive search on single attributes is done.")