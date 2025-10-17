"""
This script is to run exhaustive search on multiple attributes for data imputation.
"""
import sys
import os
sys.path.append("./")
#from scripts.framework.discretizers import *
from scripts.framework.TestSearchSpace import *
from scripts.framework.utils import *
from scripts.framework.framework_utils import *
import itertools
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    project_root = sys.path[0] + '/../'
    dataset = 'songs'
    imputer = 'KNN' # Choice of imputer: 'Simple', 'KNN', 'Iterative'
    frac = 0.001 # fraction of the data to be used for imputation
    # Read json file
    exp_config = json.load(open(os.path.join(project_root, 'data', dataset, f'{dataset}.json')))
    attributes = list(exp_config['attributes'].keys())
    
    # Load experiment data
    space_dict = {}
    for attr in attributes:
        data = pd.read_csv(os.path.join(project_root, 'data', dataset, 'scored_attributes', f'{attr}.csv'))
        # get the first 10 rows of the data
        #data = data.head(30)
        ss = TestSearchSpace(data, attr)
        space_dict[attr] = ss
    
    y_col = exp_config['target']
    raw_data = load_raw_data(project_root, exp_config)

    f_cols = [f"{attr}_ID" for attr in attributes] + [f"{attr}_bins" for attr in attributes] + ['utility', 'l2_norm', 'KLDiv', 'gpt_semantics']
    f_data = []
    # create combinations of ss.candidates
    all_candidates = list(itertools.product(*[space_dict[attr].candidates for attr in attributes]))
    all_candidates = [list(c) for c in all_candidates]
    print(f"Total number of candidates: {len(all_candidates)}")
    print(f"First 5 candidates: {all_candidates[:5]}")
    # print length of each candidates
    for k, v in space_dict.items():
        print(f"{k}: {len(v.candidates)}")
    
    ##### Uncomment the following line to limit the #####
    ##### number of candidates for testing purposes #####
    all_candidates = all_candidates[:4]
    #####################################################


    for i, strategy in enumerate(all_candidates):
        data = raw_data.copy()
        attrs_bins = {}
        IDs = []
        bins_ls = []
        l2_norm = 0
        KLDiv = 0
        gpt_semantics = 0

        for j, attr in enumerate(attributes):
            # Randomly sample 30% of the data and replace the age values with NaN
            data[attr + '.gt'] = data[attr]
            nans = raw_data.sample(frac=frac, random_state=42)
            data.loc[raw_data.index.isin(nans.index),attr] = np.nan
            attrs_bins[attr] = all_candidates[i][j].bins
            IDs.append(all_candidates[i][j].ID)
            bins_ls.append(all_candidates[i][j].bins)
            l2_norm += all_candidates[i][j].l2_norm
            KLDiv += all_candidates[i][j].KLDiv
            gpt_semantics += all_candidates[i][j].gpt_semantics
        start_time = time.time()
        utility = data_imputation_multi_attrs(data, y_col, attrs_bins, imputer=imputer)
        #utility = data_imputation_partition_dict(data, y_col, attrs_bins)
        f_data.append(IDs + bins_ls + [utility, l2_norm/len(attributes), KLDiv/len(attributes), gpt_semantics/len(attributes)])
        end_time = time.time()
        print(f"Time taken for candidate {i+1}: {end_time-start_time} seconds; Finished {i+1} out of {len(all_candidates)} candidates; Utility: {utility:.4f}, GPT Semantics: {gpt_semantics/len(attributes):.4f}.")
    
    f_data = pd.DataFrame(f_data, columns=f_cols)
    #f_data.to_csv(os.path.join(project_root, 'truth', f"{dataset}.multi_attrs.{imputer}Imputer.{frac}.csv"), index=False)
    print("Exhaustive search on multiple attributes is done.")