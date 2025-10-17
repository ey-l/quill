"""
This script is to run exhaustive search on multiple attributes for modeling.
"""
import sys
import os
sys.path.append("./")
from scripts.framework.discretizers import *
from scripts.framework.SearchSpace import *
from scripts.framework.utils import *
from scripts.framework.framework_utils import *
import warnings
warnings.filterwarnings('ignore')

def explainable_modeling_multi_attrs(data, y_col, attrs_bins:Dict) -> float:
    """
    Wrapper function to model the data using an explainable model
    :param data: DataFrame
    :param attrs_bins: Dict
    :return accuracy: float
    """
    for attr, bins in attrs_bins.items():
        data[attr + '.binned'] = pd.cut(data[attr], bins, labels=False)
        data[attr + '.binned'] = data[attr + '.binned'].astype('float64')
        data = data.dropna(subset=[attr + '.binned'])
    
    #print(f"Data shape after binning: {data.shape}")

    # Evaluate the explainable model
    X_cols = [col for col in data.columns if col != y_col and col not in list(attrs_bins.keys())]
    #print(f"X_cols: {X_cols}")
    X = data[X_cols]
    y = data[y_col]
    #print(f"Data shape before train: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a decision tree model
    model = DecisionTreeClassifier(random_state=0, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #print(f"Accuracy: {accuracy}")
    return accuracy

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
    all_candidates = all_candidates[:3]
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
            attrs_bins[attr] = all_candidates[i][j].bins
            IDs.append(all_candidates[i][j].ID)
            bins_ls.append(all_candidates[i][j].bins)
            l2_norm += all_candidates[i][j].l2_norm
            KLDiv += all_candidates[i][j].KLDiv
            gpt_semantics += all_candidates[i][j].gpt_semantics
        start_time = time.time()
        utility = explainable_modeling_multi_attrs(data, y_col, attrs_bins)
        #utility = data_imputation_partition_dict(data, y_col, attrs_bins)
        end_time = time.time()
        print(f"Time taken for candidate {i+1}: {end_time - start_time} seconds; Utility: {utility}; Finished {i+1} out of {len(all_candidates)} candidates.")
        f_data.append(IDs + bins_ls + [utility, l2_norm/len(attributes), KLDiv/len(attributes), gpt_semantics/len(attributes)])
        #break
    
    f_data = pd.DataFrame(f_data, columns=f_cols)
    #f_data.to_csv(os.path.join(project_root, 'truth', f"{dataset}.multi_attrs.DecisionTreeClassifier.csv"), index=False)
    print("Exhaustive search on multiple attributes is done.")