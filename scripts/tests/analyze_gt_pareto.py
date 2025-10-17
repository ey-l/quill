"""
This script analyzes the ground truth pareto front for the clustering methods.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.framework.discretizers import *
from scripts.framework.SearchSpace import *
from scripts.framework.utils import *
from scripts.framework.UCB import *
from scripts.framework.framework_utils import *
from baselines.exhaustive_modeling import explainable_modeling_multi_attrs
SEMANTICS = ['l2_norm', 'KLDiv', 'gpt_semantics']

if __name__ == '__main__':
    ppath = sys.path[0] + '/../../'
    dataset = 'pima'
    use_case = 'modeling'

    gt_df = pd.read_csv(os.path.join(ppath, 'exp', 'pima.multi_attrs_exhaustive_search.csv'))
    # Filter data by gpt_prompt > 0
    gt_df = gt_df[gt_df['gpt_semantics'] > 0]
    print(gt_df.head())
    print(gt_df.shape)

    # Find the pareto front
    semantic_measures = ['gpt_semantics'] #'l2_norm', 'KLDiv', 
    for semantic_metric in semantic_measures:
        datapoints = [np.array(gt_df[semantic_metric].values), np.array(gt_df['utility'].values)]
        lst = compute_pareto_front(datapoints)
        gt_df["Estimated"] = 0
        gt_df.loc[lst, "Estimated"] = 1
        gt_df["Explored"] = 1
        
        # Save the results
        gt_df = gt_df[gt_df['Estimated'] == 1]
        gt_df.to_csv(os.path.join(ppath, 'exp', f'pima.multi_attrs_exhaustive_search.gt.{semantic_metric}.csv'), index=False)
        print(gt_df.head())