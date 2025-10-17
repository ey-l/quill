"""
This script is to find the best configuration for the clustering methods.
"""
import sys
import os

ppath = sys.path[0] + '/../../'
sys.path.append(os.path.join(ppath, 'code'))
sys.path.append(os.path.join(ppath, 'code', 'framework'))

from import_packages import *
from discretizers import *
from SearchSpace import *
from utils import *
from UCB import *
from framework_utils import *
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
SEMANTICS = ['l2_norm', 'KLDiv', 'gpt_distance']
f_quality_cols = ['use_case', 'dataset', 'attr', 'method', 'semantic_metric', 'round', 'avg_dist', 'gd', 'igd','hd']
f_runtime_cols = ['use_case', 'dataset', 'attr', 'method', 'semantic_metric', 'round', 'num_explored_points']
f_cluster_stats_cols = ['semantic_metric', 'method', 'davies_bouldin', 'silhouette', 'calinski_harabasz', 'num_clusters']

if __name__ == '__main__':
    ppath = sys.path[0] + '/../../'
    dataset = 'pima'
    use_case = 'visualization'
    rounds = 2 # since we have randomness in our method, we conduct multiple rounds
    p=0.2 # sampling rate for our exploration budget: the proportion of the search space to be explored

    # read json file
    exp_config = json.load(open(os.path.join(ppath, 'code', 'configs', f'{dataset}.json')))
    attributes = exp_config['attributes'].keys()
    # Make a new folder to save the results
    date_today = datetime.datetime.today().strftime('%Y_%m_%d')
    dst = os.path.join(ppath, 'exp', f"{dataset}.{date_today}.test_pipeline_example")
    dst_folder = os.path.join(dst, use_case)
    dst_fig_folder = os.path.join(dst, use_case, 'figs')
    dst_cluster_folder = os.path.join(dst, use_case, 'figs', 'clusters')
    if not os.path.exists(dst):
        os.mkdir(dst)
        os.mkdir(dst_folder)
        os.mkdir(dst_fig_folder)
        os.mkdir(dst_cluster_folder)
    elif not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
        os.mkdir(dst_fig_folder)
        os.mkdir(dst_cluster_folder)


    for attr in attributes:
        f_quality = []
        f_runtime = []
        #f_cluster_stats = []
        # load experiment data
        data = pd.read_csv(os.path.join(ppath, 'experiment_data', dataset, use_case, f'{attr}.csv'))
        ss = TestSearchSpace(data)
        
        for semantic_metric in SEMANTICS:

            for i in range(rounds):
                datapoints, ground_truth, points_df = get_pareto_front(ss.candidates, semantic_metric)
                t = 0.7

                # Cluster sampling configuration 1
                # Cluster technique -- hierarchical clustering
                # Sampling technique -- random sampling
                
                cluster_params = {'t': t, 'criterion': 'distance'} #{'t': int(len(ss.candidates)/5), 'criterion': 'maxclust'}
                sampling_params = {'p': p}
                method_name = f'cs_linkage_rand_{t}'
                explored_points, estimated, _, clusters = cluster_sampling(ss, linkage_distributions, random_sampling_clusters_robust, semantic_metric, cluster_params, sampling_params, False)
                
                ########## LOGGING & EVALUATING RESULT ##########
                points_df["Cluster"] = clusters
                if explored_points is not None:
                    avg_dist = average_distance(ground_truth, estimated, debug=True)
                    gd = generational_distance(ground_truth, estimated)
                    igd = inverted_generational_distance(ground_truth, estimated)
                    hd = hausdorff_distance(ground_truth, estimated)
                    f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, avg_dist, gd, igd, hd])
                    f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                    if i < 5:
                        f, ax = plot_pareto_points(ground_truth, estimated, explored_points, points_df, method_name)
                        f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                

                # Cluster sampling configuration 2
                # Cluster technique -- hierarchical clustering
                # Sampling technique -- UCB
                t = 0.7
                alpha = 2
                cluster_params = {'t': t, 'criterion': 'distance'} #{'t': int(len(ss.candidates)/5), 'criterion': 'maxclust'}
                sampling_params = {'p': p}
                method_name = f'ucb_linkage_{t}_{alpha}'
                explored_points, estimated, _, clusters, _ = UCB_estimate(alpha, ss, linkage_distributions, semantic_metric, cluster_params, sampling_params, False)
                ########## LOGGING & EVALUATING RESULT ##########
                points_df["Cluster"] = clusters
                if explored_points is not None:
                    avg_dist = average_distance(ground_truth, estimated, debug=True)
                    gd = generational_distance(ground_truth, estimated)
                    igd = inverted_generational_distance(ground_truth, estimated)
                    hd = hausdorff_distance(ground_truth, estimated)
                    f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, avg_dist, gd, igd, hd])
                    f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                    if i < 5:
                        f, ax = plot_pareto_points(ground_truth, estimated, explored_points, points_df, method_name)
                        f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                
                
        f_quality_df = pd.DataFrame(f_quality, columns=f_quality_cols)
        f_quality_df.to_csv(os.path.join(dst_folder, f'{attr}_quality.csv'), index=False)
        f_runtime_df = pd.DataFrame(f_runtime, columns=f_runtime_cols)
        f_runtime_df.to_csv(os.path.join(dst_folder, f'{attr}_runtime.csv'), index=False)
        #f_cluster_stats_df = pd.DataFrame(f_cluster_stats, columns=f_cluster_stats_cols)
        #f_cluster_stats_df.to_csv(os.path.join(dst_folder, f'{attr}_cluster_stats.csv'), index=False)