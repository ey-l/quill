import sys
import os

ppath = sys.path[0] + '/../../'
sys.path.append(os.path.join(ppath, 'code'))
sys.path.append(os.path.join(ppath, 'code', 'framework'))
from framework_utils import *

ID_COUNT = 0

if __name__ == '__main__':
    #np.random.seed(0)
    ppath = sys.path[0] + '/../../'
    f_runtime_cols = ['use_case', 'dataset', 'attr', 'method', 'semantic_metric', 'round', 'num_explored_points', 'partition_gen', 'semantic_comp', 'utility_comp', 'method_comp']
    f_quality_cols = ['use_case', 'dataset', 'attr', 'method', 'semantic_metric', 'round', 'avg_dist', 'min_num_bins', 'max_num_bins']

    # Load the diabetes dataset
    use_case = 'imputation'
    N_components = [3]
    rounds = 50
    gpt_measure = True
    #raw_data = pd.read_csv(os.path.join(ppath, 'data', 'uciml_pima-indians-diabetes-database', 'diabetes.csv'))
    raw_data = pd.read_csv(os.path.join(ppath, 'data', 'titanic', 'train.csv'))
    raw_data = raw_data[['Age', 'Fare', 'SibSp', 'Survived', 'Pclass', 'Parch', 'PassengerId']]
    dataset = 'titanic' #'pima'
    min_num_bins = 2
    max_num_bins = 20
    target = 'Survived'
    semantic_metrics = ['gpt_distance', 'l2_norm', 'KLDiv']
    #attributes = {'Age': [-1, 18, 35, 50, 65, 100], 'Glucose': [-1, 140, 200], 'BMI': [-1, 18.5, 25, 30, 68], }
    attributes = {'Age': [-1, 18, 35, 50, 65, 100], 'Fare': [-1, 10, 20, 30, 40, 50, 100, 600], 'SibSp': [-1, 1, 2, 3, 4, 5, 6, 7, 8]}

    # Make a new folder to save the results
    date_today = datetime.datetime.today().strftime('%Y_%m_%d')
    dst = os.path.join(ppath, 'exp', f"{dataset}.{date_today}.linkage_distributions.run2")
    dst_folder = os.path.join(dst, use_case)
    dst_fig_folder = os.path.join(dst, use_case, 'figs')
    if not os.path.exists(dst):
        os.mkdir(dst)
        os.mkdir(dst_folder)
        os.mkdir(dst_fig_folder)
    elif not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
        os.mkdir(dst_fig_folder)

    for attr, gold_standard_bins in attributes.items():
        f_runtime = []
        f_quality = []
        
        
        for j in range(1):
            if use_case == 'modeling':
                ss = get_explainable_modeling_search_space(raw_data, attr, target, gold_standard_bins, min_num_bins, max_num_bins, gpt_measure)
            elif use_case == 'imputation':
                ss = get_data_imputation_search_space(raw_data, attr, target, gold_standard_bins, min_num_bins, max_num_bins, gpt_measure)
            else:
                raise ValueError("Invalid use case")
            
            for semantic_metric in semantic_metrics:

                for i in range(rounds):
                    datapoints, gt_pareto_points, points_df = get_pareto_front(ss.candidates, semantic_metric)
                    f_runtime.append([use_case, dataset, attr, 'exhaustive', semantic_metric] + get_runtime_stats(ss, semantic_metric) + [0])

                    cluster_params = {'t': int(len(ss.candidates)/5), 'criterion': 'maxclust'}
                    sampling_params = {'num_samples': 1}
                    explored_points, est_pareto_points, runtime_stats, clusters = cluster_sampling(ss, linkage_distributions, random_sampling_clusters, semantic_metric, cluster_params, sampling_params)
                    distance = average_distance(gt_pareto_points, est_pareto_points, debug=True)
                    method_name = f'cs_linkage_rand'
                    f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, distance, min_num_bins, max_num_bins])
                    f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i] + runtime_stats)
                    points_df["Cluster"] = clusters
                    f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                    f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                    

                    for p in [0.2, 0.25, 0.3]:
                        cluster_params = {'t': int(len(ss.candidates)/5), 'criterion': 'maxclust'}
                        sampling_params = {'p': p}
                        explored_points, est_pareto_points, runtime_stats, clusters = cluster_sampling(ss, linkage_distributions, proportional_sampling_clusters, semantic_metric, cluster_params, sampling_params)
                        distance = average_distance(gt_pareto_points, est_pareto_points, debug=True)
                        method_name = f'cs_linkage_prop_{p}'
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, distance, min_num_bins, max_num_bins])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i] + runtime_stats)
                        points_df["Cluster"] = clusters
                        f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                        f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')

                        cluster_params = {'t': int(len(ss.candidates)/10), 'criterion': 'maxclust'}
                        sampling_params = {'p': p}
                        explored_points, est_pareto_points, runtime_stats, clusters = cluster_sampling(ss, linkage_distributions, reverse_propotional_sampling_clusters, semantic_metric, cluster_params, sampling_params)
                        distance = average_distance(gt_pareto_points, est_pareto_points, debug=True)
                        method_name = f'cs_linkage_reverse_{p}'
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, distance, min_num_bins, max_num_bins])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i] + runtime_stats)
                        points_df["Cluster"] = clusters
                        f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                        f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                        
                        cluster_params = {'eps': 0.03, 'min_samples': 3}
                        sampling_params = {'p': p}
                        explored_points, est_pareto_points, runtime_stats, clusters = cluster_sampling(ss, DBSCAN_distributions, reverse_propotional_sampling_clusters, semantic_metric, cluster_params, sampling_params)
                        distance = average_distance(gt_pareto_points, est_pareto_points, debug=True)
                        method_name = f'cs_dbscan_reverse_{p}'
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, distance, min_num_bins, max_num_bins])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i] + runtime_stats)
                        points_df["Cluster"] = clusters
                        f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                        f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                        
                        cluster_params = {'min_cluster_size': 3}
                        sampling_params = {'p': p}
                        explored_points, est_pareto_points, runtime_stats, clusters = cluster_sampling(ss, HDBSCAN_distributions, reverse_propotional_sampling_clusters, semantic_metric, cluster_params, sampling_params)
                        distance = average_distance(gt_pareto_points, est_pareto_points, debug=True)
                        method_name = f'cs_hdbscan_reverse_{p}'
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, distance, min_num_bins, max_num_bins])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i] + runtime_stats)
                        points_df["Cluster"] = clusters
                        f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                        f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                        
                    for frac in [0.2, 0.4, 0.8]:
                        method_name = f'random_sampling_{frac}'
                        explored_points, est_pareto_points, runtime_stats = random_sampling(ss, semantic_metric, frac=frac)
                        distance = average_distance(gt_pareto_points, est_pareto_points, debug=True)
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, distance, min_num_bins, max_num_bins])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i] + runtime_stats)


        f_runtime_df = pd.DataFrame(f_runtime, columns=f_runtime_cols)
        f_quality_df = pd.DataFrame(f_quality, columns=f_quality_cols)
        f_runtime_df.to_csv(os.path.join(dst_folder, f'{attr}_runtime.csv'), index=False)
        f_quality_df.to_csv(os.path.join(dst_folder, f'{attr}_quality.csv'), index=False)
        