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
SEMANTICS = ['l2_norm', 'KLDiv', 'gpt_distance']
f_quality_cols = ['use_case', 'dataset', 'attr', 'method', 'semantic_metric', 'round', 'avg_dist', 'gd', 'igd','hd']
f_runtime_cols = ['use_case', 'dataset', 'attr', 'method', 'semantic_metric', 'round', 'num_explored_points']

if __name__ == '__main__':
    dataset = 'titanic'
    use_case = 'visualization'
    rounds = 20
    ppath = sys.path[0] + '/../../'
    
    # read json file
    exp_config = json.load(open(os.path.join(ppath, 'code', 'configs', f'{dataset}.json')))
    attributes = exp_config['attributes'].keys()
    # Make a new folder to save the results
    date_today = datetime.datetime.today().strftime('%Y_%m_%d')
    dst = os.path.join(ppath, 'exp', f"{dataset}.{date_today}.test1")
    dst_folder = os.path.join(dst, use_case)
    dst_fig_folder = os.path.join(dst, use_case, 'figs')
    if not os.path.exists(dst):
        os.mkdir(dst)
        os.mkdir(dst_folder)
        os.mkdir(dst_fig_folder)
    elif not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
        os.mkdir(dst_fig_folder)


    for attr in attributes:
        f_quality = []
        f_runtime = []
        # load experiment data
        data = pd.read_csv(os.path.join(ppath, 'experiment_data', dataset, use_case, f'{attr}.csv'))
        ss = TestSearchSpace(data)
        
        for semantic_metric in SEMANTICS:

            for i in range(rounds):
                datapoints, ground_truth, points_df = get_pareto_front(ss.candidates, semantic_metric)

                

                for p in [0.1, 0.2, 0.3]:
                        cluster_params = {'t': 0.7, 'criterion': 'distance'} #{'t': int(len(ss.candidates)/5), 'criterion': 'maxclust'}
                        sampling_params = {'p': p}
                        method_name = f'cs_linkage_rand_{p}'
                        explored_points, estimated, _, clusters = cluster_sampling(ss, linkage_distributions, random_sampling_clusters_robust, semantic_metric, cluster_params, sampling_params, False)
                        if explored_points is not None:
                            avg_dist = average_distance(ground_truth, estimated, debug=True)
                            gd = generational_distance(ground_truth, estimated)
                            igd = inverted_generational_distance(ground_truth, estimated)
                            hd = hausdorff_distance(ground_truth, estimated)
                            f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, avg_dist, gd, igd, hd])
                            f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                            if i < 5:
                                points_df["Cluster"] = clusters
                                f, ax = plot_pareto_points(ground_truth, estimated, explored_points, points_df, method_name)
                                f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                        
                        cluster_params = {'t': 0.5, 'criterion': 'distance'} #{'t': int(len(ss.candidates)/5), 'criterion': 'maxclust'}
                        sampling_params = {'p': p}
                        method_name = f'ucb_{p}'
                        alpha = 2
                        explored_points, estimated, _, clusters = UCB_estimate(alpha, ss, linkage_distributions, semantic_metric, cluster_params, sampling_params, False)
                        if explored_points is not None:
                            avg_dist = average_distance(ground_truth, estimated, debug=True)
                            gd = generational_distance(ground_truth, estimated)
                            igd = inverted_generational_distance(ground_truth, estimated)
                            hd = hausdorff_distance(ground_truth, estimated)
                            f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, avg_dist, gd, igd, hd])
                            f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                            if i < 5:
                                points_df["Cluster"] = clusters
                                f, ax = plot_pareto_points(ground_truth, estimated, explored_points, points_df, method_name)
                                f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')

                        cluster_params = {'t': 0.5, 'criterion': 'distance'}
                        sampling_params = {'p': p}
                        method_name = f'cs_linkage_rand_inverse_{p}'
                        explored_points, estimated, runtime_stats, clusters = cluster_sampling(ss, linkage_distributions, random_with_inverse_sampling_clusters, semantic_metric, cluster_params, sampling_params, False)
                        if explored_points is not None:
                            avg_dist = average_distance(ground_truth, estimated, debug=True)
                            gd = generational_distance(ground_truth, estimated)
                            igd = inverted_generational_distance(ground_truth, estimated)
                            hd = hausdorff_distance(ground_truth, estimated)
                            f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, avg_dist, gd, igd, hd])
                            f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                            if i < 5:
                                points_df["Cluster"] = clusters
                                f, ax = plot_pareto_points(ground_truth, estimated, explored_points, points_df, method_name)
                                f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')

                        cluster_params = {'t': 0.5, 'criterion': 'distance'}
                        sampling_params = {'p': p}
                        method_name = f'cs_linkage_reverse_{p}'
                        explored_points, estimated, runtime_stats, clusters = cluster_sampling(ss, linkage_distributions, reverse_propotional_sampling_clusters, semantic_metric, cluster_params, sampling_params, False)
                        avg_dist = average_distance(ground_truth, estimated, debug=True)
                        gd = generational_distance(ground_truth, estimated)
                        igd = inverted_generational_distance(ground_truth, estimated)
                        hd = hausdorff_distance(ground_truth, estimated)
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, avg_dist, gd, igd, hd])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                        if i < 5:
                            points_df["Cluster"] = clusters
                            f, ax = plot_pareto_points(ground_truth, estimated, explored_points, points_df, method_name)
                            f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                        
                        cluster_params = {'eps': 0.02, 'min_samples': 2}
                        sampling_params = {'p': p}
                        method_name = f'cs_dbscan_reverse_{p}'
                        explored_points, estimated, runtime_stats, clusters = cluster_sampling(ss, DBSCAN_distributions, reverse_propotional_sampling_clusters, semantic_metric, cluster_params, sampling_params, False)
                        avg_dist = average_distance(ground_truth, estimated, debug=True)
                        gd = generational_distance(ground_truth, estimated)
                        igd = inverted_generational_distance(ground_truth, estimated)
                        hd = hausdorff_distance(ground_truth, estimated)
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, avg_dist, gd, igd, hd])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                        if i < 5:
                            points_df["Cluster"] = clusters
                            f, ax = plot_pareto_points(ground_truth, estimated, explored_points, points_df, method_name)
                            f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                        
                        cluster_params = {'min_cluster_size': 3}
                        sampling_params = {'p': p}
                        method_name = f'cs_hdbscan_reverse_{p}'
                        explored_points, estimated, runtime_stats, clusters = cluster_sampling(ss, HDBSCAN_distributions, reverse_propotional_sampling_clusters, semantic_metric, cluster_params, sampling_params, False)
                        avg_dist = average_distance(ground_truth, estimated, debug=True)
                        gd = generational_distance(ground_truth, estimated)
                        igd = inverted_generational_distance(ground_truth, estimated)
                        hd = hausdorff_distance(ground_truth, estimated)
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, avg_dist, gd, igd, hd])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                        if i < 5:
                            points_df["Cluster"] = clusters
                            f, ax = plot_pareto_points(ground_truth, estimated, explored_points, points_df, method_name)
                            f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')

                for frac in [0.1, 0.2, 0.3, 0.5, 0.8]:   
                    method_name = f'random_sampling_{frac}'
                    explored_points, estimated, _ = random_sampling(ss, semantic_metric, frac=frac, if_runtime_stats=False)
                    avg_dist = average_distance(ground_truth, estimated, debug=True)
                    gd = generational_distance(ground_truth, estimated)
                    igd = inverted_generational_distance(ground_truth, estimated)
                    hd = hausdorff_distance(ground_truth, estimated)
                    f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, avg_dist, gd, igd, hd])
                    f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                    if i < 5:
                        f, ax = plot_pareto_points(ground_truth, estimated, explored_points, None, method_name)
                        f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')


        f_quality_df = pd.DataFrame(f_quality, columns=f_quality_cols)
        f_quality_df.to_csv(os.path.join(dst_folder, f'{attr}_quality.csv'), index=False)
        f_runtime_df = pd.DataFrame(f_runtime, columns=f_runtime_cols)
        f_runtime_df.to_csv(os.path.join(dst_folder, f'{attr}_runtime.csv'), index=False)
        #break
