import sys
import os

ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))
sys.path.append(os.path.join(ppath, 'code', 'framework'))

from import_packages import *
from utils import *


def cluster_strategies(df_out, semantic_col:str='gpt', utility_col:str='impute_accuracy', debug=False) -> List:
    """
    Cluster the data and estimate the Pareto front.
    Args:
        df_out (DataFrame): DataFrame containing the data
        semantic_col (str): Column name for semantic similarity
        utility_col (str): Column name for utility
    Returns:
        List: Estimated Pareto front
    """
    df = df_out.copy()
    # Cluster the data
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    pca = PCA(n_components=20)
    X = list(df['partitioned'].values)
    processed = pca.fit_transform(X)
    hdbscan_clusters = clusterer.fit_predict(processed)
    # Add the cluster labels to the dataframe
    df['cluster'] = hdbscan_clusters

    # Separate the data into outliers and non-outliers
    df_no_outliers = df[df['cluster'] != -1]
    df_outliers = df[df['cluster'] == -1]

    # For each cluster in the non-outliers, rank the points based on semantic_col
    # and get the top 1 point for each cluster
    top_points = []
    for i in range(df_no_outliers['cluster'].nunique()):
        cluster = df_no_outliers[df_no_outliers['cluster'] == i]
        cluster = cluster.sort_values(by=semantic_col, ascending=False)
        top_points.append(cluster.iloc[0])
    
    # Combine the top points with the outliers, and reset index
    df_top_points = pd.DataFrame(top_points)
    df_combined = pd.concat([df_top_points, df_outliers])
    df_combined = df_combined.reset_index(drop=True)
    print(len(df_combined))

    # Compute pareto front for the combined data
    est_pareto_points = get_pareto_points(df_combined, semantic_col=semantic_col, utility_col=utility_col)

    if debug: print("Estimated Pareto front:", est_pareto_points)
    return est_pareto_points

def sample_strategies(df_out, ratio:float=0.3, semantic_col:str='gpt', utility_col:str='impute_accuracy', debug=False) -> List:
    """
    Sample the data and estimate the Pareto front.
    Args:
        df_out (DataFrame): DataFrame containing the data
        ratio (float): Sampling ratio
        semantic_col (str): Column name for semantic similarity
        utility_col (str): Column name for utility
    Returns:
        List: Estimated Pareto front
    """
    df = df_out.copy()
    # Sample the data
    df_sample = df.sample(frac=ratio, random_state=42)
    # reset index
    df_sample = df_sample.reset_index(drop=True)
    # Compute Pareto front for the sampled data
    est_pareto_points = get_pareto_points(df_sample, semantic_col=semantic_col, utility_col=utility_col)
    
    if debug: print("Estimated Pareto front:", est_pareto_points)
    return est_pareto_points

def rank_strategies(df_out, ratio:float=0.3, semantic_col:str='gpt', utility_col:str='impute_accuracy', debug=False) -> List:
    df = df_out.copy()
    # Rank the data based on semantic_col
    df = df.sort_values(by=semantic_col, ascending=False)
    # Sample the top ratio% of the data
    df_sample = df.head(int(ratio*len(df)))
    # reset index
    df_sample = df_sample.reset_index(drop=True)
    # Compute Pareto front for the sampled data
    est_pareto_points = get_pareto_points(df_sample, semantic_col=semantic_col, utility_col=utility_col)

    if debug: print("Estimated Pareto front:", est_pareto_points)
    return est_pareto_points

if __name__ == '__main__':
    # Load the diabetes dataset
    df = pd.read_csv(os.path.join(ppath, 'data', 'uciml_pima-indians-diabetes-database', 'diabetes.csv'))

    # Load the age partitions
    col = 'Glucose'
    f = open(os.path.join(ppath, f'scratch/{col.lower()}_partitions.json'), "r")
    data = json.load(f)
    f.close()
    binnings = list(eval(data))
    outs = []
    age = list(df[col])
    N = len(age)

    semantic_col = 'gpt'
    utility_col = 'impute_accuracy'

    # Randomly sample 30% of the data and replace the age values with NaN
    data = df.copy()
    data[col + '.gt'] = data[col]
    df_nan = df.sample(frac=0.3, random_state=42)
    data.loc[df.index.isin(df_nan.index),col] = np.nan

    # Impute the missing values using KNN
    for i in binnings:
        #bins = i
        #semantic = 0
        bins = i['bins']
        semantic = i[semantic_col]
        data_i = data.copy()
        data_i[col + '.binned'] = pd.cut(data_i[col], bins=bins, labels=bins[1:])
        #data_i[col + '.binned'] = data_i[col + '.binned'].astype('float64')

        imputer = KNNImputer(n_neighbors=len(bins)-1)
        data_imputed = imputer.fit_transform(data_i[col + '.binned'].values.reshape(-1, 1))
        data_imputed = np.round(data_imputed)
        data_i[col+'.imputed'] = data_imputed
        data_i[col + '.final'] = pd.cut(data_i[col+'.imputed'], bins=bins, labels=bins[1:])
        data_i[col + '.final'] = data_i[col + '.final'].astype('float64')

        if len(data_i[data_i[col + '.final'].isnull()]) > 200:
            print(f"Skipping {bins}")
            continue
        #data_i[col + '.final'] = data_i[col + '.final'].fillna(-1)
        value_final = np.array(data_i[col + '.final'].values)
        value_final[np.isnan(value_final)] = -1
        value_final = np.round(value_final)

        # Evaluate data imputation
        data_i[col + '.gt'] = pd.cut(data_i[col + '.gt'], bins=bins, labels=bins[1:])
        data_i[col + '.gt'] = data_i[col + '.gt'].astype('float64')
        value_gt = np.array(data_i[col + '.gt'].values)
        value_gt[np.isnan(value_gt)] = -1
        value_gt = np.round(value_gt)
        #data_i[col + '.gt'] = data_i[col + '.gt'].fillna(-1)
        #data_i = data_i.dropna(subset=[col + '.final', col + '.gt'])
        impute_accuracy = accuracy_score(value_gt, value_final)

        #print(f"{bins}:", 1-i[semantic_col], impute_accuracy)

        hist, bin_edges = np.histogram(age, bins=bins)
        distribution = hist / N

        outs.append({'bins': bins, 'distribution': distribution, 'partitioned': np.array(data_i[col + '.final']), semantic_col: 1-semantic, utility_col: impute_accuracy})

    print("Number of strategies:", len(outs))
    # Compute ground truth Pareto front
    # dictionary to dataframe
    df_out = pd.DataFrame(outs)
    datapoints = [np.array(df_out[semantic_col]), np.array(df_out[utility_col])]
    lst = compute_pareto_front(datapoints)
    # label the Pareto optimal points in the dataframe as 1; otherwise 0
    df_out['pareto'] = 0
    df_out.loc[lst, 'pareto'] = 1
    # get the ground truth Pareto optimal points
    pareto_points = df_out[df_out['pareto'] == 1][[semantic_col, utility_col]]
    pareto_points = pareto_points.values.tolist()
    print("Ground Truth Pareto front:", pareto_points)

    # Estimate the Pareto points using clustering
    print("====== Clustering strategy ======")
    start_time = time.time()
    est_pareto_points = cluster_strategies(df_out, semantic_col=semantic_col, utility_col=utility_col, debug=True)
    # Average distance
    average_distance = eval_pareto_points(pareto_points, est_pareto_points, debug=True)
    print(f"Time to cluster and estimate Pareto front: {time.time() - start_time}")
    f, ax = plot_pareto_points(pareto_points, est_pareto_points, datapoints)
    f.savefig(os.path.join(ppath, 'figs', f'{col}.pareto_curve.clustering.png'))
                                            
    # Estimate the Pareto front using sampling 
    start_time = time.time()
    print("====== Sampling strategy ======")
    est_pareto_points = sample_strategies(df_out, ratio=0.2, semantic_col=semantic_col, utility_col=utility_col, debug=True)
    average_distance = eval_pareto_points(pareto_points, est_pareto_points, debug=True)
    print(f"Time to sample and estimate Pareto front: {time.time() - start_time}")
    f, ax = plot_pareto_points(pareto_points, est_pareto_points, datapoints)
    f.savefig(os.path.join(ppath, 'figs', f'{col}.pareto_curve.sampling.png'))

    # Estimate the Pareto front using semantic similarity ranking
    start_time = time.time()
    print("====== Semantic similarity ranking strategy ======")
    est_pareto_points = rank_strategies(df_out, ratio=0.2, semantic_col=semantic_col, utility_col=utility_col, debug=True)
    average_distance = eval_pareto_points(pareto_points, est_pareto_points, debug=True)
    print(f"Time to rank and estimate Pareto front: {time.time() - start_time}")
    f, ax = plot_pareto_points(pareto_points, est_pareto_points, datapoints)
    f.savefig(os.path.join(ppath, 'figs', f'{col}.pareto_curve.ranking.png'))