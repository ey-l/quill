import sys
import os
import numpy as np
import pandas as pd
from typing import List, Union, Any, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import cKDTree

MODEL_ID = "gpt-3.5-turbo"

def load_raw_data(project_root, exp_config, use_case=None, missing_data_fraction=0.3) -> pd.DataFrame:
    """
    Load the data for a given dataset, use case, and attribute
    :param project_root: str
    :param exp_config: dict
    :return: pd.DataFrame
    """
    raw_data = pd.read_csv(os.path.join(project_root, exp_config['data_path']))
    if use_case == "visualization":
        raw_data = raw_data[exp_config['features'] + [exp_config['target_viz']]]
        raw_data = raw_data.dropna(subset=exp_config['features'] + [exp_config['target_viz']])
    else: 
        raw_data = raw_data[exp_config['features'] + [exp_config['target']]]
        raw_data = raw_data.dropna(subset=exp_config['features'] + [exp_config['target']])
    dataset = exp_config['dataset']
    attributes = list(exp_config['attributes'].keys())
    
    #if dataset == 'pima':
    #    raw_data = raw_data[(raw_data['Age'] > 19) & (raw_data['Age'] < 61)]
    
    if use_case == 'imputation':
        for attr in attributes:
            data = raw_data.copy()
            data[attr + '.gt'] = data[attr]
            nans = raw_data.sample(frac=missing_data_fraction, random_state=42)
            data.loc[raw_data.index.isin(nans.index),attr] = np.nan
            raw_data = data.copy()

    return raw_data

def pairwise_distance(X, metric):
    distances = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            distances[i, j] = metric(X[i], X[j])
    return distances

def zero_pad_vectors(v1, v2):
    # Identify the length of the longer vector
    max_len = max(len(v1), len(v2))

    # Pad each vector with zeros at the end to match the length of the longer vector
    v1_padded = np.pad(v1, (0, max_len - len(v1)), 'constant')
    v2_padded = np.pad(v2, (0, max_len - len(v2)), 'constant')

    return v1_padded, v2_padded
    
def create_intervals(data):
    return [(data[i], data[i + 1]) for i in range(0, len(data)-1)]

def get_semantic_grade(attr:str, bins) -> float:
    """
    Wrapper function to calculate the semantic grade of the bins
    :param attr: str
    :param bins: List
    :return: float
    """
    from grader.gpt_grader import evaluate_groupings_gpt
    input = [(attr, create_intervals(bins), (bins[0], bins[-1]))]
    results = evaluate_groupings_gpt(input)
    print("Input: ", input, results[0][0])
    # Print results for each test case in the group
    for (feature, grouping, feature_range), (grade, explanation, reference_count, reference_links) in zip(input, results):
        return float(grade) / 4 # Normalize to [0, 1]
    
def average_distance(ground_truth:List, estimated:List, debug=False) -> float:
    """
    Evaluate the Pareto front using nearest neighbor search.
    Args:
        ground_truth (List): Ground truth Pareto front
        estimated (List): Estimated Pareto front
    Returns:
        float: Average distance between the estimated and ground truth Pareto fronts
    """
    estimated = np.array(estimated)
    ground_truth = np.array(ground_truth)
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(estimated)
    # Find nearest neighbor in estimated curve for each point in ground truth curve
    distances, _ = tree.query(ground_truth)
    # Average distance
    average_distance = np.mean(distances)
    if debug: print(f"Average Distance: {average_distance}")
    return average_distance

def euclidean_distance(point_a, point_b):
    """Compute the Euclidean distance between two points."""
    return np.linalg.norm(point_a - point_b)

def generational_distance(ground_truth, estimated):
    """
    Calculate Generational Distance (GD).
    Args:
        estimated (ndarray): Estimated Pareto front (N x d).
        ground_truth (ndarray): Ground truth Pareto front (M x d).
    Returns:
        float: GD score.
    """
    estimated = np.array(estimated)
    ground_truth = np.array(ground_truth)
    distances = [min(euclidean_distance(e, g) for g in ground_truth) for e in estimated]
    return np.sqrt(np.mean(np.square(distances)))

def inverted_generational_distance(ground_truth, estimated):
    """
    Calculate Inverted Generational Distance (IGD).
    Args:
        estimated (ndarray): Estimated Pareto front (N x d).
        ground_truth (ndarray): Ground truth Pareto front (M x d).
    Returns:
        float: IGD score.
    """
    estimated = np.array(estimated)
    ground_truth = np.array(ground_truth)
    distances = [min(euclidean_distance(g, e) for e in estimated) for g in ground_truth]
    return np.sqrt(np.mean(np.square(distances)))

def hausdorff_distance(ground_truth, estimated):
    """
    Calculate Hausdorff Distance (HD) as max(GD, IGD).
    Args:
        estimated (ndarray): Estimated Pareto front (N x d).
        ground_truth (ndarray): Ground truth Pareto front (M x d).
    Returns:
        float: HD score.
    """
    gd = generational_distance(ground_truth, estimated)
    igd = inverted_generational_distance(ground_truth, estimated)
    return max(gd, igd)

def plot_pareto_points(pareto_points:List, est_pareto_points:List, explored_points:List=None, points_df=None, title:str='') -> Tuple:
    """
    Plot the estimated and ground truth Pareto fronts.
    Args:
        pareto_points (List): Ground truth Pareto front
        est_pareto_points (List): Estimated Pareto front
    """
    # Sort the points for plotting
    pareto_points = sorted(pareto_points, key=lambda x: x[0])
    est_pareto_points = sorted(est_pareto_points, key=lambda x: x[0])
    # Plot the Pareto front
    pareto_points = np.array(pareto_points)
    est_pareto_points = np.array(est_pareto_points)
    #datapoints = np.array(datapoints)
    # Set size of the plot
    fig, ax = plt.subplots(figsize=(6, 4))
    #f, ax = plt.subplots()
    #ax.scatter(datapoints[0], datapoints[1], c='gray', label='Data Points', alpha=0.3)
    markers = ["." , "," , "o" , "v" , "^" , "<", ">"]
    if points_df is not None:
        # Plot clusters
        colors = cm.rainbow(np.linspace(0, 1, len(points_df['Cluster'].unique())))
        for cluster in points_df['Cluster'].unique():
            cluster_points = points_df[points_df['Cluster'] == cluster]
            marker_index = int(cluster % len(markers))
            ax.scatter(cluster_points['Semantic'], cluster_points['Utility'], label=cluster, color=colors[cluster], alpha=0.5, marker=markers[marker_index])

    ax.scatter(explored_points[0], explored_points[1], c='gray', label='Explored Points', marker='x',)
    ax.plot(pareto_points[:, 0], pareto_points[:, 1], '+-', c='red', label='Ground Truth')
    ax.plot(est_pareto_points[:, 0], est_pareto_points[:, 1], 'x-', c='green', label='Estimated')
    ax.legend(bbox_to_anchor=(1, 1),ncol=3)
    ax.set_xlabel('Semantic Distance', fontsize=14)
    ax.set_ylabel('Utility', fontsize=14)
    if title != '':
        ax.set_title(title, fontsize=14)
    else: ax.set_title('Pareto Curve Comparison', fontsize=14)
    return fig, ax

def plot_clusters(points_df=None, title:str='') -> Tuple:
    """
    Plot the estimated and ground truth Pareto fronts.
    Args:
        pareto_points (List): Ground truth Pareto front
        est_pareto_points (List): Estimated Pareto front
    """
    #datapoints = np.array(datapoints)
    # Set size of the plot
    fig, ax = plt.subplots(figsize=(6, 4))
    #f, ax = plt.subplots()
    #ax.scatter(datapoints[0], datapoints[1], c='gray', label='Data Points', alpha=0.3)
    markers = ["." , "," , "o" , "v" , "^" , "<", ">"]
    if points_df is not None:
        # Plot clusters
        colors = cm.rainbow(np.linspace(0, 1, len(points_df['Cluster'].unique())))
        for cluster in points_df['Cluster'].unique():
            cluster_points = points_df[points_df['Cluster'] == cluster]
            marker_index = int(cluster % len(markers))
            ax.scatter(cluster_points['Semantic'], cluster_points['Utility'], label=cluster, color=colors[cluster], alpha=0.5, marker=markers[marker_index])

    ax.legend(bbox_to_anchor=(1, 1),ncol=3)
    ax.set_xlabel('Semantic Distance', fontsize=14)
    ax.set_ylabel('Utility', fontsize=14)
    if title != '':
        ax.set_title(title, fontsize=14)
    else: ax.set_title('Pareto Curve Comparison', fontsize=14)
    return fig, ax

def compute_pareto_front(datapoints: List) -> List[int]:
    """
    Fast Pareto front computation for 2D points (maximize both objectives).
    Handles ties and returns indices of non-dominated points.
    
    Args:
        datapoints (List): 2D list/array of shape (2, N)
            Row 0: semantic similarity
            Row 1: utility

    Returns:
        List[int]: Indices of Pareto-optimal points
    """
    datapoints = np.array(datapoints)
    assert datapoints.shape[0] == 2, "Expected datapoints of shape (2, N)"
    
    points = datapoints.T  # shape (N, 2)
    indices = np.arange(len(points))

    # Sort by semantic similarity descending, breaking ties with utility descending
    sorted_order = np.lexsort((-points[:,1], -points[:,0]))
    sorted_points = points[sorted_order]
    sorted_indices = indices[sorted_order]

    pareto_indices = []
    max_utility = -np.inf

    for i in range(len(sorted_points)):
        if sorted_points[i][1] > max_utility:
            pareto_indices.append(sorted_indices[i])
            max_utility = sorted_points[i][1]

    return pareto_indices

def get_pareto_points(data, semantic_col:str='gpt', utility_col:str='impute_accuracy') -> List:
    """
    Get the Pareto front from the data.
    Args:
        data (DataFrame): DataFrame containing the data
        semantic_col (str): Column name for semantic similarity
        utility_col (str): Column name for utility
    Returns:
        List: Pareto front
    """
    # Compute Pareto front for the data
    datapoints = [np.array(data[semantic_col]), np.array(data[utility_col])]
    lst = compute_pareto_front(datapoints)
    # label the Pareto optimal points in the dataframe as 1; otherwise 0
    data['pareto'] = 0
    data.loc[lst, 'pareto'] = 1
    # get the Pareto optimal points
    pareto_points = data[data['pareto'] == 1][[semantic_col, utility_col]]
    pareto_points = pareto_points.values.tolist()
    return pareto_points