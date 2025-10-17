import sys
import os

import numpy as np
from joblib import Parallel, delayed
import json
from typing import List, Union, Any, Tuple, Dict, Iterable, Optional
import time
import re
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from scipy.stats import wasserstein_distance, spearmanr, f_oneway
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#import hdbscan
from scripts.framework.causal import data_causal_score, compute_causal_effect, warper_get_thresholds

#from scripts.framework.discretizers import *
# from scripts.framework.TestSearchSpace import *
#from scripts.framework.SearchSpace import *
from scripts.framework.utils import *

ID_COUNT = 0

def find_optimal_t(search_space, binned_values=False) -> float:
    X = None
    Z = None
    if binned_values:
        X = np.array([p.binned_values for p in search_space.candidates], dtype=object)
        pca = PCA(n_components=10, random_state=0)
        X = pca.fit_transform(X)
        Z = linkage(X, method='ward')
    else:
        X = np.array([p.distribution for p in search_space.candidates], dtype=object)
        X = pairwise_distance(X, metric=wasserstein_distance)
        # Convert to condensed form if needed
        condensed_X = squareform(X)  # X must be square and symmetric
        # Compute linkage
        Z = linkage(condensed_X, method='ward')

    # Cophenetic correlation and distances
    #coph_corr, coph_dists = cophenet(Z, condensed_X)
    t_values = np.linspace(0.36, 0.8, 20)

    results = []

    for t in t_values:
        labels = fcluster(Z, t=t, criterion='distance')
        n_clusters = len(set(labels))

        if n_clusters <= 1 or n_clusters >= len(X):
            continue

        try:
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)
            results.append((t, sil, ch, db))
        except Exception as e:
            print(f"Skipping t={t:.2f} due to error: {e}")

    # Convert to NumPy array for easier processing
    results = np.array(results)
    ts, sils, chs, dbs = results.T

    # Normalize: higher is better for silhouette and CH; lower is better for DB
    sils_norm = minmax_scale(sils)
    chs_norm = minmax_scale(chs)
    dbs_norm = minmax_scale(-dbs)  # invert so that lower DB → higher score

    # Weighted average of normalized metrics
    composite_score = 1 * sils_norm + 0 * chs_norm + 0 * dbs_norm
    best_idx = np.argmax(composite_score)

    # Display the best t
    print(f"t = {ts[best_idx]:.2f}")
    return ts[best_idx]

def explainable_modeling_using_strategy(data, y_col, strategy) -> float:
    """
    Wrapper function to model the data using an explainable model
    ***** Note: This function is only used in demo_data_modeling_case.ipynb for now *****
    :param data: DataFrame
    :param y_col: str
    :param partition_dict: Dict[str, Partition]
    :return: float
    """
    attributes = []
    for partition in strategy.partition_list:
        attr = partition.attribute
        attributes.append(attr)
        #print(f"Discretizing {attr}...")
        data[attr + '.binned'] = pd.cut(data[attr], bins=partition.bins, labels=False)
        data[attr + '.binned'] = data[attr + '.binned'].astype('float64')
        data = data.dropna(subset=[attr + '.binned'])
    
    #print(f"Data shape after binning: {data.shape}")
    
    # Evaluate the explainable model
    X_cols = [col for col in data.columns if col != y_col and col not in attributes]
    X = data[X_cols]
    y = data[y_col]
    #print(f"Data shape before train: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a decision tree model
    model = DecisionTreeClassifier(random_state=0, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)

    return model_accuracy

def visualization_one_attr_anova(data, y_col, attr:str, partition) -> float:
    """
    Wrapper function to visualize the data using ANOVA
    """
    start_time = time.time()
    data = data[[attr, y_col]]
    data[attr] = pd.cut(data[attr], bins=partition.bins, labels=partition.bins[1:])
    data[attr] = data[attr].astype('float64')
    data = data.groupby(attr)[y_col]
    data = [group[1] for group in data]
    f, p = f_oneway(*data)
    partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
    partition.utility = f
    return f

def visualization_one_attr_spearmanr(data, y_col, attr:str, partition) -> float:
    """
    Wrapper function to visualize the data using Spearman correlation
    """
    data = data[[attr, y_col]]
    data[attr] = pd.cut(data[attr], bins=partition.bins, labels=partition.bins[1:], include_lowest=True)
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
    if pd.isna(interestingness):
        interestingness = 0
    if not isinstance(interestingness, (int, float)):
        interestingness = 0
    return abs(interestingness)

def explainable_modeling_one_attr(data, y_col, attr:str, partition) -> float:
    """
    Wrapper function to model the data using an explainable model
    :param data: DataFrame
    :param attr: str
    :param bins: List
    :return: float
    """
    start_time = time.time()
    bins = partition.bins
    data[attr + '.binned'] = pd.cut(data[attr], bins=bins, labels=bins[1:])
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

def data_imputation_one_attr(data, y_col, attr:str, partition, imputer='KNN') -> float:
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
    data[attr + '.binned'] = pd.cut(data[attr], bins=bins, labels=bins[1:])
    
    # Impute the missing values using KNN
    X_cols = [col for col in data.columns if col != y_col and col != attr and col != attr + '.gt']
    X = data[X_cols]
    idx = X.columns.get_loc(attr + '.binned')
    imputer = KNNImputer(n_neighbors=len(bins)-1)
    X_imputed = imputer.fit_transform(X)
    
    # Bin imputed values
    data_imputed = np.round(X_imputed[:, idx])
    data[attr+'.imputed'] = data_imputed
    data[attr + '.final'] = pd.cut(data[attr+'.imputed'], bins=bins, labels=bins[1:])
    data[attr + '.final'] = data[attr + '.final'].astype('float64')
    value_final = np.array(data[attr + '.final'].values)
    value_final[np.isnan(value_final)] = -1
    value_final = np.round(value_final)

    # Evaluate data imputation
    data[attr + '.gt'] = pd.cut(data[attr + '.gt'], bins=bins, labels=bins[1:])
    data[attr + '.gt'] = data[attr + '.gt'].astype('float64')
    value_gt = np.array(data[attr + '.gt'].values)
    value_gt[np.isnan(value_gt)] = -1
    value_gt = np.round(value_gt)
    impute_accuracy = accuracy_score(value_gt, value_final)

    partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
    partition.utility = impute_accuracy
    return impute_accuracy

def apply_binning(data: pd.DataFrame, attrs_bins: dict,after_bin_flag=False) -> pd.DataFrame:
    """
    Applies robust binning to the given attributes using the specified bin edges.

    Args:
        data (pd.DataFrame): Acopy of the original dataset.
        attrs_bins (dict): Dictionary mapping attribute names to bin edge lists.

    Returns:
        pd.DataFrame: A new DataFrame with binned columns added.
    """
    data = data.copy()


    for attr, partition in attrs_bins.items():

        if not isinstance(partition, (list, np.ndarray)):  # Handle bad partition like string
            print(f"Skipping attribute '{attr}': partition is not a valid list: {partition}")
            return None  # Signal to skip this data
        min_val = data[attr].min()
        max_val = data[attr].max()

        # Build bin edges that safely cover the full range
        try:
            bins = [min_val] + [b for b in partition if b > min_val]
        except Exception as e:
            print(f"Error processing attribute '{attr}': {e} the partition {partition}")
            partition = [float(b) for b in partition]
            bins = [min_val] + [b for b in partition if b > min_val]
        if max_val > bins[-1]:
            bins.append(max_val)
        attr_binned=  f"{attr}.binned" if after_bin_flag else attr
        # Apply binning
        data[attr_binned] = pd.cut(data[attr], bins=bins, labels=False, include_lowest=True)
        data[attr_binned] = data[attr_binned].astype('float64')
        data = data.dropna(subset=[attr_binned])
    return data

def visualization_spearmanr(data, y_col, attrs_bins:Dict) -> float:
    """
    Wrapper function to visualize the data using Spearman correlation
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
    if pd.isna(interestingness):
        interestingness = 0
    if not isinstance(interestingness, (int, float)):
        interestingness = 0
    return abs(interestingness)

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# --- Parsing helpers ---------------------------------------------------------

def _as_tuple_bins(v) -> Tuple[float, ...]:
    """
    Normalize any bins representation to a tuple[float].
    Accepts: list/tuple, numpy array (any dtype/shape), pandas Series, or
             strings like '[ 0.  0.5  1.5 ]'. Returns tuple(float,...).
    """
    if isinstance(v, str):
        return tuple(float(x) for x in _NUM_RE.findall(v))
    if isinstance(v, (list, tuple, pd.Series, np.ndarray)):
        return tuple(float(x) for x in np.asarray(v).ravel().tolist())
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return tuple()
    raise TypeError(f"Unsupported bins type: {type(v)} -> {v!r}")

def _bins_cols(df: pd.DataFrame):
    """Columns that end with '_bins' in the given DataFrame."""
    return [c for c in df.columns if c.endswith("_bins")]

def bins_to_list(v):
    """Return a list[float] from strings like '[ 0.  0.5  1.5 ]', tuples, arrays, Series, etc."""
    if isinstance(v, str):
        return [float(x) for x in _NUM_RE.findall(v)]
    if isinstance(v, (list, tuple, pd.Series, np.ndarray)):
        return [float(x) for x in np.asarray(v).ravel().tolist()]
    # Leave NaN/None as-is; otherwise complain
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    raise TypeError(f"Unsupported bins type: {type(v)} -> {v!r}")

def df_bins_to_lists(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns ending with '_bins' to lists of floats."""
    out = df.copy()
    for c in [c for c in df.columns if c.endswith("_bins")]:
        out[c] = out[c].apply(bins_to_list)
    return out

# --- Fast exact-lookup LUT ---------------------------------------------------

def build_utility_lookup(
    truth_df: pd.DataFrame,
    attrs: Iterable[str] = None,
    utility_col: str = "utility",
) -> Dict[Tuple[Tuple[float, ...], ...], float]:
    """
    Build a dict mapping (bins per attr as tuples) -> utility.
    Works even if truth_df has *_bins as lists/arrays/strings.
    """
    if attrs is None:
        bins_cols = _bins_cols(truth_df)
        attrs = [c[:-5] for c in bins_cols]  # strip '_bins'
    else:
        bins_cols = [f"{a}_bins" for a in attrs]

    lut = {}
    for _, row in truth_df.iterrows():
        key = tuple(_as_tuple_bins(row[c]) for c in bins_cols)
        lut[key] = float(row[utility_col])
    return lut

def find_utility_from_partitions(
    partition_dict: Dict[str, Union[list, tuple, np.ndarray, pd.Series, str]],
    truth_or_lut: Union[pd.DataFrame, Dict[Tuple[Tuple[float, ...], ...], float]],
    attrs: Iterable[str] = None,
    utility_col: str = "utility",
) -> float:
    """
    Find the utility for bins in partition_dict.
    - If 'truth_or_lut' is a DataFrame, a LUT is built from it (using 'attrs' order
      or all '*_bins' columns if attrs=None).
    - If it's already a LUT (dict), it's used directly.
    - Accepts bins as lists/arrays/strings/etc. Exact match first; if not found and tol>0,
      falls back to tolerant scan with np.allclose(atol=tol, rtol=0).
    """
    # Normalize target bins to tuples; infer attrs if not provided and a DF is passed
    if isinstance(truth_or_lut, pd.DataFrame):
        if attrs is None:
            attrs = [c[:-5] for c in _bins_cols(truth_or_lut)]
        lut = build_utility_lookup(truth_or_lut, attrs=attrs, utility_col=utility_col)
    else:
        lut = truth_or_lut
        if attrs is None:
            # If attrs not given for a LUT, assume keys align with sorted partition_dict keys
            attrs = list(partition_dict.keys())

    target_key = tuple(_as_tuple_bins(partition_dict[a]) for a in attrs)

    # 1) Exact lookup (fast path)
    if target_key in lut:
        return float(lut[target_key])

    raise KeyError(
        f"No match for bins with attrs={tuple(attrs)}. "
        f"Target: {target_key}. Consider setting tol (e.g., tol=0.02)."
    )

def explainable_modeling_multi_attrs(data, y_col, partition_dict, lut=None) -> float:
    """
    Wrapper function to model the data using an explainable model
    ***** Note: This function is only used in demo_data_modeling_case.ipynb for now *****
    :param data: DataFrame
    :param y_col: str
    :param partition_dict: Dict[str, Partition]
    :return: float
    """
    if lut is not None:
        model_accuracy = find_utility_from_partitions(partition_dict, lut, attrs=("passenger_count","trip_distance","duration"))
        return model_accuracy
    start_time = time.time()
    data = apply_binning(data, partition_dict,after_bin_flag=True)
    if data is None:
        print("Skipping utility computation due to invalid partition.")
        return None
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

    return model_accuracy

def causal_inference_utility(raw_data, outcome, partition_dict, confounders, treatment, use_case,unbinned_pre_utility,after_bin_flag=False) :
    data = raw_data.copy()
    # Apply binning to the data based on the partition_dict
    data=apply_binning(data, partition_dict,after_bin_flag)
    if data is None:
        print("Skipping utility computation due to invalid partition.")
        return None if not after_bin_flag else (None, None)
    # Compute the causal effect using the specified method
    utility = data_causal_score(data, outcome, treatment, confounders, use_case, unbinned_pre_utility)
    return (utility,data) if after_bin_flag else utility

def get_unbbined_data_causal_utility(causal_case,data,known_args=None):
    if causal_case == "known":
        treatment_attr, y_col, confounders = known_args
        unbinned_pre_utility = compute_causal_effect(data, treatment_attr, y_col,
                                                          confounders=confounders)  # this is the "real" ATE
    else:  # use_case == "unknown":
        lower_tau, higher_tau, unbinned_correlations = warper_get_thresholds(data)
        unbinned_pre_utility = [lower_tau, higher_tau, unbinned_correlations]
    return unbinned_pre_utility




def data_imputation_multi_attrs(data, y_col, partition_dict, imputer='KNN', lut=None) -> float:
    """
    Wrapper function to impute missing values in a dataset
    :param data: DataFrame
    :param attr: str
    :param bins: List
    :return: float
    """
    if lut is not None:
        accuracy = find_utility_from_partitions(partition_dict, lut, attrs=("passenger_count","trip_distance","duration"))
        return accuracy
    start_time = time.time()
    # Bin attr column, with nan values
    for attr, partition in partition_dict.items():
        #print(f"Discretizing {attr}...")
        if len(partition) < 2:
            if data[attr].isna().all():
                print(f"Skipping '{attr}': all values are NaN.")
                continue
            partition = [-np.inf, partition[0], np.inf]
        data[attr + '.binned'] = pd.cut(data[attr], bins=partition, labels=False)
        
    # Impute the missing values using KNN
    gt_columns = [attr + '.gt' for attr in partition_dict.keys()]
    X_cols = [col for col in data.columns if col != y_col and col not in list(partition_dict.keys()) and col not in gt_columns]
    X = data[X_cols]
    avg_length = np.mean([len(partition) for partition in partition_dict.values()])
    if imputer == 'KNN':
        imputer = KNNImputer(n_neighbors= int(avg_length))
        X_imputed = imputer.fit_transform(X)
    elif imputer == 'Iterative':
        imputer = IterativeImputer(max_iter=3, random_state=0)
        X_imputed = imputer.fit_transform(X)
    elif imputer == 'Simple':
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
    else:
        raise ValueError(f"Unknown imputer: {imputer}")
    
    # Bin imputed values
    accuracy = []
    #data_imputed = np.round(X_imputed[:, idx])
    for attr, partition in partition_dict.items():
        idx = X.columns.get_loc(attr + '.binned')
        data[attr+'.imputed'] = X_imputed[:, idx]
        data[attr + '.final'] = pd.cut(data[attr+'.imputed'], bins=partition, labels=partition[1:])
        data[attr + '.final'] = data[attr + '.final'].astype('float64')
        value_final = np.array(data[attr + '.final'].values)
        value_final[np.isnan(value_final)] = -1
        value_final = np.round(value_final)

        # Evaluate data imputation
        data[attr + '.gt'] = pd.cut(data[attr + '.gt'], bins=partition, labels=partition[1:])
        data[attr + '.gt'] = data[attr + '.gt'].astype('float64')
        value_gt = np.array(data[attr + '.gt'].values)
        value_gt[np.isnan(value_gt)] = -1
        value_gt = np.round(value_gt)
        impute_accuracy = accuracy_score(value_gt, value_final)
        accuracy.append(impute_accuracy)

    #partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
    #partition.utility = impute_accuracy
    return np.mean(accuracy)

def get_runtime_stats(search_space, semantic_metric='l2_norm', indices=None) -> List:
    # Get runtime statistics
    runtime_stats = []
    runtime_df = search_space.get_runtime()
    partition_gen = runtime_df[runtime_df['function'] == 'get_bins']['runtime'].sum()
    if indices is not None:
        runtime_df = runtime_df[runtime_df['ID'].isin(indices)]
        num_explored_points = len(indices)
    else:
        num_explored_points = len(search_space.candidates)
    
    runtime_stats.append(num_explored_points)
    runtime_stats.append(partition_gen)
    functions = [f'cal_{semantic_metric}', 'utility_comp']
    for f in functions:
        total_time = runtime_df[runtime_df['function'] == f]['runtime'].sum()
        runtime_stats.append(total_time)
    return runtime_stats

def DBSCAN_distributions(search_space, parameters) -> List:
    """
    :param search_space: PartitionSearchSpace
    :return: List of clusters
    """
    eps = parameters['eps']
    min_samples = parameters['min_samples']
    X = np.array([p.distribution for p in search_space.candidates], dtype=object)
    X = pairwise_distance(X, metric=wasserstein_distance)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    dbscan_clusters = model.fit_predict(X)
    # For outliers, assign them to each of their separate cluster
    max_cluster = np.max(dbscan_clusters)
    for i, c in enumerate(dbscan_clusters):
        if c == -1:
            dbscan_clusters[i] = max_cluster + 1
            max_cluster += 1
    return dbscan_clusters

#def HDBSCAN_distributions(search_space, parameters) -> List:
#    """
#    :param search_space: PartitionSearchSpace
#    :return: List of clusters
#    """
#    min_cluster_size = parameters['min_cluster_size']
#    X = np.array([p.distribution for p in search_space.candidates])
#    X = pairwise_distance(X, metric=wasserstein_distance)
#    model = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size)
#    hdbscan_clusters = model.fit_predict(X)
#    # For outliers, assign them to each of their separate cluster
#    max_cluster = np.max(hdbscan_clusters)
#    for i, c in enumerate(hdbscan_clusters):
#        if c == -1:
#            hdbscan_clusters[i] = max_cluster + 1
#            max_cluster += 1
#    return hdbscan_clusters

#def HDBSCAN_binned_values(search_space, parameters) -> List:
#    """
#    :param search_space: PartitionSearchSpace
#    :return: List of clusters
#    """
#    n_components = parameters['n_components']
#    X = np.array([p.binned_values for p in search_space.candidates])
#    model = hdbscan.HDBSCAN(min_cluster_size=2)
#    pca = PCA(n_components=n_components)
#    X = pca.fit_transform(X)
#    hdbscan_clusters = model.fit_predict(X)
#    return hdbscan_clusters

def linkage_distributions(search_space, parameters) -> List:
    """
    :param search_space: PartitionSearchSpace
    :param parameters: Dict
    :return: List of clusters
    """
    t = parameters['t']
    criterion = parameters['criterion']
    X = np.array([p.distribution for p in search_space.candidates], dtype=object)
    X = pairwise_distance(X, metric=wasserstein_distance)
    Z = linkage(X, method='ward')
    agg_clusters = fcluster(Z, t=t, criterion=criterion)
    agg_clusters = [x-1 for x in agg_clusters] # 0-indexing
    return agg_clusters

def linkage_binned_values(search_space, parameters: Dict[str, Any]) -> List[int]:
    """
    Linkage (Ward) clustering over candidate binned_values using Euclidean distance.

    parameters:
      - t : float
          Threshold used by scipy.cluster.hierarchy.fcluster.
      - criterion : str
          fcluster criterion (e.g., 'distance', 'maxclust', ...).
      - n_components : int | None (optional)
          If provided, reduce binned_values with PCA(n_components) before clustering.
      - standardize : bool (optional, default False)
          If True, z-score features before PCA/clustering.

    Returns:
      List[int] : 0-indexed cluster labels.
    """
    t = parameters["t"]
    criterion = parameters["criterion"]
    n_components = 10
    standardize = parameters.get("standardize", False)

    # Collect binned values into a 2D array (n_samples x n_features)
    X = np.array([p.binned_values for p in search_space.candidates], dtype=float)

    # Optional standardization (often helpful before PCA/Ward)
    if standardize:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    # Optional dimensionality reduction to mirror your HDBSCAN pipeline
    if n_components is not None:
        pca = PCA(n_components=n_components, random_state=0)
        X = pca.fit_transform(X)

    # Ward linkage expects observations; distance metric is effectively Euclidean
    Z = linkage(X, method="ward")  # Euclidean implied for 'ward'

    # Flat clusters with user-specified threshold/criterion
    labels = fcluster(Z, t=t, criterion=criterion)

    # Convert to 0-indexed labels to match your other functions
    return [int(lbl - 1) for lbl in labels]

def parallel_pdist(X):
    """Parallel version of pdist using joblib"""
    n = len(X)
    dists = Parallel(n_jobs=-1)(
        delayed(wasserstein_distance_strategy)(X[i], X[j]) 
        for i in range(n) for j in range(i+1, n)
    )
    return dists

def linkage_strategies(strategy_space, parameters) -> list:
    t = parameters['t']
    criterion = parameters['criterion']

    X = [tuple(p.distribution for p in s.partition_list) for s in strategy_space.candidates]

    # Parallelized distance computation
    X = parallel_pdist(X)

    Z = linkage(X, method='ward')
    agg_clusters = fcluster(Z, t=t, criterion=criterion)
    agg_clusters = [x-1 for x in agg_clusters] # 0-indexing
    return agg_clusters

def wasserstein_distance_strategy(s1, s2) -> float:
    """
    Compute the average Wasserstein distance between corresponding distributions in s1 and s2.
    
    :param s1: List of distributions (arrays) for strategy 1
    :param s2: List of distributions (arrays) for strategy 2
    :return: Average Wasserstein distance
    """
    return np.mean([wasserstein_distance(a, b) for a, b in zip(s1, s2)])

def random_sampling_clusters(cluster_assignments, parameters) -> List:
    p = parameters['p']
    budget = int(len(cluster_assignments) * p)
    sampled_indices = []
    print("Budget start:", budget)
    print("Unique clusters:", np.unique(cluster_assignments))
    if budget >= len(np.unique(cluster_assignments)):
        # Only sample one partition from each cluster
        for c in np.unique(cluster_assignments):
            cluster_indices = np.where(cluster_assignments == c)[0]
            # Sample one partition from the cluster
            sampled_indices.extend(np.random.choice(cluster_indices, 1, replace=False))
        if budget > len(np.unique(cluster_assignments)):
            assignments = cluster_assignments.copy()
            # Remove sampled_indices from the cluster assignments by index
            new_list_to_sample = [item for i, item in enumerate(assignments) if i not in sampled_indices]
            sampled_indices.extend(np.random.choice(new_list_to_sample, budget - len(sampled_indices), replace=False))
        # Add gold standard partition to the sampled partitions
        if 0 not in sampled_indices:
            sampled_indices.append(0)
    
    return sampled_indices

def random_sampling_clusters_robust(cluster_assignments, parameters) -> List:
    """
    Sample partitions from clusters with a budget constraint.
    Similar to random_sampling_clusters, but more robust.
    In the sense that if the budget (n) is less than the number of clusters,
    this method will sample n clusters and sample one partition from each cluster.
    """
    p = parameters['p']
    budget = max(int(len(cluster_assignments) * p) - 1, 1)
    sampled_indices = []
    if budget >= len(np.unique(cluster_assignments)):
        sampled_indices = random_sampling_clusters(cluster_assignments, parameters)
    
    # Sample clusters when budget is less than the number of clusters
    else:
        sampled_clusters = np.random.choice(np.unique(cluster_assignments), budget, replace=False)
        for c in sampled_clusters:
            cluster_indices = np.where(cluster_assignments == c)[0]
            sampled_indices.extend(np.random.choice(cluster_indices, 1, replace=False))
    # Add gold standard partition to the sampled partitions
    if 0 not in sampled_indices:
        sampled_indices.append(0)
    return sampled_indices

def random_with_inverse_sampling_clusters(cluster_assignments, parameters) -> List:
    p = parameters['p']
    budget = int(len(cluster_assignments) * p)
    sampled_indices = []
    print("Budget start:", budget)
    print("Unique clusters:", np.unique(cluster_assignments))
    if budget >= len(np.unique(cluster_assignments)):
        # Only sample one partition from each cluster
        for c in np.unique(cluster_assignments):
            cluster_indices = np.where(cluster_assignments == c)[0]
            # Sample one partition from the cluster
            sampled_indices.extend(np.random.choice(cluster_indices, 1, replace=False))
        
        if budget > len(np.unique(cluster_assignments)):
            assignments = cluster_assignments.copy()
            # Remove sampled_indices from the cluster assignments by index
            new_cluster_assignments = [item for i, item in enumerate(assignments) if i not in sampled_indices]
            #sampled_indices.extend(np.random.choice(new_list_to_sample, budget - len(sampled_indices), replace=False))
            budget = budget - len(sampled_indices)
            # Calculate cluster size from cluster assignment
            cluster_size = [len(np.where(new_cluster_assignments == c)[0]) for c in np.unique(cluster_assignments)]
            cluster_probs = 1 / (cluster_size / np.sum(cluster_size))
            cluster_probs = cluster_probs / np.sum(cluster_probs)
            cluster_probs = np.nan_to_num(cluster_probs)
            # get number of samples per cluster, with at least one sample per cluster
            cluster_samples = find_actual_cluster_sample_size(budget, cluster_probs, cluster_size)
            # sample from each cluster based on the number of samples
            for c in np.unique(new_cluster_assignments):
                cluster_indices = np.where(new_cluster_assignments == c)[0]
                sampled_indices.extend(np.random.choice(cluster_indices, cluster_samples[c], replace=False))

        
        # Add gold standard partition to the sampled partitions
        if 0 not in sampled_indices:
            sampled_indices.append(0)
    
    return sampled_indices
    
def proportional_sampling_clusters(cluster_assignments, parameters) -> List:
    # Proportionally sample from each cluster
    p = parameters['p']
    budget = int(len(cluster_assignments) * p)
    clusters = [i for i in range(len(np.unique(cluster_assignments)))]
    # get cluster probabilities
    cluster_probs = np.bincount(cluster_assignments) / len(cluster_assignments)
    cluster_size = [len(np.where(cluster_assignments == c)[0]) for c in np.unique(cluster_assignments)]
    # get number of samples per cluster, with at least one sample per cluster
    cluster_samples = find_actual_cluster_sample_size(budget, cluster_probs, cluster_size)
    # get number of samples per cluster, with at least one sample per cluster
    #cluster_samples = [0] * len(np.unique(cluster_assignments))
    #samples = np.random.choice(clusters, p=cluster_probs, size=budget)
    #for c in samples: cluster_samples[c] += 1
    # sample from each cluster based on the number of samples
    sampled_indices = []
    for c in np.unique(cluster_assignments):
        cluster_indices = np.where(cluster_assignments == c)[0]
        sampled_indices.extend(np.random.choice(cluster_indices, cluster_samples[c], replace=False))

    # Add gold standard partition to the sampled partitions
    if 0 not in sampled_indices:
        sampled_indices.append(0)
    return sampled_indices

def find_actual_cluster_sample_size(total_budget, norm_inv_probs, cluster_sizes):

    # Step 3: Calculate the ideal number of samples for each cluster
    # Based on the inverse probabilities and total budget
    ideal_samples = [int(p * total_budget) for p in norm_inv_probs]
    ideal_excess = sum(ideal_samples) - total_budget
    #print("Inv probs:", norm_inv_probs)
    #print("Ideal samples:", ideal_samples)

    # Step 4: Initialize an array to track the actual samples drawn from each cluster
    actual_samples = [0] * len(norm_inv_probs)

    # Step 5: First pass: Assign as many samples as possible without exceeding cluster capacity
    excess_budget = 0  # Track how much of the budget is left after clusters with limited points
    for i in range(len(norm_inv_probs)):
        if ideal_samples[i] <= cluster_sizes[i]:
            # We can sample the ideal number from this cluster
            actual_samples[i] = ideal_samples[i]
        else:
            # Not enough points in this cluster, so sample all available points
            actual_samples[i] = cluster_sizes[i]
            # Add the remaining unused budget
            excess_budget += ideal_samples[i] - cluster_sizes[i]
    if ideal_excess < 0: excess_budget -= ideal_excess
    #print("Excess budget:", excess_budget)
    
    # Step 6: Redistribute the excess budget
    # Only distribute to clusters that still have points left to sample
    prev_excess_budget = 0
    remaining_inv_probs, remaining_inv_sum = [], 0
    clusters = [i for i in range(len(cluster_sizes))]
    while excess_budget > 0 and excess_budget != prev_excess_budget:
        #print("Excess budget:", excess_budget)
        remaining_inv_probs = [inv_p if actual_samples[i] < cluster_sizes[i] else 0 for i, inv_p in enumerate(norm_inv_probs)]
        remaining_inv_sum = sum(remaining_inv_probs)
        #print("Remaining inv probs:", np.array(remaining_inv_probs) / remaining_inv_sum)
        # remove clusters that have been fully sampled
        clusters = [i for i in range(len(cluster_sizes)) if remaining_inv_probs[i] > 0]
        remaining_inv_probs = [inv_p for i, inv_p in enumerate(remaining_inv_probs) if inv_p > 0]
        
        if remaining_inv_sum == 0:
            break  # No more clusters to redistribute to
        
        additionals = [0] * len(norm_inv_probs)
        samples = np.random.choice(clusters, p=np.array(remaining_inv_probs)/remaining_inv_sum, size=excess_budget)
        for c in samples: additionals[c] += 1

        for i in range(len(norm_inv_probs)):
            if actual_samples[i] < cluster_sizes[i]:
                # Compute additional samples to allocate
                additional_samples = additionals[i]
                #print("Additional samples:", additional_samples)
                # Ensure we don't exceed the cluster's capacity
                available_capacity = cluster_sizes[i] - actual_samples[i]
                
                if additional_samples <= available_capacity:
                    actual_samples[i] += additional_samples
                    prev_excess_budget = excess_budget
                    excess_budget -= additional_samples
                else:
                    # Take all remaining points from the cluster and update the excess budget
                    actual_samples[i] += available_capacity
                    prev_excess_budget = excess_budget
                    excess_budget -= available_capacity
    
    # Output: The final number of samples to draw from each cluster
    #print("Actual samples:", actual_samples)
    return actual_samples


def reverse_propotional_sampling_clusters(cluster_assignments, parameters) -> List:
    p = parameters['p']
    budget = int(len(cluster_assignments) * p)
    #print("Budget start:", budget)
    cluster_probs = 1 / (np.bincount(cluster_assignments) / len(cluster_assignments))
    cluster_probs = cluster_probs / np.sum(cluster_probs)
    # Calculate cluster size from cluster assignment
    cluster_size = [len(np.where(cluster_assignments == c)[0]) for c in np.unique(cluster_assignments)]
    sampled_indices = []
    # get number of samples per cluster, with at least one sample per cluster
    cluster_samples = find_actual_cluster_sample_size(budget, cluster_probs, cluster_size)
    # sample from each cluster based on the number of samples
    for c in np.unique(cluster_assignments):
        cluster_indices = np.where(cluster_assignments == c)[0]
        sampled_indices.extend(np.random.choice(cluster_indices, cluster_samples[c], replace=False))
    
    # Add gold standard partition to the sampled partitions
    if 0 not in sampled_indices:
        sampled_indices.append(0)
    return sampled_indices

def reverse_propotional_sampling_clusters_(cluster_assignments, parameters) -> List:
    budget = int(len(cluster_assignments) * 0.2)
    # order the clusters by size
    cluster_sizes = np.bincount(cluster_assignments)
    sorted_clusters = np.argsort(cluster_sizes)
    sampled_indices = []
    for c in sorted_clusters:
        c_budget = budget - len(sampled_indices)
        cluster_indices = np.where(cluster_assignments == c)[0]
        if len(cluster_indices) > c_budget:
            sampled_indices.extend(np.random.choice(cluster_indices, c_budget, replace=False))
        else: sampled_indices.extend(cluster_indices)
        if len(sampled_indices) >= budget:
            break
    
    # Add gold standard partition to the sampled partitions
    if 0 not in sampled_indices:
        sampled_indices.append(0)
    return sampled_indices

def cluster_sampling(search_space, clustering, sampling, semantic_metric='l2_norm', clustering_params:Dict={}, sampling_params:Dict={}, if_runtime_stats=True) -> List:
    runtime_stats = []

    # Cluster the binned data
    start_time = time.time()
    
    cluster_assignments = clustering(search_space, clustering_params)

    sampled_indices = sampling(cluster_assignments, sampling_params)
    if len(sampled_indices) == 0:
        return None, None, runtime_stats, cluster_assignments
    
    sampled_partitions = [search_space.candidates[i] for i in sampled_indices]
    explored_points, pareto_points, _ = get_pareto_front(sampled_partitions, semantic_metric)
    
    method_comp = time.time() - start_time
    #points_df['Cluster'] = cluster_assignments

    # Compute the runtime statistics
    if if_runtime_stats:
        runtime_stats.extend(get_runtime_stats(search_space, semantic_metric, sampled_indices))
        runtime_stats.append(method_comp)
    return explored_points, pareto_points, runtime_stats, cluster_assignments

def random_sampling(search_space, semantic_metric='l2_norm', frac=0.5, if_runtime_stats=True) -> List:
    runtime_stats = []
    start_time = time.time()
    # Sample frac of the partitions
    sampled_indices = np.random.choice(len(search_space.candidates), int(len(search_space.candidates) * frac), replace=False)
    sampled_partitions = [search_space.candidates[i] for i in sampled_indices]
    explored_points, pareto_points, _ = get_pareto_front(sampled_partitions, semantic_metric)

    method_comp = time.time() - start_time
    # Compute the runtime statistics
    if if_runtime_stats:
        runtime_stats.extend(get_runtime_stats(search_space, semantic_metric, sampled_indices))
        runtime_stats.append(method_comp)
    return explored_points, pareto_points, runtime_stats

def get_points(partitions, semantic_metric) -> List:
    if semantic_metric == 'l2_norm':
        semantics = [p.l2_norm for p in partitions]
    elif semantic_metric == 'gpt_semantics':
        semantics = [p.gpt_semantics / 4 for p in partitions]
    elif semantic_metric == 'KLDiv':
        semantics = [p.KLDiv for p in partitions]
    else: raise ValueError("Invalid semantic metric")
    utility = [p.utility for p in partitions]
    datapoints = [np.array(semantics), np.array(utility)]
    return datapoints

def get_pareto_front(candidates, semantic_metric='l2_norm') -> List:
    datapoints = get_points(candidates, semantic_metric)
    IDs = [p.ID for p in candidates]
    #print(f"Data points: {datapoints}")
    #print("Datapoint shape to compute Pareto points:", np.array(datapoints).shape)
    lst = compute_pareto_front(datapoints)

    # Plot the Pareto front
    pareto_df = pd.DataFrame({'ID': IDs, 'Semantic': datapoints[0], 'Utility': datapoints[1]})
    pareto_df['Pareto'] = 0
    pareto_df.loc[lst, 'Pareto'] = 1
    # TODO: Add the partition to the dataframe, robust to StrategySpace
    if hasattr(candidates[0], "bins"):
        pareto_df['Partition'] = [[p.bins] for p in candidates]
    else: # Strategy: candidates is List[Strategy]
        partition_col_bins = []
        for strategy in candidates:
            bins = [p.bins for p in strategy.partition_list]
            partition_col_bins.append(bins)
        pareto_df['Partition'] = partition_col_bins
    pareto_points = pareto_df[pareto_df['Pareto'] == 1][['Semantic', 'Utility']]
    pareto_points = pareto_points.values.tolist()
    #print(f"Pareto points: {pareto_points}")
    return datapoints, pareto_points, pareto_df

# --------- DBSCAN helpers -----------------------------

# assumes you already have:
# - pairwise_distance(X, metric=...)
# - wasserstein_distance

def _normalize_01(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    a_min, a_max = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max == a_min:
        return np.zeros_like(arr, dtype=float)
    return (arr - a_min) / (a_max - a_min)

def _default_min_samples_grid(n: int) -> List[int]:
    raw = [3, 4, 5, 7, 10, int(max(2, np.sqrt(n))), int(max(2, n/10))]
    return sorted({int(x) for x in raw if 2 <= int(x) <= max(2, n-1)})

def _knn_radii(D: np.ndarray, k: int) -> np.ndarray:
    """
    For each point, return distance to its k-th nearest neighbor (k>=1),
    using precomputed distance matrix D (zeros on diagonal).
    """
    # sort each row (ascending); skip first element (self distance = 0)
    row_sorted = np.sort(D, axis=1)[:, 1:]
    k = max(1, min(k, row_sorted.shape[1]))
    return row_sorted[:, k-1]

def find_optimal_dbscan_params(
    search_space,
    min_samples_grid: Optional[List[int]] = None,
    eps_quantiles: Tuple[float, ...] = (0.15, 0.25, 0.35, 0.5, 0.65, 0.8),
    coverage_weight: float = 0.30,
    verbose: bool = True,
) -> Tuple[Dict[str, float], Dict]:
    """
    Auto-tune DBSCAN(eps, min_samples) on Wasserstein distances.

    For each min_samples = m:
      - compute k-NN radii (k=m),
      - set eps candidates from specified quantiles of those radii,
      - fit DBSCAN(metric='precomputed') for each eps.

    Score = (1 - coverage_weight) * normalized_silhouette + coverage_weight * coverage,
    where coverage = (# non-noise) / n, silhouette computed on non-noise points only
    with metric='precomputed'. Combos with <2 clusters (among non-noise) are skipped.

    Returns
    -------
    best_params: {'eps': float, 'min_samples': int}
    info: dict with details (per-combo scores, silhouettes, coverages, etc.)
    """
    X = np.array([p.distribution for p in search_space.candidates], dtype=object)
    D = pairwise_distance(X, metric=wasserstein_distance).astype(float)
    n = D.shape[0]
    if n < 3:
        # trivial: one cluster
        return {"eps": float(np.max(D)), "min_samples": 2}, {"candidates": [], "skipped": ["n<3"]}

    if min_samples_grid is None:
        min_samples_grid = _default_min_samples_grid(n)

    cand_ms, cand_eps = [], []
    silhouettes, coverages, nclusters = [], [], []
    skipped = []

    for m in min_samples_grid:
        try:
            radii = _knn_radii(D, m)
            # ensure finite radii; guard against all zeros
            radii = radii[np.isfinite(radii)]
            if radii.size == 0:
                if verbose:
                    print(f"[m={m}] skip: no finite k-NN radii")
                skipped.append((m, "no_radii"))
                continue
            eps_grid = sorted({float(np.quantile(radii, q)) for q in eps_quantiles if 0.0 <= q <= 1.0})
            # avoid zero eps (degenerate); add a tiny epsilon if needed
            eps_grid = [e if e > 0 else float(np.nextafter(0, 1)) for e in eps_grid]

            for eps in eps_grid:
                try:
                    labels = DBSCAN(eps=eps, min_samples=m, metric='precomputed').fit_predict(D)

                    mask = labels != -1
                    kept = int(mask.sum())
                    cov = kept / n

                    # need at least two clusters among non-noise
                    if kept < 3 or len(np.unique(labels[mask])) < 2:
                        if verbose:
                            print(f"[m={m}, eps={eps:.4g}] skip: kept={kept}, clusters={len(np.unique(labels[mask]))}")
                        continue

                    D_sub = D[np.ix_(mask, mask)]
                    sil = silhouette_score(D_sub, labels[mask], metric='precomputed')

                    cand_ms.append(m)
                    cand_eps.append(eps)
                    silhouettes.append(sil)
                    coverages.append(cov)
                    nclusters.append(len(np.unique(labels[mask])))

                    if verbose:
                        print(f"[m={m}, eps={eps:.4g}] sil={sil:.4f}, cov={cov:.3f}, k={nclusters[-1]}")

                except Exception as e:
                    if verbose:
                        print(f"[m={m}, eps={eps:.4g}] error: {e}")
                    skipped.append((m, eps))
                    continue

        except Exception as e:
            if verbose:
                print(f"[m={m}] error computing kNN radii: {e}")
            skipped.append((m, "knn_error"))
            continue

    if not cand_ms:
        # fallback: a middle-of-the-road choice
        fallback_m = max(3, int(np.sqrt(n)))
        # eps from median of 5-NN radii (if possible), else global median distance
        try:
            r5 = _knn_radii(D, min(5, n-1))
            fallback_eps = float(np.median(r5[np.isfinite(r5)]))
        except Exception:
            # take median of nonzero upper-triangle distances
            tri = D[np.triu_indices_from(D, k=1)]
            tri = tri[np.isfinite(tri)]
            tri = tri[tri > 0]
            fallback_eps = float(np.median(tri)) if tri.size else float(np.max(D))
        if verbose:
            print(f"[auto] fallback: min_samples={fallback_m}, eps={fallback_eps:.4g}")
        return {"eps": fallback_eps, "min_samples": fallback_m}, {
            "candidates": [], "skipped": skipped, "fallback": True
        }

    sil_arr = np.array(silhouettes, dtype=float)
    cov_arr = np.array(coverages, dtype=float)
    sil_norm = _normalize_01(sil_arr)
    score = (1.0 - coverage_weight) * sil_norm + coverage_weight * cov_arr

    best_idx = int(np.nanargmax(score))
    best = {
        "eps": float(cand_eps[best_idx]),
        "min_samples": int(cand_ms[best_idx]),
    }
    info = {
        "candidates": [{"min_samples": int(ms), "eps": float(e)} for ms, e in zip(cand_ms, cand_eps)],
        "silhouette": sil_arr.tolist(),
        "sil_norm": sil_norm.tolist(),
        "coverage": cov_arr.tolist(),
        "score": score.tolist(),
        "n_clusters": nclusters,
        "best": {
            "min_samples": best["min_samples"],
            "eps": best["eps"],
            "score": float(score[best_idx]),
        },
        "skipped": skipped,
    }
    if verbose:
        print(f"[auto] best: m={best['min_samples']}, eps={best['eps']:.4g}, "
              f"score={info['best']['score']:.4f}")
    return best, info
