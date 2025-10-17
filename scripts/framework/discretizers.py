import pandas as pd
import numpy as np
import sys
import os
import math
import itertools
from typing import List, Union, Any, Tuple, Dict
from collections import Counter
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error
from typing import List, Union, Any, Tuple, Dict
from permetrics.regression import RegressionMetric
from scipy.stats import wasserstein_distance, binned_statistic, spearmanr, f_oneway
from scipy.spatial import cKDTree
from astropy.stats import bayesian_blocks


def apply_bins(data, intervals:Dict[str, np.ndarray], cols:List[str]=None):
    """
    Apply the intervals to the data.
    """
    if cols is None:
        cols = intervals.keys()
    for col in cols:
        bins = intervals[col]
        data[col + '.binned'] = pd.cut(data[col], bins=bins, labels=bins[1:])
        data[col + '.binned'] = data[col + '.binned'].astype('float64')
    return data

def prep_cut_points(cut_points, min_val, col_min) -> List[float]:
    """
    Prepare the cut points for apply_bins().
    Add this step because pandas.cut does not include the minimum value.
    """
    # If the first interval is col_min, replace it with min_val
    if cut_points[0] == col_min:
        cut_points[0] = min_val
    elif cut_points[0] == min_val:
        return cut_points
    elif cut_points[0] > col_min:
        cut_points = np.insert(cut_points, 0, min_val)
    cut_points = list(cut_points)
    return cut_points

def discretize(df, n_bins:int=10, method:str='equal-width', cols:List[str]=None, min_val=None) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Discretize the continuous variables in the dataframe df.
    The method can be 'equal-width' or 'equal-frequency'.
    Return the dataframe and the intervals for each column.
    """
    intervals = {}
    if cols is None:
        cols = df.columns
    for col in cols:
        # minimum value
        if min_val is None: min_val = df[col].min()-1
        if method == 'equal-width':
            try: col_data = pd.cut(df[col], n_bins)
            except: continue
        elif method == 'equal-frequency':
            try: col_data = pd.qcut(df[col], n_bins)
            except: continue
        else:
            raise ValueError('Method must be equal-width or equal-frequency.')
        intervals[col] = col_data.unique()
    # Convert intervals to a numeric array
    for col in cols:
        #intervals[col] = list(np.insert(np.sort(np.array([x.right for x in intervals[col]])), 0, min_val))
        # No intervals for col
        if col in intervals:
            intervals[col] = np.sort(np.array([x.right for x in intervals[col]]))
            intervals[col][-1] = math.ceil(df[col].max())
        else: intervals[col] = None
    return intervals

def equal_width(df, n_bins:int=10, cols:List[str]=None, min_val=None):
    """
    Discretize the continuous variables in the dataframe df using equal-width method.
    """
    return discretize(df, n_bins, 'equal-width', cols, min_val)

def equal_frequency(df, n_bins:int=10, cols:List[str]=None, min_val=None):
    """
    Discretize the continuous variables in the dataframe df using equal-frequency method.
    """
    return discretize(df, n_bins, 'equal-frequency', cols, min_val)

def chimerge(data, attr, label, max_intervals):
    """
    Original code from: https://gist.github.com/alanzchen/17d0c4a45d59b79052b1cd07f531689e
    ChiMerge discretization algorithm.
    Example: 
        for attr in ['sepal_l', 'sepal_w', 'petal_l', 'petal_w']:
            print('Interval for', attr)
            chimerge(data=iris, attr=attr, label='type', max_intervals=6)
    """
    data[attr]=data[attr].round()
    data[label]=data[label].round(1)
    distinct_vals = sorted(set(data[attr])) # Sort the distinct values
    labels = sorted(set(data[label])) # Get all possible labels
    empty_count = {l: 0 for l in labels} # A helper function for padding the Counter()
    intervals = [[distinct_vals[i], distinct_vals[i]] for i in range(len(distinct_vals))] # Initialize the intervals for each attribute
    while len(intervals) > max_intervals: # While loop
        chi = []
        for i in range(len(intervals)-1):
            #print(i)
            # Calculate the Chi2 value
            obs0 = data[data[attr].between(intervals[i][0], intervals[i][1])]
            obs1 = data[data[attr].between(intervals[i+1][0], intervals[i+1][1])]
            total = len(obs0) + len(obs1)
            count_0 = np.array([v for i, v in {**empty_count, **Counter(obs0[label])}.items()])
            count_1 = np.array([v for i, v in {**empty_count, **Counter(obs1[label])}.items()])
            count_total = count_0 + count_1
            expected_0 = count_total*sum(count_0)/total
            expected_1 = count_total*sum(count_1)/total
            chi_ = (count_0 - expected_0)**2/expected_0 + (count_1 - expected_1)**2/expected_1
            chi_ = np.nan_to_num(chi_) # Deal with the zero counts
            chi.append(sum(chi_)) # Finally do the summation for Chi2
        min_chi = min(chi) # Find the minimal Chi2 for current iteration
        for i, v in enumerate(chi):
            if v == min_chi:
                min_chi_index = i # Find the index of the interval to be merged
                break
        new_intervals = [] # Prepare for the merged new data array
        skip = False
        done = False
        for i in range(len(intervals)):
            if skip:
                skip = False
                continue
            if i == min_chi_index and not done: # Merge the intervals
                t = intervals[i] + intervals[i+1]
                new_intervals.append([min(t), max(t)])
                skip = True
                done = True
            else:
                new_intervals.append(intervals[i])
        intervals = new_intervals
    #for i in intervals:
    #    print('[', i[0], ',', i[1], ']', sep='')
    return intervals

def chimerge_wrap(df, cols, target:str, max_intervals:int=6, min_val=None):
    """
    Wrap the chimerge function.
    Return the dataframe and the intervals for each column.
    """
    intervals = {}
    for col in cols:
        if min_val is None: min_val = df[col].min()-1
        intervals[col] = chimerge(df, col, target, max_intervals)
        intervals[col] = np.array([x[1] for x in intervals[col]]).astype(np.float32)
        intervals[col] = np.unique(intervals[col], axis=0)
        # add 1 to the last interval value
        intervals[col][-1] = math.ceil(intervals[col][-1])
        #intervals[col] = prep_cut_points(intervals[col], min_val, df[col].min())
    return intervals

def KBinsDiscretizer_wrap(df, cols, n_bins:int=10, strategy='uniform', min_val=None):
    """
    Wrap the sklearn.preprocessing.KBinsDiscretizer.
    Return the dataframe and the intervals for each column.
    """
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    try: kbd.fit(df[cols])
    except Exception as e:
        print('Error:', e)
        return {}
    intervals = {}
    for i in range(len(cols)):
        if min_val is None: min_val = df[cols[i]].min()-1
        #intervals[cols[i]] = prep_cut_points(kbd.bin_edges_[i], min_val, df[cols[i]].min())
        intervals[cols[i]] = kbd.bin_edges_[i]
    return intervals

def DecisionTreeDiscretizer_wrap(df, cols, target:str, n_bins:int=5, min_val=None):
    """
    Wrap the sklearn.tree.DecisionTreeClassifier.
    Return the dataframe and the intervals for each column.
    """
    intervals = {}
    for col in cols:
        if min_val is None: min_val = df[col].min()-1
        clf = DecisionTreeClassifier(max_leaf_nodes=n_bins)
        clf.fit(df[[col]], df[target])
        thresholds = clf.tree_.threshold[clf.tree_.feature == 0]
        thresholds = np.sort(thresholds)
        thresholds = np.append(thresholds, math.ceil(df[col].max()))
        #intervals[col] = prep_cut_points(thresholds, min_val, df[col].min())
        intervals[col] = thresholds
    return intervals

def KMeansDiscretizer_wrap(df, cols, n_bins:int=5, min_val=None):
    """
    Wrap the sklearn.cluster.KMeans.
    Return the dataframe and the intervals for each column.
    """
    intervals = {}
    for col in cols:
        if min_val is None: min_val = df[col].min()-1
        kmeans = KMeans(n_clusters=n_bins)
        kmeans.fit(df[[col]])
        thresholds = np.sort(kmeans.cluster_centers_.flatten())
        #thresholds = np.append(thresholds, df[col].max()+1)
        thresholds[-1] = math.ceil(df[col].max())
        #intervals[col] = prep_cut_points(thresholds, min_val, df[col].min())
        intervals[col] = thresholds
    return intervals

def BayesianBlocksDiscretizer_wrap(df, cols, min_val=None):
    """
    Wrap the astropy.stats.bayesian_blocks.
    Return the dataframe and the intervals for each column.
    *** Does not have n_bins parameter ***
    """
    intervals = {}
    for col in cols:
        if min_val is None: min_val = df[col].min()-1
        bayesian_bins = bayesian_blocks(df[col])
        #intervals[col] = prep_cut_points(bayesian_bins, min_val, df[col].min())
        intervals[col] = bayesian_bins
    return intervals

def MDLPDiscretizer_wrap(df, cols, target:str, min_val=None):
    """
    Wrap the mdlp.discretization.MDLP.
    Return the dataframe and the intervals for each column.
    """
    from optbinning import MDLP
    intervals = {}
    for col in cols:
        if min_val is None: min_val = df[col].min()-1
        mdlp = MDLP()
        mdlp.fit(df[col], df[target])
        thresholds = mdlp.splits
        thresholds = np.sort(thresholds)
        thresholds = np.append(thresholds, math.ceil(df[col].max()))
        #intervals[col] = prep_cut_points(thresholds, min_val, df[col].min())
        intervals[col] = thresholds
    return intervals

def RandomForestDiscretizer_wrap(df, cols, target:str, n_bins:int=5, min_val=None):
    """
    Wrap the sklearn.ensemble.RandomForestClassifier.
    Return the dataframe and the intervals for each column.
    """
    intervals = {}
    for col in cols:
        if min_val is None: min_val = df[col].min()-1
        rf = RandomForestClassifier(max_leaf_nodes=n_bins)
        rf.fit(df[[col]], df[target])
        # Extract thresholds (splits) from all trees
        thresholds = []
        for tree in rf.estimators_:
            tree_thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
            thresholds.extend(tree_thresholds)

        # Determine the most common thresholds
        thresholds = np.array(thresholds)
        unique_thresholds = np.unique(thresholds)
        
        # Select the best thresholds to form bins
        selected_thresholds = np.percentile(unique_thresholds, np.linspace(0, 100, n_bins + 1)[:-1])

        selected_thresholds = np.sort(selected_thresholds)
        selected_thresholds = np.append(selected_thresholds, math.ceil(df[col].max()))
        #intervals[col] = prep_cut_points(selected_thresholds, min_val, df[col].min())
        intervals[col] = selected_thresholds
    return intervals

def zeta_score(data, labels, cut_points):
    bins = np.digitize(data, cut_points)
    total_zeta = 0
    
    for bin_val in np.unique(bins):
        bin_labels = labels[bins == bin_val]
        class_frequencies = np.bincount(bin_labels)
        bin_zeta = np.max(class_frequencies) / len(bin_labels)
        total_zeta += bin_zeta
    
    return total_zeta

def find_optimal_cut_points(data, labels, num_bins):
    potential_cut_points = np.unique(data)
    best_zeta = -np.inf
    best_cut_points = None
    
    for cut_points in itertools.combinations(potential_cut_points, num_bins - 1):
        current_zeta = zeta_score(data, labels, cut_points)
        if current_zeta > best_zeta:
            best_zeta = current_zeta
            best_cut_points = cut_points
    
    return best_cut_points

def ZetaDiscretizer_wrap(df, cols, target:str, n_bins:int=5):
    """
    Wrap the zeta_score function.
    Return the dataframe and the intervals for each column.
    Zeta is particularly effective when the goal is to maximize class separation, making it well-suited for classification tasks.
    The optimization process can be computationally intensive, especially for large datasets with many potential cut points.
    """
    intervals = {}
    for col in cols:
        cut_points = find_optimal_cut_points(df[col].values, df[target].values, n_bins)
        #intervals[col] = prep_cut_points(cut_points, df[col].min()-1, df[col].min())
        intervals[col] = cut_points
    return intervals


if __name__ == "__main__":
    ppath = sys.path[0] + '/../../'
    # Test the discretizers
    #attrs = ['Glucose', 'BMI']
    attrs = ['loudness', 'danceability']
    #target = 'Outcome'
    target = 'song_popularity'
    n_bins = 4
    #df = pd.read_csv(os.path.join(ppath, 'data', 'pima', 'diabetes.csv'))
    df = pd.read_csv(os.path.join(ppath, 'data', 'spotify', 'songs.csv'))
    # Sort on Glucose
    #df = df.sort_values(by=['Glucose'])
    intervals = equal_width(df, n_bins, attrs)
    print("Equal width:", intervals)

    # Test the equal frequency
    intervals = equal_frequency(df, n_bins, attrs)
    print("Equal frequency:", intervals)

    # Test the chimerge
    intervals = chimerge_wrap(df, attrs, target, 6)
    print("ChiMerge:", intervals)

    # Test the KBinsDiscretizer
    intervals = KBinsDiscretizer_wrap(df, attrs, n_bins)
    print("KBinsDiscretizer:", intervals)

    # Test the KBinsDiscretizer
    intervals = KBinsDiscretizer_wrap(df, attrs, n_bins, 'quantile')
    print("KBinsDiscretizer (quantile):", intervals)

    # Test the DecisionTreeDiscretizer
    intervals = DecisionTreeDiscretizer_wrap(df, attrs, target, n_bins)
    print("DecisionTreeDiscretizer:", intervals)

    # Test the KMeansDiscretizer
    intervals = KMeansDiscretizer_wrap(df, attrs, n_bins)
    print("KMeansDiscretizer:", intervals)

    # Test the BayesianBlocksDiscretizer
    intervals = BayesianBlocksDiscretizer_wrap(df, attrs)
    print("BayesianBlocksDiscretizer:", intervals)

    # Test the MDLPDiscretizer
    intervals = MDLPDiscretizer_wrap(df, attrs, target)
    print("MDLPDiscretizer:", intervals)

    # Test the RandomForestDiscretizer
    intervals = RandomForestDiscretizer_wrap(df, attrs, target, n_bins)
    print("RandomForestDiscretizer:", intervals)

    # Test the ZetaDiscretizer
    #intervals = ZetaDiscretizer_wrap(df, attrs, target, n_bins)
    #print("ZetaDiscretizer:", intervals)