import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import squareform
from scipy.stats import wasserstein_distance
import json
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.framework.TestSearchSpace import TestSearchSpace
import warnings
warnings.filterwarnings('ignore')

def pairwise_distance(X, metric):
    distances = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            distances[i, j] = metric(X[i], X[j])
    return distances

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
dataset = 'crop'  # Change to your dataset
exp_config = json.load(open(os.path.join(project_root, 'data', dataset, f'{dataset}.json')))
attributes = list(exp_config['attributes'].keys())
# Load your data here
# Create search space
search_space = {}
for attr in attributes:
    attr_df = pd.read_csv(os.path.join(project_root, 'data', dataset, 'scored_attributes', f'{attr}.csv'))
    search_space[attr] = TestSearchSpace(attr_df, attribute=attr)


for i in range(len(attributes)):
    print(f"Processing attribute: {attributes[i]}")
    X = np.array([p.distribution for p in search_space[attributes[i]].candidates])
    X = pairwise_distance(X, metric=wasserstein_distance)

    # Convert to condensed form if needed
    condensed_X = squareform(X)  # X must be square and symmetric

    # Compute linkage
    Z = linkage(condensed_X, method='ward')

    # Cophenetic correlation and distances
    coph_corr, coph_dists = cophenet(Z, condensed_X)
    print(f"Reasonable t range: [{np.min(coph_dists):.2f}, {np.max(coph_dists):.2f}]")
    # Get second smallest unique cophenetic distance
    unique_dists = np.unique(coph_dists)

    if len(unique_dists) >= 2:
        second_smallest = unique_dists[1]
        print(f"Second smallest cophenetic distance: {second_smallest:.4f}")
    
    t_values = np.linspace(0.3, 1.0, 20)

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
    print("Best t based on combined score:")
    print(f"t = {ts[best_idx]:.2f}")
    print(f"Silhouette = {sils[best_idx]:.4f}, CH = {chs[best_idx]:.2f}, DB = {dbs[best_idx]:.4f}\n")