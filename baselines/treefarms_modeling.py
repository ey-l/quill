######################################################################
# 0 .  Imports & dataset ---------------------------------------------
######################################################################
import pandas as pd
from treefarms import TREEFARMS
import time
import sys
import os

SPECS_DIR    = "./pima/scored_attributes" 
LABEL_COL    = "Outcome"        # kept (if present) at the end
ATTRIBUTES   = ["Age", "BMI", "Glucose", "Outcome"]  # list of attributes to consider
project_root = ppath = sys.path[0] + '/../'
df_path = os.path.join(project_root, "data/pima/input/diabetes-treefarms.csv")
# your dataframe with columns  ["Age","BMI","Glucose","Outcome"]
df = pd.read_csv(df_path)
X, y          = df.drop(columns=["Outcome"]), df["Outcome"]
feature_names = list(X.columns)


######################################################################
# 1 .  Configure & fit TreeFARMS -------------------------------------
######################################################################
cfg = {
    "regularization"            : 0.1,   # ≤ sparsity penalty
    "rashomon_bound_multiplier" : 0.1,   # enumerate near-optimal trees
    "depth_budget"                 : 5       # keep trees interpretable
}

start_time = time.time()
model = TREEFARMS(cfg)
model.fit(X, y)
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

for idx in range(model.model_set.get_tree_count()):
    tree = model[idx]
    if 'feature' not in tree.source: continue
    else: 
        # Get the feature index
        feature = feature_names[tree.source['feature']]