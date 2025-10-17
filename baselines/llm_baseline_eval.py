import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from llm_binning_prompts import QUESTION_PROMPT
from scripts.framework.MCC import *
from tqdm import tqdm
import argparse

load_dotenv()
sys.path.append("./")
OpenAIModelList = ["gpt-4o", "o3-2025-04-16", "gpt-4o-mini"]
output_dir = os.path.join(project_root, f"testresults/LLM/evaluation_results/final_results")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

def get_api_key(key: str) -> str:
    # get API key from environment or throw an exception if it's not set
    if key not in os.environ:
        print(f"KEY: {key}")
        print(f"{os.environ.keys()}")
        raise ValueError("key not found in environment variables")

    return os.environ[key]

def get_table_markdown(filepath, num_rows = 100):
    try: 
        table = pd.read_csv(filepath, engine='python', on_bad_lines='warn')
        return table.head(num_rows).to_csv(sep='|', index=False)
    except: return ""

def separate_runs(df: pd.DataFrame) -> dict:
    runs = {}
    run_id = 0
    current_run_rows = []

    prev_candidate = None

    for i, row in df.iterrows():
        curr_candidate = row["Candidate #"]

        if curr_candidate == 1 and (prev_candidate is not None and prev_candidate != 1):
            # Start of a new run — save the previous run if exists
            if current_run_rows:
                runs[run_id] = pd.DataFrame(current_run_rows)
                run_id += 1
                current_run_rows = []

        current_run_rows.append(row)
        prev_candidate = curr_candidate

    # Save the final run
    if current_run_rows:
        runs[run_id] = pd.DataFrame(current_run_rows)

    return runs

class Generator():
    def __init__(self, model: str="gpt-4o-mini", verbose=False):
        self.model = model
        if model in OpenAIModelList:
            self.client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))

        self.verbose = verbose

    def generate(self, prompt):
        """
        Calls the OpenAI API to get a response from the model.
        Args:
            prompt (str): The input prompt to send to the model.
        Returns:
            str: The model's response to the prompt.
        """
        max_retries = 5
        retry_count = 0
        fatal = False
        fatal_reason = None
        while retry_count < max_retries:
            try:
                ################### use to be openai.Completion.create(), now ChatCompletion ###################
                ################### also parameters are different, now messages=, used to be prompt= ###################
                # note deployment_id=... not model=...
                result = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    #max_tokens=4000,
                    # temperature=0.2,
                )
                
                break # break out of while loop if no error
            
            except Exception as e:
                print(f"An error occurred: {e}.")
                if "context_length_exceeded" in f"{e}":
                    fatal = True
                    fatal_reason = f"{e}"
                    break
                else:
                    print("Retrying...")
                    retry_count += 1
                    time.sleep(10 * retry_count)  # Wait 
        
        if fatal:
            print(f"Fatal error occured. ERROR: {fatal_reason}")
            res = f"Fatal error occured. ERROR: {fatal_reason}"
        elif retry_count == max_retries:
            print("Max retries reached. Skipping...")
            res = "Max retries reached. Skipping..."
        else:
            try:
                res = result.choices[0].message.content
            except Exception as e:
                print("Error:", e)
                res = ""
            #print(res)
        return res

def _parse_bins(val):
    """Return 1D float np.array from DF 'bins' cell (list/ndarray/string)."""
    if isinstance(val, np.ndarray):
        return val.astype(float).ravel()
    if isinstance(val, (list, tuple)):
        return np.asarray(val, dtype=float).ravel()
    if isinstance(val, str):
        s = val.strip().strip('"').strip("'")
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        # works for space/newline separated numbers (no commas needed)
        arr = np.fromstring(s, sep=' ')
        return arr.astype(float).ravel()
    raise ValueError(f"Unrecognized bins type: {type(val)}")

def _zero_pad_vectors(a, b):
    """Right-pad the shorter vector with zeros so both have equal length."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = max(a.size, b.size)
    ap = np.zeros(n, dtype=float); ap[:a.size] = a
    bp = np.zeros(n, dtype=float); bp[:b.size] = b
    return ap, bp

def _smooth_and_normalize(p, q, eps=1e-12):
    """
    Add epsilon to avoid zeros, then renormalize so they are valid distributions.
    Intended for KL(P||Q) with P=gold, Q=other.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # add epsilon to every entry to prevent log(0) / division by zero
    p = p + eps
    q = q + eps
    # renormalize
    p = p / p.sum()
    q = q / q.sum()
    return p, q

def kl_between_distributions(p_gold, q_other):
    """
    Compute directed KL divergence D_KL(p_gold || q_other), natural log.
    Handles length mismatch via zero-padding, then smooths and renormalizes.
    """
    pg, qo = _zero_pad_vectors(_parse_bins(p_gold), _parse_bins(q_other))
    pg, qo = _smooth_and_normalize(pg, qo)
    # KL = sum p * log(p/q)
    print("*************** KLDiv: ***************", float(np.sum(pg * (np.log(pg) - np.log(qo)))))
    return float(np.sum(pg * (np.log(pg) - np.log(qo))))

def min_max_kl_vs_gold(df):
    """
    Compute KL(gold distribution || other distribution) for each non-gold row.

    Returns:
        min_record, max_record
    where each record is a dict with keys: id, method, kl
    """
    gold_row = df.loc[df['method'] == 'gold-standard']
    if gold_row.empty:
        raise ValueError("No 'gold-standard' row found in df['method']")
    gold_dist = gold_row.iloc[0]['distribution']

    records = []
    for _, row in df.iterrows():
        if row['method'] == 'gold-standard':
            continue
        try:
            kl = kl_between_distributions(gold_dist, row['distribution'])
            records.append({'id': row.get('ID', None), 'method': row['method'], 'kl': kl})
        except Exception:
            # skip rows we can't parse/compute
            continue

    if not records:
        raise ValueError("No comparable rows found to compute KL.")

    min_rec = min(records, key=lambda r: r['kl'])
    max_rec = max(records, key=lambda r: r['kl'])
    return min_rec, max_rec

def l2_between_bins(x, y):
    ax, by = _zero_pad_vectors(_parse_bins(x), _parse_bins(y))
    return float(np.linalg.norm(ax - by))

def min_max_l2_vs_gold(df):
    """
    Compute L2(gold-standard bins, other bins) for each non-gold row.
    Returns:
        min_record, max_record
    where each record is a dict with keys: id, method, l2
    """
    # find gold-standard row (take the first if multiple)
    gold_row = df.loc[df['method'] == 'gold-standard']
    if gold_row.empty:
        raise ValueError("No 'gold-standard' row found in df['method']")
    gold_bins = gold_row.iloc[0]['bins']

    records = []
    for _, row in df.iterrows():
        if row['method'] == 'gold-standard':
            continue
        try:
            l2 = l2_between_bins(gold_bins, row['bins'])
            records.append({'id': row.get('ID', None), 'method': row['method'], 'l2': l2})
        except Exception as e:
            # skip rows we can't parse; optionally print(e)
            continue

    if not records:
        raise ValueError("No comparable rows found to compute L2.")

    # find min and max
    min_rec = min(records, key=lambda r: r['l2'])
    max_rec = max(records, key=lambda r: r['l2'])
    return min_rec, max_rec

# --- Example usage ---
# df = pd.read_csv(... or already constructed ...)
# min_rec, max_rec = min_max_l2_vs_gold(df)
# print("Min L2:", min_rec)
# print("Max L2:", max_rec)

def get_semantic_max_min(dataset:str, semantic_metric:str, use_case:str='modeling'):
    if use_case == 'causal_inference' :
        exp_config = json.load(open(os.path.join(project_root, 'data', dataset, f'{dataset}-causal.json')))
    else: exp_config = json.load(open(os.path.join(project_root, 'data', dataset, f'{dataset}.json')))
    attributes = exp_config['attributes'].keys()
    gold_standard=exp_config['attributes']
    raw_data = pd.read_csv(os.path.join(project_root, exp_config['data_path']))

    semantic_max_val = {}
    semantic_min_val = {}
    scored_attribute_dir = os.path.join(project_root, 'data', dataset, 'scored_attributes')
    for attr in attributes:
        attr_df = pd.read_csv(os.path.join(scored_attribute_dir, f'{attr}.csv'))
        if semantic_metric == "l2_norm":
            try:
                min_rec, max_rec = min_max_l2_vs_gold(attr_df)
                print(f"Attribute: {attr}, Min L2: {min_rec}, Max L2: {max_rec}")
                semantic_max_val[attr] = max_rec
                semantic_min_val[attr] = min_rec
            except Exception as e:
                print(f"Error processing attribute {attr}: {e}")
        elif semantic_metric == "KLDiv":
            try:
                # uses distribution column; returns {'id', 'method', 'kl'}
                min_rec, max_rec = min_max_kl_vs_gold(attr_df)
                print(f"Attribute: {attr} | Min KL: {min_rec} | Max KL: {max_rec}")
                semantic_min_val[attr] = min_rec
                semantic_max_val[attr] = max_rec
            except Exception as e:
                print(f"[WARN] KL processing error for '{attr}': {e}")
    return semantic_max_val, semantic_min_val

def get_estimated_P(dataset:str, testresult_filepath:str, semantic_metric:str='gpt_semantics', use_case:str='modeling', missing_data_fraction:float=0.3, causal_case=None, attribute=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.makedirs(output_dir, exist_ok=True)
    result_df = pd.read_csv(testresult_filepath)

    if semantic_metric in ["KLDiv", "l2_norm"]:
        semantic_max_val, semantic_min_val = get_semantic_max_min(dataset, semantic_metric, use_case)

    # Read string of dict from 'Partition (bins)' column
    if 'Partition (bins)' not in result_df.columns:
        print(f"Error: 'Partition (bins)' column not found in {testresult_filepath}.")
        return None
    result_df['Partition (bins)'] = result_df['Partition (bins)'].apply(eval)
    if use_case == 'modeling':
        result_df = result_df[result_df['Task'] == 'prediction']
    elif use_case == 'imputation':
        result_df = result_df[result_df['Task'] == 'data imputation']
    elif use_case == 'visualization':
        result_df = result_df[result_df['Task'] == 'visualization']
    elif use_case == 'causal_inference':
        result_df = result_df[result_df['Task'] == 'causal inference']
    else:
        print(f"Error: Invalid use case '{use_case}'. Expected 'modeling', 'imputation', 'visualization' or 'causal inference'.")
        return None
    
    # read json file
    if dataset == 'crop': 
        exp_config = json.load(open(os.path.join(project_root, 'data', dataset, f'{dataset}-full.json')))
    elif use_case == 'causal_inference' :
        exp_config = json.load(open(os.path.join(project_root, 'data', dataset, f'{dataset}-causal.json')))
    else: exp_config = json.load(open(os.path.join(project_root, 'data', dataset, f'{dataset}.json')))
    attributes = exp_config['attributes'].keys()
    gold_standard=exp_config['attributes']
    raw_data = pd.read_csv(os.path.join(project_root, exp_config['data_path']))
    if use_case != 'visualization': y_col = exp_config['target']
    else: y_col = exp_config['target_viz']
    raw_data = raw_data[exp_config['features'] + [y_col]]
    raw_data = raw_data.dropna(subset=exp_config['features'] + [y_col])
    if use_case == 'imputation':
        for attr in attributes:
            data = raw_data.copy()
            data[attr + '.gt'] = data[attr]
            nans = raw_data.sample(frac=missing_data_fraction, random_state=42)
            data.loc[raw_data.index.isin(nans.index),attr] = np.nan
            raw_data = data.copy()
    if use_case == 'causal_inference':
        if causal_case is None:
            raise ValueError("causal_use_case must be specified for causal inference use case.")
        if causal_case not in ['known', 'unknown']:
            raise ValueError("causal_use_case must be either 'known' or 'unknown'.")
        treatment = exp_config['treatment']
        confounders = [attr for attr in attributes if attr != treatment]  # remove treatment from confounders
        if causal_case == "known":
            unbinned_pre_utility = compute_causal_effect(raw_data, treatment, y_col,
                                                              confounders=confounders)  # this is the "real" ATE
        else:  # use_case == "unknown":
            lower_tau, higher_tau, unbinned_correlations = warper_get_thresholds(raw_data)
            unbinned_pre_utility = [lower_tau, higher_tau, unbinned_correlations]

    est_P_dict = {}
    result_dfs = {}
    # If use_case == "visualization", filter out rows where partition_dict's key[0] != attribute
    if use_case == "visualization" and attribute is not None:
        result_df = result_df[result_df['Partition (bins)'].apply(lambda x: list(x.keys())[0] == attribute)]
        if result_df.empty:
            print(f"Error: No results found for attribute '{attribute}' in visualization use case.")
            return None

    result_dfs['zero_shot'] = result_df[(result_df['Chain of Thought'] == False) & (result_df['Few-shot Used'] == False) & (result_df['RAG Used'] == False)]
    result_dfs['few_shot'] = result_df[(result_df['Chain of Thought'] == False) & (result_df['Few-shot Used'] == True) & (result_df['RAG Used'] == False)]
    result_dfs['rag'] = result_df[(result_df['Chain of Thought'] == False) & (result_df['Few-shot Used'] == False) & (result_df['RAG Used'] == True)]
    result_dfs['few_shot_rag'] = result_df[(result_df['Chain of Thought'] == False) & (result_df['Few-shot Used'] == True) & (result_df['RAG Used'] == True)]
    result_dfs['cot'] = result_df[(result_df['Chain of Thought'] == True) & (result_df['Few-shot Used'] == False) & (result_df['RAG Used'] == False)]
    result_dfs['cot_few_shot'] = result_df[(result_df['Chain of Thought'] == True) & (result_df['Few-shot Used'] == True) & (result_df['RAG Used'] == False)]
    result_dfs['cot_rag'] = result_df[(result_df['Chain of Thought'] == True) & (result_df['Few-shot Used'] == False) & (result_df['RAG Used'] == True)]
    result_dfs['cot_few_shot_rag'] = result_df[(result_df['Chain of Thought'] == True) & (result_df['Few-shot Used'] == True) & (result_df['RAG Used'] == True)]

    for variation, df in result_dfs.items():
        # Separate runs
        # if variation not in ['cot_few_shot_rag']: continue #TODO: remove this line when we have results for cot_rag and cot_few_shot_rag
        print(f"Processing variation: {variation}")
        runs_dict = separate_runs(df)
        est_P_dict[variation] = {}
        for run_id, run_df in tqdm(runs_dict.items()):
            # For each row
            for index, row in run_df.iterrows():
                data_i = raw_data.copy()
                partition_dict = row['Partition (bins)']
                semantic_score = 0
                utility_score = 0
                try:
                    # Calculate utility score
                    if use_case == 'modeling':
                        utility_score = explainable_modeling_multi_attrs(data_i, y_col, partition_dict)
                    elif use_case == 'imputation':
                        utility_score = data_imputation_multi_attrs(data_i, y_col, partition_dict, imputer="KNN")
                    elif use_case == 'visualization':
                        #raise NotImplementedError("Visualization use case is not implemented yet.")
                        # Check if partition_dict's values are all lists of numbers
                        for bins in partition_dict.values():
                            if (isinstance(bins, list) and all(isinstance(b, (int, float)) for b in bins)):
                                #raise ValueError(f"Invalid bins format for visualization: {bins}. Expected a list of numbers.")
                                try: utility_score = visualization_spearmanr(data_i, y_col, partition_dict)
                                except Exception as e: print(f"Error occurred while calculating utility score for visualization: {e}")
                            else: 
                                print(f"Warning: Skipping invalid bins format for visualization: {bins}. Expected a list of numbers.")
                                utility_score = 0
                        
                    elif use_case == 'causal_inference':
                        utility_score = causal_inference_utility(data_i, y_col, partition_dict, treatment=treatment, confounders=confounders, unbinned_pre_utility=unbinned_pre_utility, use_case=causal_case)
                    else:
                        raise ValueError(f"Unknown use case: {use_case}")
                except Exception as e:
                    print(f"Error occurred while calculating utility score: {e}")
                    utility_score = 0

                # Calculate semantic score for each attribute
                for attr, bins in partition_dict.items():
                    if semantic_metric == "gpt_semantics":
                        max_retries = 3
                        for attempt in range(1, max_retries + 1):
                            try:
                                attr_semantic = get_semantic_grade(attr, bins)
                                break  # success, exit the loop
                            except Exception as e:
                                print(f"⚠️ Error on attempt {attempt} for attribute '{attr}' with bins {bins}: {e}")
                                if "rate_limit" in str(e).lower() or "quota" in str(e).lower():
                                    wait_time = 10 * attempt  # exponential backoff
                                    print(f"⏳ Rate limit hit. Waiting {wait_time} seconds...")
                                    time.sleep(wait_time)
                                else:
                                    print(f"❌ Skipping semantic grade due to error: {e}")
                                    attr_semantic = -1
                                    break
                    elif semantic_metric == "l2_norm":
                        try:
                            x, y = zero_pad_vectors(bins, gold_standard[attr])
                            attr_semantic = np.linalg.norm(x - y)
                            attr_semantic = (attr_semantic - semantic_min_val[attr]["l2"]) / (semantic_max_val[attr]["l2"] - semantic_min_val[attr]["l2"] + 1e-8)  # normalize to [0, 1]
                        except Exception as e:
                            print(f"Error calculating L2 norm for attribute {attr} with bins {bins}: {e}")
                            attr_semantic = 0
                    elif semantic_metric == "KLDiv":
                        try:
                            values = list(raw_data[attr].values)
                            hist, _ = np.histogram(values, bins=bins)
                            distribution = hist / len(values)
                            
                            hist, _ = np.histogram(values, bins=gold_standard[attr])
                            gold_distribution = hist / len(values)
                            # Compute directed KL: D_KL(P_gold || Q_candidate)
                            # gold_standard[attr] should be the gold distribution for this attribute
                            kl_val = kl_between_distributions(gold_distribution, distribution)

                            # Normalize to [0, 1] using precomputed per-attribute extrema
                            kmin = semantic_min_val[attr].get("kl", None)
                            kmax = semantic_max_val[attr].get("kl", None)

                            if kmin is None or kmax is None:
                                # Fallback if min/max are unavailable
                                attr_semantic = 0.0
                            else:
                                denom = (kmax - kmin) + 1e-8
                                attr_semantic = (kl_val - kmin) / denom
                                # Clamp for numerical safety
                                if attr_semantic < 0.0: attr_semantic = 0.0
                                if attr_semantic > 1.0: attr_semantic = 1.0
                            print("Attribute:", attr, "KL:", kl_val, "Normalized Semantic:", attr_semantic)
                        except Exception as e:
                            print(f"Error calculating KLDiv for attribute {attr}: {e}")
                            attr_semantic = 0.0
                    else:
                        raise ValueError(f"Unknown semantic metric: {semantic_metric}")
                    semantic_score += attr_semantic
                semantic_score /= len(partition_dict)
                run_df.loc[index, semantic_metric] = semantic_score
                run_df.loc[index, 'utility'] = utility_score

            est_P_dict[variation][run_id] = np.array([run_df[semantic_metric].values, run_df['utility'].values]).T
        output_path = os.path.join(output_dir, f'{dataset}_{use_case}_{semantic_metric}_{variation}_backup.csv')
        results_df =  pd.concat([pd.DataFrame(points, columns=['semantic', 'utility']).assign(run_id=run_id)

    for run_id, points in est_P_dict[variation].items()
], ignore_index=True)
        results_df.to_csv(output_path, index=False)
    return est_P_dict

def evaluate_llm_binning(testresult_filename, dataset, semantic_metric, use_case, imputer='KNN', missing_data_fraction=0.3, use_backup=False, early_stopping=False, causal_case=None):
    results = []
    timestamp= time.strftime("%Y%m%d_%H%M")
    # Load ground truth data
    if use_case == 'modeling':
        utility_function = 'DecisionTreeClassifier'  # or 'RandomForestClassifier'
    elif use_case == 'imputation':
        utility_function = f'{imputer}Imputer.{missing_data_fraction}'  # or 'KNNImputer', 'SimpleImputer'
    elif use_case == 'visualization':
        utility_function = 'spearmanr'
    elif use_case == 'causal_inference':
        utility_function = 'causal'
    
    testresult_dir = os.path.join(project_root, 'testresults', "LLM")
    if use_backup:
        est_P_dict = use_backup_get_estimated_P(dataset, use_case=use_case, semantic_metric=semantic_metric)
    else:
        est_P_dict = get_estimated_P(dataset, os.path.join(testresult_dir, testresult_filename), semantic_metric, use_case, causal_case=causal_case)
    if est_P_dict is None:
        print(f"Error: No estimated Pareto front found in {testresult_filename}.")
        return
    if early_stopping:
        return
    
    
    gt_data = pd.read_csv(os.path.join(project_root, 'truth', f'{dataset}.multi_attrs.{utility_function}.gpt_semantics.csv'))

    evaluator = Evaluator(gt_data, semantic_metric, use_case)
    
    # Evaluate each variation
    for variation, est_P_runs in est_P_dict.items():
        gd_list = []
        igd_list = []
        ahd_list = []
        print(f"Evaluating {variation} variation...")

        for run_id, est_P in est_P_runs.items():
            # print(f"Run {run_id} - Estimated Pareto front points: {est_P}")
            est_P = est_P[~np.isnan(est_P).any(axis=1)]
            gd_value = evaluator.gd(est_P)
            igd_value = evaluator.igd(est_P)
            ahd_value = Evaluator.average_hausdorff_distance(gd_value, igd_value, mode='max')

            gd_list.append(gd_value)
            igd_list.append(igd_value)
            ahd_list.append(ahd_value)

        # Summary statistics
        gd_mean, gd_median, gd_std = np.mean(gd_list), np.median(gd_list), np.std(gd_list)
        igd_mean, igd_median, igd_std = np.mean(igd_list), np.median(igd_list), np.std(igd_list)
        ahd_mean, ahd_median, ahd_std = np.mean(ahd_list), np.median(ahd_list), np.std(ahd_list)

        print(f"{variation} - GD: mean={gd_mean:.2f}, median={gd_median:.2f}, std={gd_std:.2f}")
        print(f"{variation} - IGD: mean={igd_mean:.2f}, median={igd_median:.2f}, std={igd_std:.2f}")
        print(f"{variation} - AHD: mean={ahd_mean:.2f}, median={ahd_median:.2f}, std={ahd_std:.2f}")

        results.append({
            'variation': variation,
            'GD_mean': gd_mean,
            'GD_median': gd_median,
            'GD_std': gd_std,
            'IGD_mean': igd_mean,
            'IGD_median': igd_median,
            'IGD_std': igd_std,
            'AHD_mean': ahd_mean,
            'AHD_median': ahd_median,
            'AHD_std': ahd_std
        })

    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'eval_{dataset}_{use_case}_{semantic_metric}_{timestamp}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Evaluation results saved to: {output_path}")

def evaluate_llm_binning_viz(testresult_filename, dataset, semantic_metric, use_case, use_backup=False):
    timestamp= time.strftime("%Y%m%d_%H%M")
    utility_function = 'spearman'
    
    testresult_dir = os.path.join(project_root, 'testresults', "LLM")
    
    exp_config = json.load(open(os.path.join(project_root, 'data', dataset, f'{dataset}.json')))
    #attributes = exp_config['attributes'].keys()
    attributes = [exp_config['attribute_viz']]  # For single attribute evaluation
    for attr in attributes:
        results = []
        gt_data = pd.read_csv(os.path.join(project_root, 'truth', 'single_attrs', f'{dataset}.{attr}.{utility_function}.csv'))
        evaluator = Evaluator(gt_data, semantic_metric, use_case)

        if use_backup:
            est_P_dict = use_backup_get_estimated_P(dataset, use_case=use_case, semantic_metric=semantic_metric)
        else:
            est_P_dict = get_estimated_P(dataset, os.path.join(testresult_dir, testresult_filename), semantic_metric, use_case, causal_case=None, attribute=attr)
        if est_P_dict is None:
            print(f"Error: No estimated Pareto front found in {testresult_filename}.")
            return
        
        # Evaluate each variation
        for variation, est_P_runs in est_P_dict.items():
            gd_list = []
            igd_list = []
            ahd_list = []
            print(f"Evaluating {variation} variation...")

            for run_id, est_P in est_P_runs.items():
                # print(f"Run {run_id} - Estimated Pareto front points: {est_P}")
                est_P = est_P[~np.isnan(est_P).any(axis=1)]
                gd_value = evaluator.gd(est_P)
                igd_value = evaluator.igd(est_P)
                ahd_value = Evaluator.average_hausdorff_distance(gd_value, igd_value, mode='max')

                gd_list.append(gd_value)
                igd_list.append(igd_value)
                ahd_list.append(ahd_value)

            # Summary statistics
            gd_mean, gd_median, gd_std = np.mean(gd_list), np.median(gd_list), np.std(gd_list)
            igd_mean, igd_median, igd_std = np.mean(igd_list), np.median(igd_list), np.std(igd_list)
            ahd_mean, ahd_median, ahd_std = np.mean(ahd_list), np.median(ahd_list), np.std(ahd_list)

            print(f"{variation} - GD: mean={gd_mean:.2f}, median={gd_median:.2f}, std={gd_std:.2f}")
            print(f"{variation} - IGD: mean={igd_mean:.2f}, median={igd_median:.2f}, std={igd_std:.2f}")
            print(f"{variation} - AHD: mean={ahd_mean:.2f}, median={ahd_median:.2f}, std={ahd_std:.2f}")

            results.append({
                'variation': variation,
                'GD_mean': gd_mean,
                'GD_median': gd_median,
                'GD_std': gd_std,
                'IGD_mean': igd_mean,
                'IGD_median': igd_median,
                'IGD_std': igd_std,
                'AHD_mean': ahd_mean,
                'AHD_median': ahd_median,
                'AHD_std': ahd_std
            })

        # Save results
        results_df = pd.DataFrame(results)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'eval_{dataset}_{attr}_{use_case}_{semantic_metric}_{timestamp}.csv')
        results_df.to_csv(output_path, index=False)
        print(f"Evaluation results saved to: {output_path}")

def use_backup_get_estimated_P(dataset:str,use_case:str='modeling', semantic_metric:str='gpt_semantics'):
    est_P_dict=defaultdict(dict)
    variations= ['zero_shot', 'few_shot', 'rag', 'few_shot_rag', 'cot', 'cot_few_shot', 'cot_rag', 'cot_few_shot_rag']
    for variation in variations:
        est_P = pd.read_csv(os.path.join(output_dir, f'{dataset}_{use_case}_{semantic_metric}_{variation}_backup.csv'))
        for run_id in est_P['run_id'].unique():
            est_P_run = est_P[est_P['run_id'] == run_id]
            est_P_dict[variation][run_id] = np.array([est_P_run['semantic'].values, est_P_run['utility'].values]).T
        # est_P_dict[variation] = np.array([est_P['semantic'].values, est_P['utility'].values]).T
    return est_P_dict

def main():
    generator = Generator(model="gpt-4o-mini", verbose=True)
    #prompt = "What is the capital of France?"
    dataset_name = "titanic"
    prompt = QUESTION_PROMPT.format(
        file_name=f"{dataset_name}.csv",
        table_snippet=get_table_markdown(os.path.join(project_root, f"data/{dataset_name}/input/{dataset_name}.csv"), num_rows=10),
    )
    response = generator.generate(prompt)
    print("Response:", response)
    
if __name__ == "__main__":
    start_time = time.time()
    print("Starting evaluation of LLM binning...",start_time)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'pima' or 'titanic')")
    parser.add_argument("--use_case", type=str, default="visualization",
                        help="Use case: modeling, imputation, visualization, or causal_inference")
    parser.add_argument("--semantic_metric", type=str, default="gpt_semantics",
                        help="Semantic metric: gpt_semantics, KLDiv, l2_norm")
    parser.add_argument("--imputer", type=str, default="KNN", help="Imputer method: KNN, Simple, Iterative")
    parser.add_argument("--missing_frac", type=float, default=0.3, help="Fraction of missing data to simulate")
    parser.add_argument("--testresult_filename", type=str, required=True,
                        help="CSV filename in the testresults/LLM directory")
    parser.add_argument("--use_backup", action="store_true", help="Enable use_backup=True")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early_stopping=True")
    parser.add_argument("--causal_use_case", type=str, default=None,required=False,choices=['known', 'unknown'])

    args = parser.parse_args()

    # testresult_dir = os.path.join(project_root, 'testresults', 'LLM')
    # full_path = os.path.join(testresult_dir, args.testresult_filename)

    if args.use_case == 'visualization':
        evaluate_llm_binning_viz(
            args.testresult_filename,
            args.dataset,
            args.semantic_metric,
            args.use_case,
            use_backup=args.use_backup
        )
    else:
        evaluate_llm_binning(
            args.testresult_filename,
            args.dataset,
            args.semantic_metric,
            args.use_case,
            args.imputer,
            args.missing_frac,
            use_backup=args.use_backup,
            early_stopping=args.early_stopping,
            causal_case=args.causal_use_case
        )
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n✅ Total runtime: {duration:.2f} seconds")
