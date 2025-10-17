import random

from dowhy import CausalModel
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
import seaborn as sns
from .discretizers import *
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, f1_score
import re

LOWER_BOUND=0
UPPER_BOUND=1
MEAN=0
STD=1
SAMPLE_SIZE=5
DATA_PATH = "../../data/causal/"
# f1_score_fie_path = os.path.join(DATA_PATH, "f1_scores_epsilon_data2_spearman.csv")
def main_causal(data, attributes,treatment, outcome,DAG,ATE_status,saved_results,multi_method=None,corr_method='spearman'):

    if ATE_status=="known":
        real_ATE=compute_causal_effect(data, treatment, outcome, DAG)
        print("real_ATE",real_ATE)
        if len(attributes)==1:
            attribute = attributes[0]
            apply_single_attribute_pipline(data, treatment, outcome, attribute, real_ATE,DAG)

        elif len(attributes)>1:
            if not saved_results:
                print("[DEBUG] No saved results provided, applying single attribute pipelines for each attribute.")
                saved_results=[]
                for attribute in attributes:
                    apply_single_attribute_pipline(data, treatment, outcome, attribute, real_ATE, DAG)
                    saved_results.append(f"binned_data_{attribute}_utility.csv")
            apply_multiple_attributes_pipline(data, treatment, outcome, attributes, real_ATE,saved_results,DAG, multi_method)


    elif ATE_status=="unknown":
        real_ATE=compute_causal_effect(data, treatment, outcome, DAG)
        print("real_ATE",real_ATE)
        utility_df = pd.read_csv(os.path.join(DATA_PATH, "binned_data_multi_attributes_utility_all.csv"))
        utility_df['method'] = utility_df['method'].apply(
            lambda x: eval(x)[0] + "_" + eval(x)[1] if isinstance(x, str) else x[0] + "_" + x[1])
        print(f"utility shape: {utility_df.shape}")
        results = []
        epsilons_to_try = [0.001, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1,"Quartiles","Deciles","Thirds","Fifths","Traditional"]
        for epsilon in epsilons_to_try[-5:]:
            print(f"Processing epsilon: {epsilon}")
            lower_tau, higher_tau,unbinned_correlations= warper_get_thresholds(data,corr_method,epsilon)
            all_correlations= wrapper_pairwise_correlations(data,corr_method=corr_method)
            # all_correlations.to_csv(os.path.join(DATA_PATH, f"all_binned_correlations_{corr_method}.csv"), index=False)
            # all_correlations = pd.read_csv(os.path.join(DATA_PATH,f"all_binned_correlations_{corr_method}.csv"))
            # print(f"all_correlations shape: {all_correlations.shape}")
            F1_df=F1_wrapper(unbinned_correlations, all_correlations, lower_tau, higher_tau)
            # F1_df.to_csv(f1_score_fie_path, index=False)
            # F1_df= pd.read_csv(f1_score_fie_path)
            # refutation_scores_df = wrapper_pairwise_refutation(data, treatment, outcome, DAG,real_ATE)
            # refutation_scores_df.to_csv(os.path.join(DATA_PATH, "refutation_scores.csv"), index=False)
            F1_df['method'] = F1_df['method'].apply(fix_bayesianblocks_method) #TODO hotfix,adding n_bins to BayesianBlocksDiscretizer
            F1_utility_relation_grpah(F1_df, utility_df[['method', 'utility']],epsilon)



def data_causal_score(data, outcome, treatment, confounders, ATE_status, unbinned_pre_score):
    """
    Calculate the causal score for the given data and attributes.

    Parameters:
    - data: DataFrame containing the dataset after binning.
    - outcome: the outcome coulmn name.
    - ATE_status: Use case for the causal score calculation (known or unknown ATE).
    - treatment: The treatment column name.
    - confounders: List of confounder attributes.

    Returns:
    - utility: Calculated utility score.
    """
    if ATE_status=="known":
        ATE=compute_causal_effect(data, treatment, outcome, confounders=confounders)
        utility=np.exp(-np.abs(unbinned_pre_score - ATE))
    else: # ATE_status=="unknown":
        lower_tau, higher_tau,unbinned_correlations = unbinned_pre_score
        binned_corrs = compute_pairwise_correlations(data,data.columns,binning_name1="binned", binning_name2="binned")
        utility=compute_f1_preservation(unbinned_correlations, binned_corrs, lower_tau, higher_tau)
    return utility



def warper_get_thresholds(data,corr_method='spearman',epsilon:Union[str, int] = "Thirds"):
    unbinned_correlations = compute_pairwise_correlations(data, data.columns, binning_name1="unbinned",
                                                          binning_name2="unbinned", corr_method=corr_method)
    # pd.DataFrame(unbinned_correlations).to_csv(
    #     os.path.join(DATA_PATH, f"unbinned_correlations_data2_{corr_method}.csv"))
    # unbinned_correlations = pd.read_csv(os.path.join(DATA_PATH, f"unbinned_correlations_data2_{corr_method}.csv"))
    lower_tau, higher_tau = get_thresholds(unbinned_correlations,epsilon)
    return lower_tau, higher_tau,unbinned_correlations

def refute_dowhy_causal(data, treatment, outcome, DAG,real_ATE):
    """
    Refute the causal effect using DoWhy's refutation methods.

    Parameters:
    - data: DataFrame containing the data
    - treatment: Name of the treatment variable
    - outcome: Name of the outcome variable
    - DAG: List of strings representing the causal graph

    Returns:
    - None
    """
    causal_graph = """
                               digraph {
                               """
    for line in DAG:
        causal_graph = causal_graph + line + "\n"
    causal_graph = causal_graph + "}"
    model = CausalModel(
        data=data,
        graph=causal_graph.replace("\n", " "),
        treatment=treatment,
        outcome=outcome)

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    causal_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

    # Apply multiple refuters
    refuters = [
        "random_common_cause",
        "placebo_treatment_refuter",
        "data_subset_refuter",
        "dummy_outcome_refuter",
        # "add_unobserved_common_cause",
        "bootstrap_refuter"
    ]

    refutations = []
    for ref in refuters:
        refutation = model.refute_estimate(identified_estimand, causal_estimate, method_name=ref)
        # print(f"\nRefutation: {ref}")
        if ref=="dummy_outcome_refuter":
            refutation=refutation[0]
        # print(refutation)
        refutations.append(refutation)

    pass_rate = compute_pass_rate(refutations, real_ATE, epsilon=0.05, pval_thresh=0.05)
    # print(f"pass_rate: {pass_rate}")
    return pass_rate

def wrapper_pairwise_refutation(df,treatment, outcome, DAG,real_ATE):
    binnings_C1 = pd.read_csv(os.path.join(DATA_PATH, "binned_data_C1.csv"))
    binnings_C2 = pd.read_csv(os.path.join(DATA_PATH, "binned_data_C2.csv"))

    binning_names_C1 = binnings_C1.columns
    binning_names_C2 = binnings_C2.columns
    selected_binnings_C1 = random.sample(list(binning_names_C1), 15)
    selected_binnings_C2 = random.sample(list(binning_names_C2), 15)
    results = []
    for i,bin_C1 in enumerate(selected_binnings_C1):
        for bin_C2 in tqdm(selected_binnings_C2):
            df_binned = df.copy()
            df_binned['C1'] = binnings_C1[bin_C1]
            df_binned['C2'] = binnings_C2[bin_C2]
            # print(f"Calculating refute score for binned C1: {bin_C1}, C2: {bin_C2}")
            pairwise_score = refute_dowhy_causal(df_binned,treatment, outcome, DAG,real_ATE)
            results.append({ 'C': f"{bin_C1},{bin_C2}",
                'attr1_binning': bin_C1,
                'attr2_binning': bin_C2,
                'refutation_score': pairwise_score
            })
        pd.DataFrame(results).to_csv(os.path.join(DATA_PATH, "pairwise_refutation_backup.csv"),
                                         index=False)
    all_refutations = pd.DataFrame(results)
    return all_refutations


def compute_pass_rate(refutations, original_effect, epsilon=0.05, pval_thresh=0.05):
    """
    Compute fraction of refutation tests passed based on pass-if-either rule.

    Args:
        refutations: list of dicts with 'method', 'new_effect', 'p_value'
        original_effect: float
        epsilon: max allowed ATE change
        pval_thresh: min acceptable p-value

    Returns:
        float: fraction of refutations passed (0 to 1)
    """
    passed = 0
    failed_tests = []

    for ref in refutations:
        method= ref.refutation_type
        new_effect = ref.new_effect
        pval = ref.refutation_result
        pval = pval.get('p_value', None) if isinstance(pval, dict) else None

        if isinstance(new_effect, tuple):
            low, high = new_effect
            if low <= original_effect <= high:
                passed += 1
            else:
                failed_tests.append({
                    'method': method,
                    'reason': f'Original ATE not in interval [{low:.3f}, {high:.3f}]',
                    'new_effect': new_effect,
                    'p_value': None
                })
        else:
            ate_diff = abs(original_effect - new_effect)
            passed_test = False
            reason = []
            if pval is not None:
                if pval > pval_thresh:
                    passed_test = True
                else:
                    reason.append(f'p-value too low ({pval:.3f})')
            if ate_diff < epsilon:
                passed_test = True
            else:
                reason.append(f'ATE change too large (Δ={ate_diff:.3f})')

            if passed_test:
                passed += 1
            else:
                failed_tests.append({
                    'method': method,
                    'reason': ' and '.join(reason),
                    'new_effect': new_effect,
                    'p_value': pval
                })
    # print(f"Passed: {passed}")
    if failed_tests:
        print("Failed tests:")
        for test in failed_tests:
            print(f"  Method: {test['method']}, Reason: {test['reason']}, New Effect: {test['new_effect']}, p-value: {test['p_value']}")

    return passed / len(refutations)

def fix_bayesianblocks_method(method):
    # Replace any instance of BayesianBlocksDiscretizer not followed by _<number>
    return re.sub(r'BayesianBlocksDiscretizer(?!_\d)', 'BayesianBlocksDiscretizer_3', method)

def get_thresholds(correlations_df, epsilon:Union[str, int] = "Quartiles"):
    """
    Get the lower and higher thresholds for corr scores based on percentiles.

    Parameters:
    - correlations_df: DataFrame containing corr scores on unbinned data

    Returns:
    - tau_low: lower threshold for "low" correlation
    - tau_high: higher threshold for "high" correlation
    """
    # epsilon= 0.01
    corr_values = correlations_df['correlation']
    epsilon_mapping={"Quartiles": [25, 75],
    "Deciles": [10, 90],
    "Thirds": [33.3, 66.7],
                     "Fifths": [20, 80],"Traditional":[30,70]}
    if isinstance(epsilon, str):
        if epsilon not in epsilon_mapping:
            raise ValueError(f"Invalid epsilon value: {epsilon}. Choose from {list(epsilon_mapping.keys())}.")
        tau_low =  np.percentile(corr_values, epsilon_mapping[epsilon][LOWER_BOUND])
        tau_high =  np.percentile(corr_values, epsilon_mapping[epsilon][UPPER_BOUND])
    else:
        tau_low = corr_values.min()+epsilon
        tau_high = corr_values.max()-epsilon
    print(f"Lower tau threshold: {tau_low}, Higher tau threshold: {tau_high}")

    return tau_low, tau_high

def F1_utility_relation_grpah(F1_scores_df, utility_df,epsilon:Union[str, int]):
    """
    Create a graph showing the relationship between F1 scores and utility.

    Parameters:
    - F1_scores_df: DataFrame containing F1 scores for different methods
    - utility_df: DataFrame containing utility values for different methods

    Returns:
    - None
    """
    merged_df = F1_scores_df.merge(utility_df, on='method', how='inner')
    x = merged_df['utility']
    y = merged_df['f1_score']
    pearson_r, pval = pearsonr(x, y)
    r_squared = pearson_r ** 2
    print(f"📈 Pearson correlation (r): {pearson_r:.4f}")
    print(f"📊 R²: {r_squared:.4f}")
    print(f"🧪 p-value: {pval:.4g}")
    if pval < 0.05:
        print("✅ The correlation is statistically significant (p < 0.05).")
    else:
        print("⚠️ The correlation is not statistically significant.")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged_df, x='utility', y='f1_score', hue='method', style='method')
    plt.title('F1 Score vs Utility for espsilon = {}'.format(epsilon))
    plt.xlabel('Utility')
    plt.ylabel('F1 Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def F1_wrapper(unbinned_correlations,all_correlations,tau_low,tau_high):
    all_correlations['method'] = all_correlations['attr1_binning'] + "_" + all_correlations['attr2_binning']
    all_methods=all_correlations['method'].unique()
    results=[]

    for method in tqdm(all_methods):
        after_df = all_correlations[all_correlations['method'] == method]
        f1_score = compute_f1_preservation(unbinned_correlations, after_df, tau_low, tau_high)
        # print(f"F1 score for {method}: {f1_score}")
        results.append({"method":method, "f1_score":f1_score})
    results_df = pd.DataFrame(results)
    return results_df



def compute_f1_preservation(before_df, after_df, tau_low, tau_high):
    """
    Compute a single F1 score for correlation preservation based on strict TP, TN, FP, FN definitions.

    Definitions:
    - TP: high before AND high after
    - TN: low before AND low after
    - FP: low before BUT high after
    - FN: high before BUT low after

    Parameters:
    - before_df: DataFrame with 'C' and 'correlation' (before binning)
    - after_df:  DataFrame with 'C' and 'correlation' (after binning)
    - tau_low: threshold for "low" correlation
    - tau_high: threshold for "high" correlation

    Returns:
    - f1: single F1 score for preservation of strong/weak correlations
    """

    # Merge on pair name
    merged = before_df[['C', 'correlation']].merge(
        after_df[['C', 'correlation']], on='C', suffixes=('_before', '_after'))

    # Label only low/high
    def label(corr):
        if corr >= tau_high:
            return 1 #'high'
        elif corr <= tau_low:
            return 0 #'low'
        else:
            return -1 #'mid'

    merged['label_before'] = merged['correlation_before'].apply(label)
    merged['label_after'] = merged['correlation_after'].apply(label)

    # # Keep only those that are either low or high before
    # filtered = merged[merged['label_before'].isin(['low', 'high']) & merged['label_after'].isin(['low', 'high'])]
    #
    # # Define binary classes: 1 for "high", 0 for "low"
    # y_true = (filtered['label_before'] == 'high').astype(int).to_list()
    # y_pred = (filtered['label_after'] == 'high').astype(int).to_list()
    # Only consider items that were 'high' or 'low' BEFORE (regardless of after)
    # relevant = merged[merged['label_before'].isin(['low', 'high'])]
    y_true=merged['label_before'].to_list()
    y_pred=merged['label_after'].to_list()
    # y_true = merged['label_before'].map({'low': 0, 'high': 1}).to_list()
    # # Predict '1' only if label_after is 'high'; '0' if 'low'; otherwise mark as 'missing'
    # y_pred = merged['label_after'].map({'low': 0, 'high': 1, 'mid': -1}).to_list()

    # Filter out -1 if you want to only measure predictions that stayed in the binary space
    # final_y_true = [yt for yt, yp in zip(y_true, y_pred) if yp != -1]
    # final_y_pred = [yp for yp in y_pred if yp != -1]
    # print(f"[DEBUG] y_true: {merged['correlation_before'].to_list()}, y_pred: {merged['correlation_after'].to_list()}")  # Debugging line to check values
    # print(f"[DEBUG] y_true: {y_true}, y_pred: {y_pred}")
    f1= f1_score(y_true, y_pred, average='macro', zero_division=0)
    # Calculate precision, recall, and F1 score
    precision= precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall= recall_score(y_true, y_pred, average='macro', zero_division=0)
    # print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")
    return f1

def compute_relevant_pairwise_correlations(df, binned_attr, treatment, outcome, corr_method='spearman'):
    corr_matrix = df.corr(method=corr_method).abs()
    records = []

    # pairs with the binned attribute
    for other in df.columns:
        if other == binned_attr:
            continue
        pair = f"{binned_attr},{other}"
        value = corr_matrix.loc[binned_attr, other]
        records.append({
            'C': pair,
            'correlation': value,
            'attr1': binned_attr,
            'attr2': other
        })
    #X,Y correlation
    if treatment in df.columns and outcome in df.columns:
        pair = f"{treatment},{outcome}"
        value = corr_matrix.loc[treatment, outcome]
        records.append({
            'C': pair,
            'correlation': value,
            'attr1': treatment,
            'attr2': outcome
        })

    return pd.DataFrame(records)

def compute_pairwise_correlations(df, attributes, binning_name1=None, binning_name2=None,corr_method='spearman'):
    records = []
    corr_matrix = df.corr(method=corr_method).abs()

    for attr1, attr2 in combinations(attributes, 2):
        value=corr_matrix.loc[attr1, attr2]
        corr_record = {
        'C': f"{attr1},{attr2}", 'attr1_binning': binning_name1, 'attr2_binning': binning_name2,
        'correlation':value}
        records.append(corr_record)
    return pd.DataFrame(records)

def wrapper_pairwise_correlations(df,corr_method='spearman'):
    binnings_C1 = pd.read_csv(os.path.join(DATA_PATH, "binned_data_C1.csv"))
    binnings_C2 = pd.read_csv(os.path.join(DATA_PATH, "binned_data_C2.csv"))

    binning_names_C1 = binnings_C1.columns
    binning_names_C2 = binnings_C2.columns

    all_correlations = pd.DataFrame()

    for bin_C1 in binning_names_C1:
        for bin_C2 in binning_names_C2:
            df_binned = df.copy()
            df_binned['C1'] = binnings_C1[bin_C1]
            df_binned['C2'] = binnings_C2[bin_C2]
            # print(f"Calculating correlations for binned C1: {bin_C1}, C2: {bin_C2}")
            pairwise_correlations = compute_pairwise_correlations(df_binned, df_binned.columns, binning_name1=bin_C1, binning_name2=bin_C2,corr_method=corr_method)
            all_correlations = pd.concat([all_correlations, pairwise_correlations], ignore_index=True)
    return all_correlations




def heatmap_inner_correlations(df, treatment, attributes):
    binnings_C1 = pd.read_csv(os.path.join(DATA_PATH, "binned_data_C1.csv"))
    binnings_C2 = pd.read_csv(os.path.join(DATA_PATH, "binned_data_C2.csv"))

    binning_names_C1 = binnings_C1.columns
    binning_names_C2 = binnings_C2.columns

    records = []

    # Unbinned correlations for C1 and C2
    records.append({
        'C': 'C1,X', 'bin_C1': "unbinned", 'bin_C2': "unbinned",
        'correlation': df['C1'].corr(df[treatment])
    })
    records.append({
        'C': 'C2,X', 'bin_C1': "unbinned", 'bin_C2': "unbinned",
        'correlation': df['C2'].corr(df[treatment])
    })

    # Binned correlations
    for bin_C1 in binning_names_C1:
        for bin_C2 in binning_names_C2:
            df_binned = df.copy()
            df_binned['C1'] = binnings_C1[bin_C1]
            df_binned['C2'] = binnings_C2[bin_C2]

            records.append({
                'C': 'C1,X', 'bin_C1': bin_C1, 'bin_C2': bin_C2,
                'correlation': df_binned['C1'].corr(df_binned[treatment])
            })
            records.append({
                'C': 'C2,X', 'bin_C1': bin_C1, 'bin_C2': bin_C2,
                'correlation': df_binned['C2'].corr(df_binned[treatment])
            })

    results_df = pd.DataFrame(records)
    print(results_df.groupby('C')['correlation'].describe())

    top_n = 30

    unbinned_rows = results_df[
        (results_df['bin_C1'] == 'unbinned') & (results_df['bin_C2'] == 'unbinned')
    ]

    # Get top-n (non-unbinned) rows by absolute correlation
    top_rows = results_df[
        ~((results_df['bin_C1'] == 'unbinned') & (results_df['bin_C2'] == 'unbinned'))
    ].sort_values(by='correlation', key=lambda x: x.abs(), ascending=False).head(top_n)

    # Combine and reset index
    plot_df = pd.concat([unbinned_rows, top_rows]).reset_index(drop=True)
    print("Feature pairs in heatmap:", plot_df['C'].unique())

    # Pivot for heatmap
    pivot = plot_df.pivot(index='C', columns=['bin_C1', 'bin_C2'], values='correlation')

    # Plot heatmap
    plt.figure(figsize=(len(pivot.columns) * 0.7, 8))
    sns.heatmap(pivot, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation with X for top (C1, C2) combinations (with unbinned)")
    plt.xlabel("(C1 binning, C2 binning)")
    plt.ylabel("Feature Pair")
    plt.tight_layout()
    plt.show()
    # print(f"[DEBUG] Generating heatmap for inner correlations with treatment: {treatment} and attributes: {attributes}")
    #
    # # Calculate the correlation matrix for the treatment and attributes
    # corr_matrix = data[[treatment] + attributes].corr()
    #
    # # Display the heatmap
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title(f"Inner Correlations Heatmap for {treatment} and Attributes")
    # plt.show()

    print("[DEBUG] Heatmap generated.")
# def show_inner_correlations(data, treatment, attributes):
#     print(f"[DEBUG] Showing inner correlations for treatment: {treatment} and attributes: {attributes}")
#
#     for attribute in attributes:
#         corr = data[[treatment, attribute]].corr().iloc[0, 1]
#         print(f"Correlation between {treatment} and {attribute}: {corr:.4f}")
#     display_heat_map(data, attributes, treatment)
#     print("[DEBUG] Inner correlations displayed.")




def apply_multiple_attributes_pipline(data, treatment, outcome, attributes, real_ATE, saved_results, DAG, multi_method):
    print(f"[DEBUG] apply_multiple_attributes_pipline called with attributes={attributes}, multi_method={multi_method}")
    relevant_strategies = sample_rel_stategies(multi_method, saved_results)
    print(f"[DEBUG] relevant_starategies: {relevant_strategies}")
    combo_list = create_multi_attributes_combination(relevant_strategies, len(attributes))
    print(f"[DEBUG] combo_list length: {len(combo_list)}")
    ATE_4_binned_data = apply_binning_strategies_multi(combo_list, data, attributes, treatment, outcome, DAG)
    print(f"[DEBUG] ATE_4_binned_data head:\n{ATE_4_binned_data.head()}")
    binned_data_w_utility = calc_known_utility(ATE_4_binned_data, real_ATE)
    print(f"[DEBUG] binned_data_w_utility head:\n{binned_data_w_utility.head()}")
    binned_data_w_utility.to_csv(os.path.join(DATA_PATH, f"binned_data_multi_attributes_utility_{multi_method}.csv"), index=False)

def sample_rel_stategies(multi_method, saved_results)->List[List[str]]:
    methods=[]
    for res in saved_results:
        file_path = os.path.join(DATA_PATH, res)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {res} does not exist in the expected directory.")
        attribute_results=pd.read_csv(file_path)
        if multi_method == "random":
            options = attribute_results['method'].tolist()
            selected_methods = np.random.choice(options, size=SAMPLE_SIZE, replace=False).tolist()
            methods.append(selected_methods)
        elif multi_method == "top":
            attribute_results.sort_values(by='utility', ascending=False, inplace=True)
            selected_methods=attribute_results.head(SAMPLE_SIZE)['method'].tolist()
            methods.append(selected_methods)
        elif multi_method == "all":
            if len(methods)==0:
                selected_methods = attribute_results['method'].tolist()
                methods.append(selected_methods)
            else:
                methods.append([]) # Placeholder for other attributes since we already have all methods from the first attribute
        else:
            raise ValueError("Unknown multi_method: {}".format(multi_method))
    return methods

def create_multi_attributes_combination(relevant_strategies, num_attributes)->List[tuple]:
    if len(relevant_strategies) != num_attributes:
        raise ValueError(f"Mismatch: {len(relevant_strategies)} strategy lists for {num_attributes} attributes")

        # Merge all strategies and get unique ones
    all_strategies = []
    for strategies in relevant_strategies:
        all_strategies.extend(strategies)

    unique_strategies = list(set(all_strategies))  # Remove duplicates
    print(f"Unique strategies pool: {len(unique_strategies)} strategies from {num_attributes} attributes")

    # Create all combinations using Cartesian product (with repetition and order matters)
    # This creates unique_strategies^num_attributes combinations
    combinations = list(itertools.product(unique_strategies, repeat=num_attributes))

    print(f"Generated {len(combinations)} strategy combinations for {num_attributes} attributes")
    print(f"Expected: {len(unique_strategies)}^{num_attributes} = {len(unique_strategies) ** num_attributes}")
    return combinations




def parse_strategy_name(strategy_name):
    """
    Parse strategy name to extract base name and n_bins.

    Args:
        strategy_name: Strategy name like "Equal width_5" or "KMeans_3"

    Returns:
        tuple: (base_name, n_bins)
    """
    if '_' in strategy_name:
        base_name, n_bins_str = strategy_name.rsplit('_', 1)
        try:
            n_bins = int(n_bins_str)
        except ValueError:
            base_name = strategy_name
            n_bins = None
    else:
        base_name = strategy_name
        n_bins = None

    return base_name, n_bins


def find_discretizer_by_name(base_name, attribute, outcome):
    """
    Find discretizer by base name.

    Args:
        base_name: Base name of the discretizer (e.g., "Equal width")
        attribute: Attribute name
        outcome: Outcome variable name

    Returns:
        dict or None: Discretizer dictionary if found, None otherwise
        looks like- {"name": "Equal width", "func": equal_width, "args": lambda df, n_bins: (df, n_bins, attrs), "kwargs": {}},
    """
    print(f"[DEBUG] Looking for discretizer: {base_name} for attribute: {attribute}")
    available_discretizers = get_potential_binning_strategies(attribute, outcome)
    for disc in available_discretizers:
        if disc["name"] == base_name:
            return disc
    return None


def apply_discretizer_to_attribute(df, attribute, discretizer, n_bins):
    """
    Apply a single discretizer to a single attribute.

    Args:
        df: DataFrame to modify (modified in place)
        attribute: Attribute name to bin
        discretizer: Discretizer dictionary
        n_bins: Number of bins

    Returns:
        list: Bins used for binning
    """
    print(f"[DEBUG] Applying discretizer to attribute: {attribute}")
    copy_df= df.copy()
    func = discretizer["func"]
    args_func = discretizer.get("args")
    kwargs = discretizer.get("kwargs", {})

    args = args_func(copy_df, n_bins)
    intervals = func(*args, **kwargs)

    # Get bins and apply them
    print(f"[DEBUG] intervals keys: {list(intervals.keys())}, expected attribute: {attribute}")
    bins = intervals[attribute]
    min_value = copy_df[attribute].min()
    max_value = copy_df[attribute].max()
    bins = [min_value] + [b for b in bins if b > min_value] + ([max_value] if max_value > bins[-1] else [])

    # Apply binning to the dataframe
    print(f"[DEBUG] Binning applied to {attribute}: {bins}")

    return bins


def apply_combo_binning(df,combo_strategies,attributes,outcome):
    """
    Custom function to apply a combination of strategies to multiple attributes.
    This function signature matches what's expected by the discretizer framework.

    Args:
        df: DataFrame to process
        combo_strategies: Tuple of strategy names for each attribute
        attributes: List of attribute names

    Returns:
        Dictionary with binned results for each attribute (to match discretizer interface)
    """


    print(f"[DEBUG] Applying combo strategy: {combo_strategies} to attributes: {attributes}")

    # Apply each strategy to its corresponding attribute
    for i, attribute in enumerate(attributes):
        strategy_name = combo_strategies[i]
        print(f"[DEBUG] i={i}, attribute={attribute}, strategy_name={strategy_name}")

        # Parse strategy name to get base name and n_bins
        base_name, strategy_n_bins = parse_strategy_name(strategy_name)

        print(f"[DEBUG] Parsed base_name: {base_name}, strategy_n_bins: {strategy_n_bins}")

        # Find the discretizer for this strategy
        discretizer = find_discretizer_by_name(base_name, attribute, outcome)
        if discretizer is None:
            raise ValueError(f"Discretizer '{base_name}' not found for attribute '{attribute}'")

        # Apply the discretizer to this attribute
        bins=apply_discretizer_to_attribute(df, attribute, discretizer, strategy_n_bins)
        if base_name=="BayesianBlocksDiscretizer" :
            print(f"[DEBUG] BayesianBlocksDiscretizer so no need to check bins length")
        else:
            print(
            f"[DEBUG] expected bins of size {strategy_n_bins + 1}, got bins: {len(bins)}, the size matches the expected n_bins: {len(bins) == strategy_n_bins + 1}")

        df[attribute] = pd.cut(df[attribute], bins=bins, labels=False, include_lowest=True)

    return df


def apply_binning_strategies_multi(discretizer_list, df, attributes, treatment, outcome, DAG):
    """
    Apply binning strategies using the standard discretizer_list format (adapted for multi-attribute).
    This matches the apply_binning_strategies function signature.
    """
    print("Applying multi-attribute binning strategies...")
    # Sort the DataFrame by the first attribute
    # df = df.sort_values(by=attributes[0])
    results = []

    for i,combo_strategies in enumerate(tqdm(discretizer_list)):
        try:
            # For combo strategies, we don't iterate over n_bins since each strategy has its own
            copy_df = df.copy()


            copy_df=apply_combo_binning(copy_df,combo_strategies,attributes,outcome)

            # Compute causal effect
            causal_estimate_reg = compute_causal_effect(copy_df, treatment, outcome, DAG)
            print("Causal effect estimate:", causal_estimate_reg)

            results.append({
                'method': combo_strategies,
                'ATE': causal_estimate_reg
            })
            if i % 100 == 0:  # save every 100 iterations
                pd.DataFrame(results).to_csv(os.path.join(DATA_PATH, "binned_data_multi_attributes_backup.csv"), index=False)
        except Exception as e:
            print(f"Countered error in {combo_strategies}: {e}")

    return pd.DataFrame(results)

def calc_known_utility(binned_data, real_ATE)->pd.DataFrame:
    "exp(-abs(real_ATE-ATE))"
    binned_data['utility'] = np.exp(-np.abs(real_ATE - binned_data['ATE']))
    return binned_data
def apply_single_attribute_pipline(data:pd.DataFrame, treatment:str, outcome:str, attribute:str, real_ATE:float,DAG:list):
    potential_binning_startegies = get_potential_binning_strategies(attribute, outcome)
    ATE_4_binned_data = apply_binning_strategies(potential_binning_startegies, data, attribute, treatment, outcome,DAG)
    binned_data_w_utility = calc_known_utility(ATE_4_binned_data, real_ATE)
    binned_data_w_utility.to_csv(os.path.join(DATA_PATH, f"binned_data_{attribute}_utility.csv"), index=False)

def compute_causal_effect(data, treatment, outcome, DAG=None,confounders=None):
    if DAG:
        causal_graph = """
                                   digraph {
                                   """
        for line in DAG:
            causal_graph = causal_graph + line + "\n"
        causal_graph = causal_graph + "}"
        model = CausalModel(
            data=data,
            graph=causal_graph.replace("\n", " "),
            treatment=treatment,
            outcome=outcome)
    elif confounders:
        # If no DAG is provided, use confounders to create a backdoor adjustment
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders
        )
    else:
        raise ValueError("Either DAG or confounders must be provided for causal effect computation.")

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    causal_estimate_reg = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True
    )
    return causal_estimate_reg.value
def get_potential_binning_strategies(attrs, outcome):
    if type(attrs)==str: # If a single attribute is passed, convert it to a list
        attrs = [attrs]
    discretizers = [
        {"name": "Equal width", "func": equal_width, "args": lambda df, n_bins: (df, n_bins, attrs), "kwargs": {}},
        {"name": "Equal frequency", "func": equal_frequency, "args": lambda df, n_bins: (df, n_bins, attrs), "kwargs": {}},
        {"name": "ChiMerge", "func": chimerge_wrap, "args": lambda df, n_bins: (df, attrs, outcome, n_bins), "kwargs": {}},
        {"name": "KBinsDiscretizer", "func": KBinsDiscretizer_wrap, "args": lambda df, n_bins: (df, attrs, n_bins), "kwargs": {}},
        {"name": "KBinsDiscretizer (quantile)", "func": KBinsDiscretizer_wrap, "args": lambda df, n_bins: (df, attrs, n_bins), "kwargs": {"strategy": "quantile"}},
        {"name": "KMeansDiscretizer", "func": KMeansDiscretizer_wrap, "args": lambda df, n_bins: (df, attrs, n_bins), "kwargs": {}, "handle_error": True},
        {"name": "BayesianBlocksDiscretizer", "func": BayesianBlocksDiscretizer_wrap, "args": lambda df, n_bins: (df, attrs), "kwargs": {}, "handle_error": True},
    ]
    return discretizers



def apply_binning_strategies(discretizer_list, df, attribute, treatment, outcome, DAG) -> pd.DataFrame:
    print("Applying binning strategies...")
    # Sort the DataFrame by the first attribute
    original_index = df.index
    df = df.sort_values(by=attribute)
    # binned_data_df = pd.DataFrame(index=df.index)
    # Iterate through each discretizer
    results = []

    for disc in tqdm(discretizer_list):
        name = disc["name"]
        print("Discretizer:", name)
        handle_error = disc.get("handle_error", False)
        try:
            for n_bins in range(3, 11):
                copy_df = df.copy()

                curr_name = f"{name}_{n_bins}"
                if name == "BayesianBlocksDiscretizer" :
                    if n_bins > 3:
                        print(f"Skipping {name} for n_bins={n_bins} (It calculate its own bins)")
                        continue
                    # curr_name=name
                # Use the refactored helper function
                bins = apply_discretizer_to_attribute(copy_df, attribute, disc, n_bins)

                print("______________________________________________________________________")
                print(f"{curr_name}:", {attribute: bins})
                copy_df[attribute] = pd.cut(copy_df[attribute], bins=bins, labels=False, include_lowest=True)
                # binned_data_df[curr_name]= copy_df[attribute]#TODO remove this line
                causal_estimate_reg = compute_causal_effect(copy_df, treatment, outcome, DAG)
                print("Causal effect estimate:", causal_estimate_reg)
                results.append({'method': curr_name, 'ATE': causal_estimate_reg})

        except Exception as e:
            if handle_error:
                print(f"Handled error in {name}: {e}")
            else:
                raise
    # Save the binned data with utility
    # binned_data_df = binned_data_df.loc[original_index]  #TODO remove this line
    # binned_data_df.to_csv(os.path.join(DATA_PATH, f"binned_data_{attribute}.csv"),index=False)#TODO remove this line
    # sys.exit("Terminating program due to finished binning strategies application.") #TODO remove this line
    results_data = pd.DataFrame(results)
    return results_data



def generate_dag_data(Y_effects,X_effects,n=1000, seed=42, force_x=None, two_mediators=False, confounders_range=(0, 100), C2_args=(50,15),M1_args=(50,10), Y_noise_args=(0, 5)):
    np.random.seed(seed)

    # Generate confounders (C1, C2)
    C1 = np.random.uniform(confounders_range[0], confounders_range[1], n)  # Uniform distribution
    C2 = np.random.normal(C2_args[MEAN], C2_args[STD], n)   # Normal distribution, mean=50, std=15
    # C2 = 0.5 * C1 + np.random.normal(25, 10, n) #a case where there is an edge between the confounters.

    C2 = np.clip(C2, confounders_range[0], confounders_range[1])            # Keep values in range

    # Treatment (X) influenced by confounders
    if force_x is None:
      # Treatment (X) influenced by confounders
      X_prob = 1 / (1 + np.exp(-(X_effects["C1"] * C1 +X_effects["C2"] * C2)))  # Logistic function
      X = np.random.binomial(1, X_prob, n)  # Binary treatment
    else:
      X = np.full(n, force_x)  # Force X to 1 or 0 for counterfactual estimation

    # Generate mediators (M1, M2) influenced by treatment
    M1 = X * X_effects["M1"] + np.random.normal(M1_args[MEAN], M1_args[STD], n)  # Effect of X + noise


    # Define Outcome (Y) in a **Linear Way** for clarity
    Y = (Y_effects["X"] * X) + (Y_effects["M1"] * M1) + (Y_effects["C1"] * C1) + (Y_effects["C2"] * C2) + np.random.normal(Y_noise_args[MEAN], Y_noise_args[STD], n)
    if two_mediators:
      M2 = X * 5 + np.random.normal(50, 10, n)  # Effect of X + noise
      Y+= (0.02 * M2)

    # Create DataFrame
    df = pd.DataFrame({'C1': C1, 'C2': C2, 'X': X, 'M1': M1, 'Y': Y})
    if two_mediators:
      df['M2']=M2
    return df


def run_data_generation():

    generated_dag_data= generate_dag_data(
    n=50000,
    seed=42,
    X_effects={
        "C1": 0.08,   # C1 → X (medium effect)
        "C2": -0.05,  # C2 → X (mild reverse influence)
        "M1": 20      # X → M1 (strong influence)
    },
    Y_effects={
        "X": 1.2,     # X → Y (strong)
        "M1": 0.1,    # M1 → Y (medium)
        "C1": -0.05,  # C1 → Y (mild)
        "C2": 0.05    # C2 → Y (mild)
    },
    C2_args=(50, 15),          # Keep the default normal distribution
    M1_args=(50, 10),          # Mean and std for M1 noise
    Y_noise_args=(0, 2),       # Lower noise for cleaner signals
    force_x=None,
    two_mediators=False,        # Adds M2 with a small contribution
    confounders_range=(0, 100)
)
    #     Y_effects={"M1": 0.03,"X":0.2,"C1":-0.04,"C2":0.05}
    #     X_effects={"C1": 0.05, "C2": -0.03, "M1": 10}
    #     generated_dag_data = generate_dag_data(n=50000, X_Y_effect=0.2, force_x=None, two_mediators=False)

    generated_dag_data.to_csv("../../data/causal/generated_causal_data2.csv", index=False)
if __name__ == "__main__":

    # run_data_generation()
    data= pd.read_csv("../../data/causal/data1/generated_causal_data.csv")
    # df = pd.read_csv(os.path.join(ppath, 'data', 'causal', 'synthetic_data.csv'))

    treatment = "X"
    outcome = "Y"
    attributes=["C1","C2"]  # List of attributes to be binned
    DAG = [
        "C1 -> X",
        "C2 -> X",
        "C1 -> Y",
        "C2 -> Y",
        "X -> M1",
        "M1 -> Y",
        "X -> Y",
    ]
    ATE_status= "unknown"  # Options:"known", "unknown"
    saved_results= []#["binned_data_C1_utility.csv","binned_data_C2_utility.csv"]  # List of saved results files
    multi_method= "all"  # Options: "random", "top", "all"

    main_causal(data, attributes, treatment, outcome,DAG, ATE_status=ATE_status, saved_results=saved_results,multi_method=multi_method)
    # refute_dowhy_causal(data, treatment, outcome, DAG)