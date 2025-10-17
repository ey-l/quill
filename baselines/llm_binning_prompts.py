QUESTION_PROMPT = """
You are a helpful assistant that generates natural language explanations for every column in the provided data file.
Data file names: {file_name}

The following is a snippet of the data files:
{table_snippet}

Now think step-by-step carefully. 
"""

SYSTEM_PROMPT_GPT_SEMANTICS = (
        "You are a data scientist specializing in feature engineering and statistical learning. "
        "Given a dataset, an outcome feature, and numerical attributes, you design intelligent discretization strategies "
        "that improve the performance of ML tasks like prediction, imputation, visualization, or causal inference, "
        "while also enhancing semantic interpretability."
        "Each binning must maximize the utility of the given task and provide meaningful, domain-aligned groupings. "
        "You will return multiple candidate binnings that form a Pareto frontier over task utility and semantic clarity. "
        "Semantic value is evaluated based on how commonly the grouping is used with that feature in expert sources, "
        "with one of four levels:\n"
        "1 - Not used at all\n"
        "2 - Has seen very few references\n"
        "3 - Rare but used\n"
        "4 - Very commonly used"
    )

SYSTEM_PROMPT_L2_NORM = (
        "You are a data scientist specializing in feature engineering and statistical learning. "
        "Given a dataset, an outcome feature, and numerical attributes, you design intelligent discretization strategies "
        "that improve the performance of ML tasks like prediction, imputation, visualization, or causal inference, "
        "while also enhancing semantic interpretability."
        "Each binning must maximize the utility of the given task and provide meaningful, domain-aligned groupings. "
        "You will return multiple candidate binnings that form a Pareto frontier over task utility and semantic clarity. "
        "Semantic value is evaluated based on the L2 distance between a candidate binning and a given gold-standard binning,"
        "output your semantic score as a float between 0 and 1, where 1 means identical to the gold-standard."
    )

def generate_discretization_prompt(data_description: str,
                                   attributes: list,
                                   outcome: str,
                                   task: str,
                                   treatment: str,
                                   include_chain_of_thought: bool = False,
                                   few_shot_example: str = None,
                                   retrieved_info: str = None,
                                   system_prompt: str = SYSTEM_PROMPT_GPT_SEMANTICS):
    """
    Generate a prompt for LLM-based discretization suggestion.

    Parameters:
    - data_description: a short description of the dataset (can include schema, stats, etc.)
    - attributes: list of numerical attributes to discretize
    - outcome: name of the outcome variable
    - task: one of ['prediction', 'visualization', 'causal inference', 'data imputation']
    - include_chain_of_thought: whether to instruct the model to reason step-by-step
    - few_shot_example: optional string with example(s) to add before the actual request
    - retrieved_info: optional domain knowledge or relevant documents for RAG

    Returns:
    - full prompt string
    """

    # Few-shot examples (optional)
    few_shot_block = f"\n\n### Example:\n{few_shot_example}" if few_shot_example else ""

    # Retrieved info block (optional)
    rag_block = f"\n\n### Domain Knowledge:\n{retrieved_info}" if retrieved_info else ""

    # Chain-of-thought instruction
    cot_instruction = "\n\nPlease think step by step before choosing bins for each attribute." if include_chain_of_thought else ""

    # Final user prompt
    user_prompt = f"""
    You are given the following dataset:

    {data_description}

    Attributes to discretize: {', '.join(attributes)}
    Outcome variable: {outcome}
    Task: {task}
    {'Treatment variable: ' + treatment if treatment else ''}
    {'Note: For causal inference, the remaining attributes are confounders and should be discretized to help balance the treatment and outcome. '
     'You are also asked to discretize the treatment variable, but remember that the goal is to enable reliable ATE estimation by properly balancing the confounders.'
    if task == 'causal inference' else ''}
    {'Note: For imputation, 30% of values are randomly masked across all attributes. The imputation model reconstructs the masked values using correlations and patterns in the other features. Design discretizations that preserve predictive relationships between variables.' if task == 'data imputation' else ''}

    Your objective is to suggest a **discretization for each attribute** that maximizes:
    1. Utility for the task (e.g., predictive accuracy, causal ATE, imputation quality, or visualization informativeness),
    2. Semantic interpretability (i.e., alignment with common, intuitive, or domain-specific bins).

    You are allowed to:
    - Apply **different discretization methods** per attribute.
    - Choose **different number of bins** per attribute.
    - Use **statistical, supervised, unsupervised, or knowledge-based** approaches.
    - Leverage **domain knowledge**, if available.

    Return a list of **Pareto-optimal candidate partitions**, each with a tradeoff between utility and interpretability.

    {cot_instruction}
    {few_shot_block}
    {rag_block}

    ### Output format:
    ```json
    {{
      "0": {{
        "estimated_P": [[semantic_score, utility_score], ...],
        "estimated_partitions": [
          {{
            "Attribute1": [cut1, cut2, ..., cutN],
            "Attribute2": [...],
            "__semantic__": float (1 to 4),
            "__utility__": float (0-1)
          }},
          ...
        ]
      }}
    }}
"""

    return {
        "system": system_prompt,
        "user": user_prompt.strip()
    }

def get_rel_few_shot(task):
    """
    Returns a few-shot example for the given task.
    """

    # Base dataset description (same for all tasks)
    base_dataset = """Dataset: Adult income dataset with 32,561 records
Attributes to discretize: age (17-90), hours_per_week (1-99), capital_gain (0-99999)
Outcome variable: income_class (>50K, <=50K)"""

    if task == "prediction":
        return f"""{base_dataset}
Task: prediction

For prediction accuracy, I need to find splits that maximize information gain and separate the outcome classes effectively.

Age: Standard life stages are semantically meaningful and align with income patterns
Hours_per_week: Work intensity categories that employers and researchers commonly use
Capital_gain: Financial thresholds that differentiate investment behaviors

```json
{{
        "0": {{
        "estimated_P": [[4.0, 0.85], [3.5, 0.89], [2.0, 0.92]],
    "estimated_partitions": [
      {{
        "age": [25, 45, 65],
        "hours_per_week": [32, 40],
        "capital_gain": [0],
        "__semantic__": 4.0,
        "__utility__": 0.85
      }},
      {{
        "age": [30, 50],
        "hours_per_week": [20, 40, 50],
        "capital_gain": [1000, 5000],
        "__semantic__": 3.5,
        "__utility__": 0.89
      }},
      {{
        "age": [23, 28, 35, 42, 55, 67],
        "hours_per_week": [15, 25, 35, 42, 48],
        "capital_gain": [500, 2500, 7500],
        "__semantic__": 2.0,
        "__utility__": 0.92
      }}
      ]
  }}
}}
```"""

    elif task == "visualization":
        return f"""{base_dataset}
Task: visualization

For visualization interestingness, I need bins that reveal clear patterns in income distribution and are easy to interpret in charts.

Age: Life stage categories that show distinct income patterns across age groups
Hours_per_week: Work intensity levels that create meaningful employment categories  
Capital_gain: Investment activity levels that separate different investor types

```json
{{
        "0": {{
        "estimated_P": [[4.0, 0.78], [3.0, 0.85], [2.5, 0.88]],
    "estimated_partitions": [
      {{
        "age": [25, 45, 65],
        "hours_per_week": [20, 40],
        "capital_gain": [0, 5000],
        "__semantic__": 4.0,
        "__utility__": 0.76
      }},
      {{
        "age": [30, 50],
        "hours_per_week": [25, 35, 45],
        "capital_gain": [1000],
        "__semantic__": 3.0,
        "__utility__": 0.82
      }},
      {{
        "age": [28, 42, 58],
        "hours_per_week": [18, 32, 42],
        "capital_gain": [500, 3000, 10000],
        "__semantic__": 2.5,
        "__utility__": 0.86
      }}
    ]
  }}
}}
```"""

    elif task == "causal inference":
        return f"""{base_dataset}
Task: causal inference

For ATE estimation, I need to balance confounders while maintaining interpretable groups that don't introduce bias.

Age: Career stage categories that affect both income potential and demographic factors
Hours_per_week: Employment status groups that influence income through different causal pathways
Capital_gain: Wealth accumulation levels that may confound income relationships

```json
{{
        "0": {{
        "estimated_P": [[4.0, 0.82], [3.5, 0.87], [2.0, 0.91]],
    "estimated_partitions": [
      {{
        "age": [35, 55],
        "hours_per_week": [32, 45],
        "capital_gain": [0, 1000],
        "__semantic__": 4.0,
        "__utility__": 0.81
      }},
      {{
        "age": [30, 45, 60],
        "hours_per_week": [25, 40],
        "capital_gain": [500, 5000],
        "__semantic__": 3.5,
        "__utility__": 0.85
      }},
      {{
        "age": [28, 38, 48, 62],
        "hours_per_week": [20, 35, 42],
        "capital_gain": [200, 2000, 8000],
        "__semantic__": 2.0,
        "__utility__": 0.89
      }}
    ]
  }}
}}
```"""

    elif task == "data imputation":
        return f"""{base_dataset}
Task: data imputation

For imputation accuracy, I need bins that capture strong correlations between attributes to predict missing values effectively.

Age: Life stage groups where income and work patterns are most predictable
Hours_per_week: Employment categories that strongly correlate with income levels
Capital_gain: Investment behavior groups that help predict socioeconomic status

```json
{{
        "0": {{
        "estimated_P": [[4.0, 0.79], [3.0, 0.84], [2.5, 0.87]],
    "estimated_partitions": [
      {{
        "age": [30, 50],
        "hours_per_week": [32, 45],
        "capital_gain": [0, 2000],
        "__semantic__": 4.0,
        "__utility__": 0.77
      }},
      {{
        "age": [25, 40, 60],
        "hours_per_week": [25, 40],
        "capital_gain": [500, 5000],
        "__semantic__": 3.0,
        "__utility__": 0.83
      }},
      {{
        "age": [27, 35, 48, 58],
        "hours_per_week": [22, 35, 44],
        "capital_gain": [100, 1500, 7500],
        "__semantic__": 2.5,
        "__utility__": 0.86
      }}
    ]
  }}
}}
```"""

    else:
        raise ValueError(
            f"Unknown task: {task}. Supported tasks are: 'prediction', 'visualization', 'causal inference', 'data imputation'")


DISCRETIZERS = [
    {
        "name": "Equal Width",
        "type": "unsupervised",
        "notes": "Divides the value range into bins of equal size. Simple, but ignores data distribution."
    },
    {
        "name": "Equal Frequency",
        "type": "unsupervised",
        "notes": "Each bin has roughly the same number of instances. Better for skewed data than equal width."
    },
    {
        "name": "ChiMerge",
        "type": "supervised",
        "notes": "Merges intervals using a chi-squared test for class-label independence. Good for classification tasks."
    },
    {
        "name": "KBinsDiscretizer (Uniform)",
        "type": "unsupervised",
        "notes": "Sklearn’s KBinsDiscretizer with uniform binning. Like equal-width, but integrates well into pipelines."
    },
    {
        "name": "KBinsDiscretizer (Quantile)",
        "type": "unsupervised",
        "notes": "Sklearn’s quantile strategy. Each bin has equal number of samples. Robust to outliers."
    },
    {
        "name": "KMeans Discretizer",
        "type": "unsupervised",
        "notes": "Clusters numeric values using k-means. Adapts to data structure; bins may not be contiguous."
    },
    {
        "name": "Decision Tree Discretizer",
        "type": "supervised",
        "notes": "Learns splits by fitting a shallow decision tree using the outcome variable. Excellent for prediction."
    },
    {
        "name": "Random Forest Discretizer",
        "type": "supervised",
        "notes": "Uses feature importances and splits from a random forest to derive informative bins. Robust for noise."
    },
    {
        "name": "MDLP Discretizer",
        "type": "information-theoretic",
        "notes": "Applies Minimum Description Length Principle. Balances predictive power and bin simplicity."
    },
    {
        "name": "Bayesian Blocks Discretizer",
        "type": "model-based",
        "notes": "Adaptive method that finds optimal change points using a Bayesian model. Ideal for irregular data or time series."
    }
]


def generate_discretization_prompt_l2_norm(data_description: str,
                                   attributes: list,
                                   outcome: str,
                                   task: str,
                                   treatment: str,
                                   include_chain_of_thought: bool = False,
                                   few_shot_example: str = None,
                                   retrieved_info: str = None,
                                   system_prompt: str = SYSTEM_PROMPT_L2_NORM,
                                   attribute_gold_standards: dict = None):
    """
    Generate a prompt for LLM-based discretization suggestion.

    Parameters:
    - data_description: a short description of the dataset (can include schema, stats, etc.)
    - attributes: list of numerical attributes to discretize
    - outcome: name of the outcome variable
    - task: one of ['prediction', 'visualization', 'causal inference', 'data imputation']
    - include_chain_of_thought: whether to instruct the model to reason step-by-step
    - few_shot_example: optional string with example(s) to add before the actual request
    - retrieved_info: optional domain knowledge or relevant documents for RAG

    Returns:
    - full prompt string
    """

    # Few-shot examples (optional)
    few_shot_block = f"\n\n### Example:\n{few_shot_example}" if few_shot_example else ""

    # Retrieved info block (optional)
    rag_block = f"\n\n### Domain Knowledge:\n{retrieved_info}" if retrieved_info else ""

    # Chain-of-thought instruction
    cot_instruction = "\n\nPlease think step by step before choosing bins for each attribute." if include_chain_of_thought else ""

    # Final user prompt
    user_prompt = f"""
    You are given the following dataset:

    {data_description}

    Attributes to discretize: {', '.join(attributes)}
    Attribute gold-standard discretizations: {attribute_gold_standards}
     (Use these to compute L2-norm based semantic similarity)
    Outcome variable: {outcome}
    Task: {task}
    {'Treatment variable: ' + treatment if treatment else ''}
    {'Note: For causal inference, the remaining attributes are confounders and should be discretized to help balance the treatment and outcome. '
     'You are also asked to discretize the treatment variable, but remember that the goal is to enable reliable ATE estimation by properly balancing the confounders.'
    if task == 'causal inference' else ''}
    {'Note: For imputation, 30% of values are randomly masked across all attributes. The imputation model reconstructs the masked values using correlations and patterns in the other features. Design discretizations that preserve predictive relationships between variables.' if task == 'data imputation' else ''}

    Your objective is to suggest a **discretization for each attribute** that maximizes:
    1. Utility for the task (e.g., predictive accuracy, causal ATE, imputation quality, or visualization informativeness),
    2. Semantic interpretability (i.e., alignment with common, intuitive, or domain-specific bins).

    You are allowed to:
    - Apply **different discretization methods** per attribute.
    - Choose **different number of bins** per attribute.
    - Use **statistical, supervised, unsupervised, or knowledge-based** approaches.
    - Leverage **domain knowledge**, if available.

    Return a list of **Pareto-optimal candidate partitions**, each with a tradeoff between utility and interpretability.

    {cot_instruction}
    {few_shot_block}
    {rag_block}

    ### Output format:
    ```json
    {{
      "0": {{
        "estimated_P": [[semantic_score, utility_score], ...],
        "estimated_partitions": [
          {{
            "Attribute1": [cut1, cut2, ..., cutN],
            "Attribute2": [...],
            "__semantic__": float [0-1],
            "__utility__": float [0-1]
          }},
          ...
        ]
      }}
    }}
"""

    return {
        "system": system_prompt,
        "user": user_prompt.strip()
    }


SYSTEM_PROMPT_KL_DIVERGENCE = (
        "You are a data scientist specializing in feature engineering and statistical learning. "
        "Given a dataset, an outcome feature, and numerical attributes, you design intelligent discretization strategies "
        "that improve the performance of ML tasks like prediction, imputation, visualization, or causal inference, "
        "while also enhancing semantic interpretability."
        "Each binning must maximize the utility of the given task and provide meaningful, domain-aligned groupings. "
        "You will return multiple candidate binnings that form a Pareto frontier over task utility and semantic clarity. "
        "Semantic value is evaluated based on the KL divergence between a candidate binning and a given gold-standard binning,"
        "output your semantic score as a float between 0 and 1, where 1 means identical to the gold-standard."
    )

def generate_discretization_prompt_kldiv(data_description: str,
                                   attributes: list,
                                   outcome: str,
                                   task: str,
                                   treatment: str,
                                   include_chain_of_thought: bool = False,
                                   few_shot_example: str = None,
                                   retrieved_info: str = None,
                                   system_prompt: str = SYSTEM_PROMPT_KL_DIVERGENCE,
                                   attribute_gold_standards: dict = None):
    """
    Generate a prompt for LLM-based discretization suggestion.

    Parameters:
    - data_description: a short description of the dataset (can include schema, stats, etc.)
    - attributes: list of numerical attributes to discretize
    - outcome: name of the outcome variable
    - task: one of ['prediction', 'visualization', 'causal inference', 'data imputation']
    - include_chain_of_thought: whether to instruct the model to reason step-by-step
    - few_shot_example: optional string with example(s) to add before the actual request
    - retrieved_info: optional domain knowledge or relevant documents for RAG

    Returns:
    - full prompt string
    """

    # Few-shot examples (optional)
    few_shot_block = f"\n\n### Example:\n{few_shot_example}" if few_shot_example else ""

    # Retrieved info block (optional)
    rag_block = f"\n\n### Domain Knowledge:\n{retrieved_info}" if retrieved_info else ""

    # Chain-of-thought instruction
    cot_instruction = "\n\nPlease think step by step before choosing bins for each attribute." if include_chain_of_thought else ""

    # Final user prompt
    user_prompt = f"""
    You are given the following dataset:

    {data_description}

    Attributes to discretize: {', '.join(attributes)}
    Attribute gold-standard discretizations: {attribute_gold_standards}
     (Use these to compute KL divergence based semantic similarity)
    Outcome variable: {outcome}
    Task: {task}
    {'Treatment variable: ' + treatment if treatment else ''}
    {'Note: For causal inference, the remaining attributes are confounders and should be discretized to help balance the treatment and outcome. '
     'You are also asked to discretize the treatment variable, but remember that the goal is to enable reliable ATE estimation by properly balancing the confounders.'
    if task == 'causal inference' else ''}
    {'Note: For imputation, 30% of values are randomly masked across all attributes. The imputation model reconstructs the masked values using correlations and patterns in the other features. Design discretizations that preserve predictive relationships between variables.' if task == 'data imputation' else ''}

    Your objective is to suggest a **discretization for each attribute** that maximizes:
    1. Utility for the task (e.g., predictive accuracy, causal ATE, imputation quality, or visualization informativeness),
    2. Semantic interpretability (i.e., alignment with common, intuitive, or domain-specific bins).

    You are allowed to:
    - Apply **different discretization methods** per attribute.
    - Choose **different number of bins** per attribute.
    - Use **statistical, supervised, unsupervised, or knowledge-based** approaches.
    - Leverage **domain knowledge**, if available.

    Return a list of **Pareto-optimal candidate partitions**, each with a tradeoff between utility and interpretability.

    {cot_instruction}
    {few_shot_block}
    {rag_block}

    ### Output format:
    ```json
    {{
      "0": {{
        "estimated_P": [[semantic_score, utility_score], ...],
        "estimated_partitions": [
          {{
            "Attribute1": [cut1, cut2, ..., cutN],
            "Attribute2": [...],
            "__semantic__": float [0-1],
            "__utility__": float [0-1]
          }},
          ...
        ]
      }}
    }}
"""

    return {
        "system": system_prompt,
        "user": user_prompt.strip()
    }

def get_rel_few_shot_distance_based(task):
    """
    Returns a few-shot example for the given task.
    """

    # Base dataset description (same for all tasks)
    base_dataset = """Dataset: Adult income dataset with 32,561 records
Attributes to discretize: age (17-90), hours_per_week (1-99), capital_gain (0-99999)
Outcome variable: income_class (>50K, <=50K)"""

    if task == "prediction":
        return f"""{base_dataset}
Task: prediction

For prediction accuracy, I need to find splits that maximize information gain and separate the outcome classes effectively.

Age: Standard life stages are semantically meaningful and align with income patterns
Hours_per_week: Work intensity categories that employers and researchers commonly use
Capital_gain: Financial thresholds that differentiate investment behaviors

```json
{{
        "0": {{
        "estimated_P": [[1.0, 0.85], [0.875, 0.89], [0.5, 0.92]],
    "estimated_partitions": [
      {{
        "age": [25, 45, 65],
        "hours_per_week": [32, 40],
        "capital_gain": [0],
        "__semantic__": 1.0,
        "__utility__": 0.85
      }},
      {{
        "age": [30, 50],
        "hours_per_week": [20, 40, 50],
        "capital_gain": [1000, 5000],
        "__semantic__": 0.875,
        "__utility__": 0.89
      }},
      {{
        "age": [23, 28, 35, 42, 55, 67],
        "hours_per_week": [15, 25, 35, 42, 48],
        "capital_gain": [500, 2500, 7500],
        "__semantic__": 0.5,
        "__utility__": 0.92
      }}
      ]
  }}
}}
```"""

    elif task == "visualization":
        return f"""{base_dataset}
Task: visualization

For visualization interestingness, I need bins that reveal clear patterns in income distribution and are easy to interpret in charts.

Age: Life stage categories that show distinct income patterns across age groups
Hours_per_week: Work intensity levels that create meaningful employment categories  
Capital_gain: Investment activity levels that separate different investor types

```json
{{
        "0": {{
        "estimated_P": [[1.0, 0.78], [0.67, 0.85], [0.55, 0.88]],
    "estimated_partitions": [
      {{
        "age": [25, 45, 65],
        "hours_per_week": [20, 40],
        "capital_gain": [0, 5000],
        "__semantic__": 1.0,
        "__utility__": 0.76
      }},
      {{
        "age": [30, 50],
        "hours_per_week": [25, 35, 45],
        "capital_gain": [1000],
        "__semantic__": 0.67,
        "__utility__": 0.82
      }},
      {{
        "age": [28, 42, 58],
        "hours_per_week": [18, 32, 42],
        "capital_gain": [500, 3000, 10000],
        "__semantic__": 0.55,
        "__utility__": 0.86
      }}
    ]
  }}
}}
```"""

    elif task == "causal inference":
        return f"""{base_dataset}
Task: causal inference

For ATE estimation, I need to balance confounders while maintaining interpretable groups that don't introduce bias.

Age: Career stage categories that affect both income potential and demographic factors
Hours_per_week: Employment status groups that influence income through different causal pathways
Capital_gain: Wealth accumulation levels that may confound income relationships

```json
{{
        "0": {{
        "estimated_P": [[1.0, 0.82], [0.875, 0.87], [0.5, 0.91]],
    "estimated_partitions": [
      {{
        "age": [35, 55],
        "hours_per_week": [32, 45],
        "capital_gain": [0, 1000],
        "__semantic__": 1.0,
        "__utility__": 0.81
      }},
      {{
        "age": [30, 45, 60],
        "hours_per_week": [25, 40],
        "capital_gain": [500, 5000],
        "__semantic__": 0.875,
        "__utility__": 0.85
      }},
      {{
        "age": [28, 38, 48, 62],
        "hours_per_week": [20, 35, 42],
        "capital_gain": [200, 2000, 8000],
        "__semantic__": 0.5,
        "__utility__": 0.89
      }}
    ]
  }}
}}
```"""

    elif task == "data imputation":
        return f"""{base_dataset}
Task: data imputation

For imputation accuracy, I need bins that capture strong correlations between attributes to predict missing values effectively.

Age: Life stage groups where income and work patterns are most predictable
Hours_per_week: Employment categories that strongly correlate with income levels
Capital_gain: Investment behavior groups that help predict socioeconomic status

```json
{{
        "0": {{
        "estimated_P": [[1.0, 0.79], [0.67, 0.84], [0.55, 0.87]],
    "estimated_partitions": [
      {{
        "age": [30, 50],
        "hours_per_week": [32, 45],
        "capital_gain": [0, 2000],
        "__semantic__": 1.0,
        "__utility__": 0.77
      }},
      {{
        "age": [25, 40, 60],
        "hours_per_week": [25, 40],
        "capital_gain": [500, 5000],
        "__semantic__": 0.67,
        "__utility__": 0.83
      }},
      {{
        "age": [27, 35, 48, 58],
        "hours_per_week": [22, 35, 44],
        "capital_gain": [100, 1500, 7500],
        "__semantic__": 0.55,
        "__utility__": 0.86
      }}
    ]
  }}
}}
```"""

    else:
        raise ValueError(
            f"Unknown task: {task}. Supported tasks are: 'prediction', 'visualization', 'causal inference', 'data imputation'")
