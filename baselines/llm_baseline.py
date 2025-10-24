#the copy that stays in the repo
import json
import re
import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv
import datetime
import argparse
from openai import OpenAI
from pathlib import Path
sys.path.append("./")
from prompts.llm_binning_prompts import generate_discretization_prompt, get_rel_few_shot, DISCRETIZERS, generate_discretization_prompt_kldiv, get_rel_few_shot_distance_based

def generate_data_description(df, max_unique_vals=5):
    description = []
    for col in df.columns:
        dtype = df[col].dtype
        col_desc = f"- {col} ({dtype})"

        if pd.api.types.is_numeric_dtype(df[col]):
            col_desc += f": min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}"
        elif df[col].nunique() <= max_unique_vals:
            unique_vals = df[col].unique()
            col_desc += f": values={list(unique_vals)}"
        else:
            col_desc += f": {df[col].nunique()} unique values"

        description.append(col_desc)

    return "\n".join(description)


def format_rag_block(discretizer_knowledge):
    lines = ["You may consider the following discretization techniques:"]
    for d in discretizer_knowledge:
        line = f"- {d['name']}: {d['type'].capitalize()} — {d['notes']}"
        lines.append(line)
    return "\n".join(lines)

def get_data_info(data_name, task):
    if task == "causal inference":
        exp_config = json.load(open(project_root.parent / 'data' / data_name / f'{data_name}-causal.json'))
    else: exp_config = json.load(open(project_root.parent / 'data' / data_name / f'{data_name}.json'))
    info_dict={'attributes': list(exp_config['attributes'].keys()),'outcome': exp_config['target'],'features': exp_config['features'],'path': exp_config['data_path'],'treatment':None}
    if task == "causal inference":
        info_dict['attributes'] = [attr for attr in info_dict['attributes'] if attr != info_dict['treatment']]
        info_dict['treatment'] = exp_config['treatment']
    if task == "visualization":
        info_dict['attributes'] = [exp_config['attribute_viz']]
        info_dict['target'] = [exp_config['target_viz']]
    return info_dict


def get_feature_correlations(df: pd.DataFrame, attributes: list) -> str:
    corr = df[attributes].corr().round(2)
    lines = ["### Pairwise Correlations"]
    for i, row_name in enumerate(corr.index):
        for j, col_name in enumerate(corr.columns):
            if i < j:  # upper triangle only
                val = corr.loc[row_name, col_name]
                lines.append(f"- {row_name} vs {col_name}: {val}")
    return "\n".join(lines)

def get_feature_stats(df: pd.DataFrame, attributes: list) -> dict:
    stats = {}
    for col in attributes:
        series = df[col].dropna()
        stats[col] = {
            "min": series.min(),
            "max": series.max(),
            "mean": series.mean(),
            "std": series.std(),
            "percentiles": {
                "10%": series.quantile(0.10),
                "25%": series.quantile(0.25),
                "50%": series.quantile(0.50),
                "75%": series.quantile(0.75),
                "90%": series.quantile(0.90)
            },
            "num_unique": series.nunique()
        }
    return stats

def format_stats_for_prompt(stats_dict):
    lines = ["### Data Distribution Summary"]
    for feature, s in stats_dict.items():
        lines.append(f"{feature}:")
        lines.append(f"- Min: {s['min']}, Max: {s['max']}, Mean: {s['mean']:.2f}, Std: {s['std']:.2f}")
        lines.append(f"- 10%: {s['percentiles']['10%']}, 25%: {s['percentiles']['25%']}, 50%: {s['percentiles']['50%']}, "
                     f"75%: {s['percentiles']['75%']}, 90%: {s['percentiles']['90%']}")
        lines.append(f"- Unique values: {s['num_unique']}")
        lines.append("")
    return "\n".join(lines)

def get_input_args(dataset, task):
    info_dict=get_data_info(dataset, task)
    df= pd.read_csv(project_root.parent/info_dict['path'])
    return df, info_dict['features'],info_dict['attributes'],info_dict['outcome'],info_dict['treatment']


def main(dataset,task="prediction"):
    """main function to run all combination of tasks, datasets, and flags (RAG, few-shot, chain of thought)"""
    os.makedirs("outputs", exist_ok=True)

    results = []
    timing_records = []
    df, features, attributes, outcome,treatment=get_input_args(dataset, task)
    data_stats = get_feature_stats(df, attributes)
    data_desc = format_stats_for_prompt(data_stats)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if task=='data imputation':
        correlation_block = get_feature_correlations(df.dropna(subset=attributes), attributes)
        data_desc = f"{data_desc}\n{correlation_block}"
    for include_chain_of_thought in [False, True]:
        for few_shot_example in [None, get_rel_few_shot_distance_based(task)]:
            for retrieved_info in [None, format_rag_block(DISCRETIZERS)]:
                print(f"Generating prompt for dataset={dataset}, task={task}, "
                      f"include_chain_of_thought={include_chain_of_thought}, "
                      f"few_shot_example={few_shot_example is not None}, "
                      f"retrieved_info={retrieved_info is not None}")
                variation_start = time.time()
                for i in range(20):
                    run_start = time.time()
                    print(f"\nProcessing response {i + 1} for task={task}...")
                    if task=="visualization":
                        for attribute in attributes:
                            print(f"Generating prompt for attribute={attribute} in task={task}")
                            # Generate prompt for each attribute separately
                            result_entry = get_prompt_n_results(
                                data_desc,
                                [attribute],
                                outcome,
                                task,
                                include_chain_of_thought,
                                few_shot_example,
                                retrieved_info,i
                            )
                            results.append(result_entry)
                    else:
                        result_entry=get_prompt_n_results(data_desc,attributes,outcome,task,include_chain_of_thought,few_shot_example,retrieved_info,i,treatment)
                        results.append(result_entry)
                    run_end = time.time()
                    timing_records.append({
                        "dataset": dataset,
                        "task": task,
                        "include_chain_of_thought": include_chain_of_thought,
                        "few_shot_used": few_shot_example is not None,
                        "rag_used": retrieved_info is not None,
                        "variation": f"{include_chain_of_thought}_{few_shot_example is not None}_{retrieved_info is not None}",
                        "run_number": i + 1,
                        "run_time_sec": run_end - run_start
                    })
                    print(f"Time for run {i + 1}: {run_end - run_start:.2f} seconds")
                variation_end = time.time()
                timing_records.append({
                    "dataset": dataset,
                    "task": task,
                    "include_chain_of_thought": include_chain_of_thought,
                    "few_shot_used": few_shot_example is not None,
                    "rag_used": retrieved_info is not None,
                    "variation": f"{include_chain_of_thought}_{few_shot_example is not None}_{retrieved_info is not None}",
                    "run_number": "total",
                    "run_time_sec": variation_end - variation_start
                })
                print(f" Total time for this variation: {variation_end - variation_start:.2f} seconds")
    file_path = f"outputs/{dataset}_discretization_results_{timestamp}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ All prompts processed. Results saved to `outputs/{dataset}_discretization_results_{timestamp}.json`.")
    # Save timing table
    timing_df = pd.DataFrame(timing_records)
    timing_df.to_csv(f"outputs/{dataset}_timing_{timestamp}.csv", index=False)
    return timestamp

def get_prompt_n_results(data_desc,attributes,outcome,task,include_chain_of_thought,few_shot_example,retrieved_info,i,treatment=None):
    generator = Generator(model="gpt-4o", verbose=True)
    prompt = generate_discretization_prompt_kldiv(
        data_desc,
        attributes,
        outcome,
        task,
        treatment,
        include_chain_of_thought=include_chain_of_thought,
        few_shot_example=few_shot_example,
        retrieved_info=retrieved_info
    )
    print(prompt)
    response = generator.generate(prompt)
    print(f"Response {i + 1}:", response)
    result_entry = {
        "dataset": dataset,
        "task": task,
        "atrributes": attributes,
        "include_chain_of_thought": include_chain_of_thought,
        "few_shot_used": few_shot_example is not None,
        "rag_used": retrieved_info is not None,
        "prompt": prompt,
        "response_number": i + 1,
        "response": response
    }
    return result_entry

def get_api_key(key: str) -> str:
    # get API key from environment or throw an exception if it's not set
    if key not in os.environ:
        print(f"KEY: {key}")
        print(f"{os.environ.keys()}")
        raise ValueError("key not found in environment variables")

    return os.environ[key]

def extract_clean_json(text):
    """Extract the first JSON block in the response and clean it."""
    try:
        # Try to extract triple-backtick JSON block
        match = re.search(r"```json(.*?)```", text, re.DOTALL)
        if not match:
            match = re.search(r"```(.*?)```", text, re.DOTALL)
        json_text = match.group(1) if match else text

        # Clean invalid JSON formatting
        json_text = json_text.strip()
        json_text = re.sub(r"'", '"', json_text)  # Replace single quotes with double
        json_text = re.sub(r"None", "null", json_text)
        json_text = re.sub(r"True", "true", json_text)
        json_text = re.sub(r"False", "false", json_text)

        return json.loads(json_text)
    except Exception as e:
        return None

def parse_response_entry(entry):
    dataset = entry.get("dataset")
    task = entry.get("task")
    cot = entry.get("include_chain_of_thought", False)
    few_shot = entry.get("few_shot_used", False)
    rag = entry.get("rag_used", False)
    response_number = entry.get("response_number", None)

    response_text = entry.get("response", "")
    parsed_json = extract_clean_json(response_text)
    if not parsed_json:
        print(f"❌ Failed parsing response JSON for {dataset}")
        return []

    rows = []
    for i, partition in enumerate(parsed_json.get("0", {}).get("estimated_partitions", []), start=1):
        estimated_P = parsed_json["0"].get("estimated_P", [])[i-1] if i-1 < len(parsed_json["0"].get("estimated_P", [])) else None
        partition_clean = {k: v for k, v in partition.items() if not k.startswith("__")}
        rows.append({
            "Dataset": dataset,
            "Task": task,
            "Chain of Thought": cot,
            "Few-shot Used": few_shot,
            "RAG Used": rag,
            "Candidate #": i,
            "estimated_P": estimated_P,
            "response_number": response_number,
            "Partition (bins)": partition_clean
        })
    return rows

def parse_response(dataset,timestamp,task):
    # Load JSON file (replace with your path or dictionary)
    with open(f"outputs/{dataset}_discretization_results_{timestamp}.json", "r") as f:
        response_data = json.load(f)

    # For a single response:
    # rows = parse_response_entry(response_data)

    # For multiple responses:
    rows = []
    for entry in response_data:
        rows.extend(parse_response_entry(entry))

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns for final table
    df = df[[
        "Dataset", "Task", "Chain of Thought", "Few-shot Used", "RAG Used",
        "Candidate #", "response_number","estimated_P", "Partition (bins)"
    ]]
    # df.to_csv(f"outputs/{dataset}_df_discretization_results_{timestamp}.csv", index=False)
    df.to_csv(project_root.parent / "testresults" / "LLM" / f"{dataset}_df_discretization_results_{timestamp}_{task}.csv", index=False)


class Generator():
    def __init__(self, model: str = "o3-2025-04-16", verbose=False):
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
                    messages=[
                        {"role": "system", "content":  prompt["system"]},
                        {"role": "user", "content": prompt["user"]}
                    ],
                    temperature=0.2,
                    max_tokens=4096
                )
                break  # break out of while loop if no error

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
            # print(res)
        return res

data_description = lambda df: generate_data_description(df, max_unique_vals=5)
load_dotenv()
OpenAIModelList = ["gpt-4o", "o3-2025-04-16"]

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
project_root = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'diabetes')")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., 'causal inference')")
    args = parser.parse_args()

    dataset = args.dataset
    task = args.task
    timestamp = main(dataset, task)
    elapsed_seconds = time.time() - datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M").timestamp()
    hours, remainder = divmod(int(elapsed_seconds), 3600)
    minutes, _ = divmod(remainder, 60)
    print(f"the time taken to run the task is: {hours} hours, {minutes} minutes")

    parse_response(dataset, timestamp,task)
    print(
        f"✅ Response parsing complete. Results saved to `testresults" / "LLM" / f"{dataset}.{timestamp}.{task}.csv`.")
