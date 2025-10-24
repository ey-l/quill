# Quill: Interpretable Attribute Discretization

Quill is a research codebase for **interpretable attribute discretization**—turning continuous attributes into human-meaningful bins while preserving downstream task utility. It provides baselines and evaluators for utility/semantics trade-offs, along with scripts and prompts to reproduce results.

> 🔎 Why this matters: good bins make models easier to explain, visualizations clearer, and causal/statistical analyses more robust—without giving up performance.


<p align="center">
  <img src="./quill-overview.png" width="500"/>
  <br>
  <em>Figure 1: Quill overview.</em>
</p>

---

## Features

- 📊 **Utility–Semantics trade-off**: evaluate partitions for both task performance and human-meaningfulness.
- 🧪 **Baselines included**: UCB- and MCC-style evaluators (see `evaluate_MCC.py`, `evaluate_MCC_single.py`).
- 📁 **Organized repo layout**: data inputs, ground-truths, prompts, and scripts for repeatable runs.
- 📜 **Technical report**: a PDF with background and methods (`full_technical_report.pdf`).

---

## Repository Structure

```
quill/
├─ baselines/                # Reference/benchmark implementations
├─ data/                     # Input datasets
├─ prompts/                  # LLM prompts used in experiments
├─ scripts/                  # End-to-end or helper scripts
├─ truth/                    # Ground-truth partitions or labels
├─ evaluate_MCC.py           # MCC-based search (Quill) evaluator
├─ evaluate_MCC_single.py    # UCB-guided MCC variant (Quill) evaluator
├─ requirements_cleaned.txt  # Python dependencies
└─ full_technical_report.pdf # Technical write-up
```

---

## Quickstart

### 1) Environment Setup

```bash
# (Optional) create an isolated environment
conda create -y -n quill python=3.10
conda activate quill

# Install dependencies
pip install -r requirements_cleaned.txt
```

### 2) Provide Data

Place your input CSVs or tables under:  
```
data/
```

Place expected ground-truths (if applicable) under:
```
truth/
```

### 3) Run Evaluators

```bash
# MCC-based search (for multi-attribute settings)
python evaluate_MCC.py --dataset <dataset> --use_case <use_case> --semantic_metric <semantic_metric>

# UCB-guided MCC variant (for single attribute settings)
python evaluate_MCC_single.py --dataset <dataset> --use_case <use_case> --semantic_metric r<semantic_metric>
```

---

## Output Interpretation

Evaluation outputs typically include:

- **Utility metrics:** how well downstream tasks perform on discretized features
- **Semantic metrics:** alignment of bins with domain meaning or human interpretability
- **Pareto sets:** partitions that optimally trade off utility and semantic quality

If you produce CSV/JSON results, consider adding a short script to aggregate them and plot the utility–semantics frontier.

---

