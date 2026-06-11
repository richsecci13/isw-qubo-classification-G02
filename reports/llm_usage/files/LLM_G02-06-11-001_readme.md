Congratulations on completing the pipeline! You have built a robust, enterprise-grade machine learning system that strictly adheres to the project specifications.

Below is the complete, professional `README.md` file tailored exactly to the requirements outlined in the PDF. It covers the project description, repository structure, installation instructions, GUI launch commands, CLI execution, and automated testing .

You can copy and paste this directly into your repository's root directory.

---

### **`README.md`**

```markdown
# QUBO Feature Reduction & Credit Risk Classification

This repository contains a complete Python machine learning pipeline for binary classification, specifically designed to assess credit risk ("0" for reliable, "1" for risky). The core innovation of this project is the use of Quadratic Unconstrained Binary Optimization (QUBO) via Simulated Annealing to perform advanced feature reduction prior to model training.

This project was developed for the Ingegneria del Software (Software Engineering) course (AA 2025-26).

## 🏗️ Project Architecture

The pipeline is divided into four main modular phases:
1. **Preprocessing:** Cleans raw CSV data, drops sparse features based on a configurable threshold, normalizes numeric data using Z-score standardization, and splits the data sequentially into training and testing sets.
2. **QUBO Feature Selection:** Formulates feature selection as a QUBO problem. It balances target influence (Spearman correlation with the target) against feature independence (penalizing collinearity). It uses a Two-Phase Binary/Linear search with D-Wave's `neal` Simulated Annealing sampler to find the global energy minimum for a targeted percentage of features.
3. **Model Training:** Supports Random Forest, XGBoost, and Logistic Regression. It automatically handles class imbalances, performs hyperparameter grid-search targeting the F1-score, and dynamically tunes the decision threshold. 
4. **Prediction & Inference:** Evaluates the trained models on the reduced test set, producing detailed metrics (Accuracy, ROC-AUC, F1, Precision, Recall, and Confusion Matrices).

## 📂 Repository Structure

The repository strictly follows the required layout for automated evaluation:

```text
isw-qubo-classification-GXX/
├── README.md
├── requirements.txt
├── group_info.yaml
├── data/
│   └── sample_test_dataset.csv         # Small dataset for automated pytest
├── src/
│   └── qubo_project/
│       ├── __init__.py
│       ├── preprocessing.py            # Phase 1: Cleaning & Normalization
│       ├── feature_selection.py        # Phase 2: QUBO Optimization
│       ├── model.py                    # Phase 3 & 4: Training & Inference
│       └── gui.py                      # Streamlit Graphical User Interface
├── tests/
│   ├── test_preprocessing.py           # Pytest suite for Phase 1
│   ├── test_feature_selection.py       # Pytest suite for Phase 2
│   └── test_model.py                   # Pytest suite for Phase 3 & 4
├── outputs/                            # Runtime artifacts (CSVs, Models, JSONs)
├── reports/
│   └── project_report.yaml             # Final project report
└── llm_usage/
    └── (Markdown logs of LLM interactions)

```

## ⚙️ Installation & Setup

**Requirements:** Python 3.11 or higher.

1. Clone the repository:
```bash
git clone <repository_url>
cd isw-qubo-classification-GXX
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## 🖥️ Graphical User Interface (GUI)

A fully interactive dashboard built with Streamlit is provided to manage the entire pipeline without using the command line. Ensure you are in the repository root directory, then run:

```bash
streamlit run src/qubo_project/gui.py
```

## 💻 Command Line Interface (CLI) Execution

The modules are designed to be executed sequentially via the command line for reproducibility and automated grading. Run these from the root directory.

### 1. Preprocessing

```bash
python src/qubo_project/preprocessing.py \
    --input data/input_dataset.csv \
    --target target \
    --out-data outputs/normalized.csv \
    --out-json outputs/preprocessing_result.json \
    --min-perc-valid 0.05
```

### 2. Feature Selection (QUBO)

```bash
python src/qubo_project/feature_selection.py \
    --in-normalized outputs/normalized.csv \
    --out-train outputs/training_reduced.csv \
    --out-test outputs/test_reduced.csv \
    --out-optimizations outputs/optimizations.csv \
    --out-json outputs/feature_selection_result.json \
    --target target \
    --perc-selected 0.20 \
    --allowance 1 \
    --perc-test 0.30 \
    --seed 42 \
    --alpha-computations 100
```

### 3. Model Training

```bash
python src/qubo_project/model.py train \
    --classifier random_forest \
    --in-reduced outputs/training_reduced.csv \
    --target target \
    --out-model outputs/model.joblib \
    --out-metrics outputs/training_metrics.json \
    --seed 42
```

*(Supported classifiers: `random_forest`, `xgboost`, `logistic_regression`)*

### 4. Prediction

```bash
python src/qubo_project/model.py predict \
    --input-testset outputs/test_reduced.csv \
    --target target \
    --model outputs/model.joblib \
    --out-predictions outputs/predictions.csv \
    --out-stats outputs/classification_stats.json
```

## 🧪 Automated Testing

The project includes an automated test suite verifying all core requirements. The tests execute against the physical `data/sample_test_dataset.csv` file. To run the suite:

```bash
pytest tests/ -v
```

## 👥 Group Information

* **Group ID:** GXX
* **Members:** [Add Matricola 1], [Add Matricola 2]

***

### **Final Polish Checklist**
Before you zip the project or push it to GitHub for the final submission, ensure you do the following:
1. Replace the `GXX` and `[Add Matricola X]` placeholders in the `README.md`, repository folder name, and `group_info.yaml` with your actual group details.
2. Clear out any massive experimental datasets from the `data/` folder, leaving ONLY the `sample_test_dataset.csv` required for `pytest`.
3. Generate your `requirements.txt` using `pip freeze > requirements.txt` (clean it up so it only includes the necessary libraries like pandas, scikit-learn, xgboost, dimod, neal, streamlit, and pytest).
```