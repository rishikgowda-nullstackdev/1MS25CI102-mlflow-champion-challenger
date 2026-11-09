# Champion–Challenger Model Comparison with Hugging Face and MLflow

| Course Assessment | Mini Project: End-to-End MLflow Deployment (50 Marks) |
|---|---|
| **Student Name** | Rishikgowda S M |
| **Student USN** | 1MS25CI102 |
| **Task** | Binary Sentiment Classification |
| **Experiment Name** | `hf-sentiment-champion-challenger` |
| **Registered Model** | `sentiment-model` |

---

## 1. Task and Models

- **Task**: Binary Sentiment Classification (Classifying customer and critic review sentences as `POSITIVE` or `NEGATIVE`).
- **Dataset**: Stanford Sentiment Treebank v2 (`glue/sst2` validation split).
- **Evaluation Subset Size**: 150 balanced samples (75 positive, 75 negative) with fixed `random_state=42`, saved in `data/eval_inputs.csv`.
- **Inference Verification Set**: 8 fresh, unseen sentences saved in `data/new_data.csv`.

### Candidate Models:
1. **Model A (Champion Baseline Candidate)**:
   - **Hugging Face ID**: `distilbert-base-uncased-finetuned-sst-2-english`
   - **Architecture**: DistilBERT (6 transformer layers, 768 hidden dimension)
   - **Parameters**: ~67 Million (~268 MB)
   - **Role**: Lightweight, low-latency production baseline champion.
2. **Model B (Challenger Candidate)**:
   - **Hugging Face ID**: `textattack/bert-base-uncased-SST-2`
   - **Architecture**: Standard BERT Base (12 transformer layers, 768 hidden dimension)
   - **Parameters**: ~110 Million (~440 MB)
   - **Role**: Higher parameter-capacity challenger model evaluated to test whether accuracy gains justify larger size and inference cost.

---

## 2. Champion Rule (Declared Prior to Final Run)

As required by Section 5.4 and Ground Rules of the assignment, the Champion decision rule is formally stated before running the evaluation:

1. **Primary Metric**: **Macro-F1 Score** (Macro-averaged F1 across Positive and Negative classes).
   - *Direction*: **Higher is better**.
2. **Minimum Improvement Margin ($\Delta \ge +0.01$)**:
   - The Challenger (Model B) must exceed the Champion (Model A) by at least **1.0 percentage point (+0.01)** in Macro-F1 score to justify the overhead of its larger parameter size (~110M vs ~67M) and deployment resource requirements.
3. **Guardrail / Tie-Breaker**:
   - **Latency Guardrail**: The average inference latency per sample of the Challenger must **not exceed 2.0×** the Champion's latency ($Latency_B \le 2.0 \times Latency_A$).
   - **Tie-Breaker**: If the Challenger's F1 score improvement is less than +0.01, or if the latency guardrail is violated, the baseline Champion (Model A) retains the `@champion` title due to its computational efficiency.

---

## 3. How to Run

Follow these commands in sequence to reproduce the entire pipeline from scratch:

### Step 1: Environment Setup
```powershell
# Create virtual environment with Python 3.12
py -3.12 -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Run Evaluation Pipeline
Executes both models on identical inputs, logs runs to MLflow, computes metrics, and generates comparison artifacts:
```powershell
python evaluate.py
```

### Step 3: Register Models and Assign Champion / Challenger Aliases
Registers both models in the MLflow Model Registry under `sentiment-model` and automatically sets the `@champion` and `@challenger` aliases:
```powershell
python register_champion.py
```

### Step 4: Run Inference via MLflow Alias
Loads the Champion model strictly via `models:/sentiment-model@champion` without directly importing Hugging Face pipelines:
```powershell
python predict_from_mlflow.py
```

### Step 5: View MLflow Tracking Dashboard
Launch the MLflow UI to inspect runs, artifacts, compare views, and registry aliases:
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open your browser at `http://127.0.0.1:5000`.

---

## 4. Results Table

*(Will be populated upon pipeline execution)*

| Metric / Attribute | Model A (DistilBERT) | Model B (BERT SST-2) | Difference (B - A) | Superior |
|---|---|---|---|---|
| **Parameters** | ~67M | ~110M | +43M (+64%) | Model A (Smaller) |
| **Accuracy** | TBD | TBD | TBD | TBD |
| **Macro-F1** (Primary) | TBD | TBD | TBD | TBD |
| **Precision** | TBD | TBD | TBD | TBD |
| **Recall** | TBD | TBD | TBD | TBD |
| **Avg Latency (ms)** | TBD | TBD | TBD | TBD |
| **Total Inference (s)**| TBD | TBD | TBD | TBD |

---

## 5. Decision & Discussion (150–250 Words)

*(Will be completed following the final evaluation runs)*

---

## 6. Limitations and Declaration

- **Sample Size**: Evaluated on 150 balanced sentences from SST-2 validation data. While sufficient for statistical signal and rapid CPU testing, a larger set (1,000+ examples) would provide tighter confidence bounds.
- **Hardware Variation**: Latency measurements were benchmarked on local CPU execution; relative speedups may differ on GPU inference or under concurrent serving loads.
- **Academic Declaration**: This project was developed as part of the Skill Lab (MLOps with MLflow) curriculum. All code, MLflow logging logic, custom PyFunc wrappers, and analysis were written and verified by the student. AI coding assistance was used for initial template scaffolding and documentation formatting in accordance with course guidelines.
