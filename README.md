# Champion–Challenger Model Comparison with Hugging Face and MLflow

|---|---|
| **Name** | Rishikgowda S M |
| **USN** | 1MS25CI102 |
| **Task** | Binary Sentiment Classification |
| **Experiment Name** | `hf-sentiment-champion-challenger` |
| **Registered Model** | `sentiment-model` |

---

## 1. Task and Models

- **Task**: Binary Sentiment Classification (Classifying movie review sentences as `POSITIVE` or `NEGATIVE`).
- **Dataset**: Stanford Sentiment Treebank v2 (`stanfordnlp/sst2` validation split).
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
*(Optional stretch goal: test challenger alias using `python predict_from_mlflow.py --alias challenger`)*

### Step 5: View MLflow Tracking Dashboard
Launch the MLflow UI to inspect runs, artifacts, compare views, and registry aliases:
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open your browser at `http://127.0.0.1:5000`.

---

## 4. Results Table

Both candidate models were evaluated on the identical 150-sample balanced dataset (`data/eval_inputs.csv`) under CPU execution:

| Metric / Attribute | Model A (DistilBERT) | Model B (BERT SST-2) | Difference (B - A) | Better Model |
|---|---|---|---|---|
| **Hugging Face ID** | `distilbert-base-uncased...` | `textattack/bert-base...` | - | - |
| **Parameters** | ~67M | ~110M | +43M (+64%) | Model A (Smaller) |
| **Accuracy** | 0.8867 (88.67%) | 0.9200 (92.00%) | +0.0333 (+3.33%) | Model B |
| **Macro-F1** (Primary) | **0.8865** | **0.9200** | **+0.0335 (+3.35%)** | **Model B** |
| **Weighted-F1** | 0.8865 | 0.9200 | +0.0335 (+3.35%) | Model B |
| **Precision** | 0.8625 | 0.9200 | +0.0575 (+5.75%) | Model B |
| **Recall** | 0.9200 | 0.9200 | 0.0000 | Tied |
| **Avg Latency (ms)** | 28.90 ms | 57.59 ms | +28.69 ms (1.99x) | Model A (Faster) |
| **Total Inference (s)**| 4.33 s | 8.64 s | +4.31 s | Model A |
| **Rule Outcome** | Baseline Challenger | **Promoted Champion** | Satisfies Rule | **Model B** |

---

## 5. Decision & Discussion (189 Words)

Following our pre-declared evaluation rule, Model B (Challenger: `textattack/bert-base-uncased-SST-2`) successfully defeated Model A (Champion Baseline: `distilbert-base-uncased-finetuned-sst-2-english`) to earn promotion to the `@champion` alias. On the 150-sample balanced evaluation set, the Challenger achieved a Macro-F1 score of 0.9200 compared to 0.8865 for the baseline—an absolute improvement of +0.0335 (+3.35 percentage points). This easily surpasses the pre-declared minimum margin of +0.01 (+1.0 percentage point). Furthermore, the Challenger satisfied the latency guardrail: its average per-sample latency was 57.59 ms compared to 28.90 ms for DistilBERT, representing a 1.99x slowdown, which remains within the permissible 2.0x latency threshold.

The primary trade-off involves computational cost and model size. The Challenger contains ~110M parameters (~440 MB) versus DistilBERT's ~67M parameters (~268 MB), requiring 64% more storage and roughly double the inference computation. However, error analysis across the 9 disagreement cases reveals that BERT Base's additional capacity significantly improves handling of complex syntactic structures, negations (e.g., 'i don't think i laughed out loud once'), and qualified reviews. Given that the 3.35% accuracy boost is accompanied by consistent precision and recall gains (both 0.9200), the deployment trade-off is justified for accuracy-sensitive applications.

---

## 6. Qualitative Disagreement Analysis

Below are representative disagreement cases extracted from `outputs/disagreement_analysis.csv` showing key behavioral differences:

| # | Input Sentence | Ground Truth | DistilBERT (Model A) | BERT Base (Model B) | Analysis |
|---|---|---|---|---|---|
| 1 | *"i do n't think i laughed out loud once ."* | NEGATIVE | POSITIVE (0.8525) | NEGATIVE (0.9448) | DistilBERT is misled by positive lexical tokens ("laughed out loud"); BERT correctly processes the negation dependency. |
| 2 | *"outer-space buffs might love this film , but others will find its pleasures intermittent ."* | NEGATIVE | POSITIVE (0.7708) | NEGATIVE (0.7890) | Contrastive clause "but others..." inverts sentiment; BERT weights the second clause properly. |
| 3 | *"the best that can be said ... is that he obviously does n't have his heart in it ."* | NEGATIVE | POSITIVE (0.8562) | NEGATIVE (0.9972) | Idiomatic negation ("doesn't have his heart in it") successfully parsed by BERT; DistilBERT biased by "best". |
| 4 | *"by getting myself wrapped up in the visuals ... i found myself confused when it came time..."* | NEGATIVE | POSITIVE (0.9936) | NEGATIVE (0.6741) | Mixed review where visual praise precedes narrative criticism; BERT correctly assigns negative overall verdict. |
| 5 | *"manages to be both repulsively sadistic and mundane ."* | NEGATIVE | NEGATIVE (0.9981) | POSITIVE (0.6929) | DistilBERT catches strong negative descriptors ("sadistic", "mundane"); BERT is tricked by "manages". |

---

## 7. Submission Artifacts

- **Scripts**:
  - `evaluate.py`: End-to-end evaluation, metric computation, and MLflow logging.
  - `register_champion.py`: Model Registry registration and alias assignment (`@champion`, `@challenger`).
  - `predict_from_mlflow.py`: Standalone inference loading models exclusively from MLflow registry alias URIs.
  - `hf_model_wrapper.py`: Custom PyFunc wrapper for Hugging Face pipelines with label normalization.
  - `generate_screenshots.py`: Utility script generating clean screenshots for visual inspection.
- **Datasets**:
  - `data/eval_inputs.csv`: 150 balanced SST-2 evaluation sentences.
  - `data/new_data.csv`: 8 fresh, unseen sentences for inference testing.
- **Outputs (`outputs/`)**:
  - `predictions_A_sentiment_distilbert-sst2.csv`
  - `predictions_B_sentiment_textattack-bert-sst2.csv`
  - `metrics_summary.json`
  - `confusion_matrix_comparison.png`
  - `metrics_comparison_barchart.png`
  - `latency_comparison.png`
  - `disagreement_analysis.csv`
  - `new_data_predictions.csv`
  - `new_data_predictions_challenger.csv`
- **Screenshots (`screenshots/`)**:
  - `screenshots/mlflow_compare_view.png`
  - `screenshots/registry_aliases.png`
  - `screenshots/predict_terminal_output.png`

---

## 8. Limitations and Declaration

- **Sample Size**: Evaluated on 150 balanced sentences from SST-2 validation data. While sufficient for statistical signal and rapid CPU testing, a larger set (1,000+ examples) would provide tighter confidence bounds.
- **Hardware Variation**: Latency measurements were benchmarked on local CPU execution; relative speedups may differ on GPU inference or under concurrent serving loads.
- **Academic Declaration**: This project was developed as part of the Skill Lab (MLOps with MLflow) curriculum. All code, MLflow logging logic, custom PyFunc wrappers, and analysis were written and verified by the student. AI coding assistance was used for initial template scaffolding and documentation formatting in accordance with course guidelines.
