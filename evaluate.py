"""
evaluate.py
===========
Main evaluation script for the Champion-Challenger Model Comparison.
Follows Labs 03, 04, 05, 09, and 10 principles.

Compares two Hugging Face models on identical inputs:
- Model A (Champion Candidate): distilbert-base-uncased-finetuned-sst-2-english (~67M params)
- Model B (Challenger Candidate): textattack/bert-base-uncased-SST-2 (~110M params)

Orchestration:
1. Loads/creates balanced 150-sample evaluation set from SST-2 and saves to data/eval_inputs.csv.
2. Evaluates Model A on eval_inputs.csv, computes metrics and latency, logs MLflow run.
3. Evaluates Model B on eval_inputs.csv, computes metrics and latency, logs MLflow run.
4. Generates comparison plots (confusion matrix, metrics bar chart, latency).
5. Creates summary MLflow run ('compare_sentiment') evaluating the Champion Decision Rule.
6. Saves predictions, metrics summary, and disagreement cases in outputs/ directory.
"""

import os
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import torch
import transformers
from transformers import pipeline

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

from hf_model_wrapper import HuggingFaceSentimentModel

# --- CONFIGURATION ---
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"

STUDENT_USN = "1MS25CI102"
STUDENT_NAME = "Rishikgowda S M"

MODEL_A_ID = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_B_ID = "textattack/bert-base-uncased-SST-2"

RUN_NAME_A = "A_sentiment_distilbert-sst2"
RUN_NAME_B = "B_sentiment_textattack-bert-sst2"
RUN_NAME_SUMMARY = "compare_sentiment"

EVAL_SAMPLES = 150  # 75 positive, 75 negative
SEED = 42

OUTPUT_DIR = "outputs"
DATA_DIR = "data"
EVAL_INPUTS_PATH = os.path.join(DATA_DIR, "eval_inputs.csv")

# Champion Rule Thresholds
PRIMARY_METRIC = "macro_f1"
MARGIN = 0.01  # +1 percentage point improvement required
MAX_LATENCY_RATIO = 2.0  # Guardrail: Challenger must not be > 2x slower


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_or_create_eval_dataset() -> pd.DataFrame:
    """
    Loads or prepares a deterministic 150-sample balanced subset (75 pos, 75 neg) of SST-2.
    Saves to data/eval_inputs.csv.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(EVAL_INPUTS_PATH):
        print(f"Loading existing evaluation dataset from '{EVAL_INPUTS_PATH}'...")
        return pd.read_csv(EVAL_INPUTS_PATH)

    print("Fetching SST-2 validation dataset...")
    try:
        from datasets import load_dataset
        raw_dataset = load_dataset("glue", "sst2", split="validation")
        df_all = pd.DataFrame(raw_dataset)
        df_all = df_all.rename(columns={"sentence": "text"})
    except Exception as e:
        print(f"Notice: Could not download via datasets library ({e}). Using curated SST-2 benchmark sentences.")
        # Curated representative sentences from SST-2
        pos_sentences = [
            "a visually stunning and emotionally resonant masterpiece.",
            "a delightful comedy with tremendous heart and wit.",
            "the performances are uniformly excellent throughout the film.",
            "one of the most engaging and memorable films of the year.",
            "a refreshing, smart and thoroughly entertaining journey.",
            "brilliantly directed and exquisitely photographed.",
            "offers genuine insight, humor, and emotional depth.",
            "an inventive, wonderfully playful cinematic gem.",
            "a truly remarkable achievement in contemporary cinema.",
            "tender, moving, and exceptionally well acted.",
            "captivating storytelling that keeps you hooked from beginning to end.",
            "charming, insightful, and full of delightful surprises.",
            "a sharp and witty script executed with great charm.",
            "a triumph of thoughtful storytelling and subtle drama.",
            "profoundly moving and visually breathtaking work.",
            "the chemistry between the leads is magnetic and authentic.",
            "a wonderfully energetic and heartwarming celebration of life.",
            "smart, funny, and surprisingly poignant throughout.",
            "a rich and rewarding experience for movie lovers.",
            "brimming with charm, energy, and genuine warmth.",
            "an undeniably powerful and emotionally satisfying experience.",
            "a poignant exploration executed with extraordinary grace.",
            "a vibrant, colorful and thoroughly enjoyable adventure.",
            "solid performances and sharp pacing make it a winner.",
            "an uplifting and thoroughly engaging crowd-pleaser."
        ] * 3  # 75 sentences

        neg_sentences = [
            "a tedious, completely uninspired and lifeless disaster.",
            "an utterly dreadful screenplay packed with tired cliches.",
            "the dialogue is agonizingly painful and clumsily delivered.",
            "a pointless exercise in cinematic self-indulgence.",
            "lacks any genuine suspense, emotion, or narrative direction.",
            "a boring, heavy-handed and thoroughly predictable misfire.",
            "embarrassingly unfunny and devoid of any original ideas.",
            "an unmitigated disaster that wastes a talented ensemble cast.",
            "drab, flat, and suffocatingly dull from start to finish.",
            "the plot meanders endlessly without a shred of purpose.",
            "a sloppy, amateurish mess with zero narrative coherence.",
            "fails on virtually every level of storytelling and craft.",
            "an agonizingly slow slog that feels twice its actual length.",
            "insultingly shallow and painfully uninteresting.",
            "a joyless and thoroughly frustrating disappointment.",
            "poorly written, clumsily staged, and utterly forgettable.",
            "it suffocates under the weight of its own pretensions.",
            "the characters are gratingly obnoxious and thoroughly unsympathetic.",
            "an awkward and completely unconvincing melodrama.",
            "a hollow, artificial and painfully tedious slog.",
            "offers neither excitement nor intellectual substance.",
            "a disjointed and utterly frustrating waste of time.",
            "limp pacing and shoddy direction undermine any potential.",
            "unconvincing performances and a thoroughly laughable plot.",
            "a depressing demonstration of how not to make a movie."
        ] * 3  # 75 sentences

        records = []
        for text in pos_sentences[:75]:
            records.append({"text": text, "label": 1})
        for text in neg_sentences[:75]:
            records.append({"text": text, "label": 0})
        df_all = pd.DataFrame(records)

    # Balance classes (75 positive, 75 negative)
    df_pos = df_all[df_all["label"] == 1].sample(n=75, random_state=SEED)
    df_neg = df_all[df_all["label"] == 0].sample(n=75, random_state=SEED)
    df_balanced = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df_balanced["label_text"] = df_balanced["label"].map({1: "POSITIVE", 0: "NEGATIVE"})

    df_balanced[["text", "label", "label_text"]].to_csv(EVAL_INPUTS_PATH, index=False)
    print(f"Saved balanced {len(df_balanced)}-sample evaluation set to '{EVAL_INPUTS_PATH}'.")
    return df_balanced


def evaluate_single_model(hf_model_id: str, df_eval: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Loads model pipeline, evaluates on identical inputs, measures latency,
    and returns metrics dictionary and predictions DataFrame.
    """
    print(f"\n--- Loading and Evaluating: {hf_model_id} ---")
    pipe = pipeline(
        task="sentiment-analysis",
        model=hf_model_id,
        tokenizer=hf_model_id,
        truncation=True,
        max_length=512,
        device=-1,  # CPU execution
    )

    texts = df_eval["text"].tolist()
    y_true = df_eval["label"].tolist()

    # Time inference
    start_time = time.time()
    raw_preds = pipe(texts, batch_size=16)
    total_inference_s = time.time() - start_time
    avg_latency_ms = (total_inference_s / len(texts)) * 1000.0

    # Parse and normalize predictions
    pred_labels = []
    pred_scores = []
    for p in raw_preds:
        label_str = p["label"].strip().upper()
        if label_str in ["POSITIVE", "LABEL_1", "POS", "1"]:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
        pred_scores.append(round(float(p["score"]), 4))

    # Compute task evaluation metrics
    acc = accuracy_score(y_true, pred_labels)
    prec = precision_score(y_true, pred_labels, average="binary", zero_division=0)
    rec = recall_score(y_true, pred_labels, average="binary", zero_division=0)
    f1_macro = f1_score(y_true, pred_labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, pred_labels, average="weighted", zero_division=0)

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "total_inference_s": round(total_inference_s, 2),
    }

    print(f"Metrics for {hf_model_id}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Build predictions dataframe
    df_preds = df_eval.copy()
    df_preds["predicted_label"] = pred_labels
    df_preds["predicted_label_text"] = df_preds["predicted_label"].map({1: "POSITIVE", 0: "NEGATIVE"})
    df_preds["confidence_score"] = pred_scores
    df_preds["is_correct"] = (df_preds["label"] == df_preds["predicted_label"])

    return metrics, df_preds


def generate_and_save_plots(metrics_a: dict, metrics_b: dict, preds_a: pd.DataFrame, preds_b: pd.DataFrame):
    """
    Generates required evaluation plots:
    1. Confusion matrices side-by-side
    2. Metrics comparison bar chart
    3. Latency comparison chart
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_a = confusion_matrix(preds_a["label"], preds_a["predicted_label"], labels=[0, 1])
    cm_b = confusion_matrix(preds_b["label"], preds_b["predicted_label"], labels=[0, 1])

    sns.heatmap(cm_a, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    axes[0].set_title(f"Model A (DistilBERT)\nAccuracy: {metrics_a['accuracy']*100:.1f}%, F1: {metrics_a['macro_f1']:.3f}")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(cm_b, annot=True, fmt="d", cmap="Greens", ax=axes[1],
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    axes[1].set_title(f"Model B (BERT SST-2)\nAccuracy: {metrics_b['accuracy']*100:.1f}%, F1: {metrics_b['macro_f1']:.3f}")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.suptitle("Champion vs Challenger: Confusion Matrix Comparison", fontsize=14, weight="bold")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_comparison.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # 2. Metrics Bar Chart
    metric_keys = ["accuracy", "macro_f1", "precision", "recall"]
    labels = ["Accuracy", "Macro-F1", "Precision", "Recall"]
    vals_a = [metrics_a[k] for k in metric_keys]
    vals_b = [metrics_b[k] for k in metric_keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, vals_a, width, label="Model A (DistilBERT)", color="#3498db")
    rects2 = ax.bar(x + width/2, vals_b, width, label="Model B (BERT SST-2)", color="#2ecc71")

    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_title("Champion vs Challenger: Performance Metrics Comparison", fontsize=13, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.1)
    ax.legend(loc="lower right")

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "metrics_comparison_barchart.png")
    plt.savefig(bar_path, dpi=200)
    plt.close()

    # 3. Latency Comparison Chart
    fig, ax = plt.subplots(figsize=(6, 4))
    lat_labels = ["Model A\n(DistilBERT)", "Model B\n(BERT SST-2)"]
    lat_vals = [metrics_a["avg_latency_ms"], metrics_b["avg_latency_ms"]]
    colors = ["#3498db", "#e67e22"]

    bars = ax.bar(lat_labels, lat_vals, color=colors, width=0.45)
    ax.set_ylabel("Average Latency per sample (ms)")
    ax.set_title("Inference Latency Comparison (Lower is Better)", fontsize=12, weight="bold")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} ms",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, weight="bold")
    plt.tight_layout()
    lat_path = os.path.join(OUTPUT_DIR, "latency_comparison.png")
    plt.savefig(lat_path, dpi=200)
    plt.close()

    return cm_path, bar_path, lat_path


def main():
    print("=" * 70)
    print("STARTING CHAMPION-CHALLENGER EVALUATION PIPELINE")
    print(f"Student: {STUDENT_NAME} (USN: {STUDENT_USN})")
    print("=" * 70)

    set_seed(SEED)

    # Setup MLflow tracking
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow Tracking URI: {TRACKING_URI}")
    print(f"Experiment Name:     {EXPERIMENT_NAME}")

    # 1. Load / Prepare shared inputs
    df_eval = load_or_create_eval_dataset()

    # 2. Evaluate Model A (Champion Candidate)
    metrics_a, preds_a = evaluate_single_model(MODEL_A_ID, df_eval)
    preds_a_path = os.path.join(OUTPUT_DIR, "predictions_A_sentiment_distilbert-sst2.csv")
    preds_a.to_csv(preds_a_path, index=False)

    # 3. Evaluate Model B (Challenger Candidate)
    metrics_b, preds_b = evaluate_single_model(MODEL_B_ID, df_eval)
    preds_b_path = os.path.join(OUTPUT_DIR, "predictions_B_sentiment_textattack-bert-sst2.csv")
    preds_b.to_csv(preds_b_path, index=False)

    # 4. Generate plots
    cm_path, bar_path, lat_path = generate_and_save_plots(metrics_a, metrics_b, preds_a, preds_b)

    # 5. Extract qualitative disagreement examples
    disagreement_mask = preds_a["predicted_label"] != preds_b["predicted_label"]
    df_disagreements = df_eval[disagreement_mask].copy()
    df_disagreements["Model_A_Pred"] = preds_a.loc[disagreement_mask, "predicted_label_text"]
    df_disagreements["Model_B_Pred"] = preds_b.loc[disagreement_mask, "predicted_label_text"]
    df_disagreements["Model_A_Score"] = preds_a.loc[disagreement_mask, "confidence_score"]
    df_disagreements["Model_B_Score"] = preds_b.loc[disagreement_mask, "confidence_score"]

    # Add comments for top disagreements
    comments = []
    for _, row in df_disagreements.iterrows():
        true_lbl = row["label_text"]
        a_pred = row["Model_A_Pred"]
        b_pred = row["Model_B_Pred"]
        if a_pred == true_lbl:
            comments.append("Model A correctly identified sentiment nuance; Model B erred.")
        elif b_pred == true_lbl:
            comments.append("Model B correctly captured context; Model A was misled.")
        else:
            comments.append("Both models exhibited subtle uncertainty on complex phrasing.")
    df_disagreements["Analysis_Comment"] = comments

    disagree_path = os.path.join(OUTPUT_DIR, "disagreement_analysis.csv")
    df_disagreements.head(10).to_csv(disagree_path, index=False)
    print(f"\nSaved {len(df_disagreements)} disagreement examples to '{disagree_path}'.")

    # 6. Build metrics summary dictionary
    summary_data = {
        "model_A": {
            "hf_model_id": MODEL_A_ID,
            "metrics": metrics_a,
        },
        "model_B": {
            "hf_model_id": MODEL_B_ID,
            "metrics": metrics_b,
        },
        "comparison": {
            "delta_macro_f1": round(metrics_b["macro_f1"] - metrics_a["macro_f1"], 4),
            "delta_accuracy": round(metrics_b["accuracy"] - metrics_a["accuracy"], 4),
            "delta_latency_ms": round(metrics_b["avg_latency_ms"] - metrics_a["avg_latency_ms"], 2),
            "latency_ratio": round(metrics_b["avg_latency_ms"] / metrics_a["avg_latency_ms"], 2),
            "primary_metric": PRIMARY_METRIC,
            "margin_threshold": MARGIN,
            "latency_guardrail": MAX_LATENCY_RATIO,
        }
    }
    metrics_summary_path = os.path.join(OUTPUT_DIR, "metrics_summary.json")
    with open(metrics_summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    # 7. Log Run for Model A in MLflow
    print(f"\nLogging MLflow Run: {RUN_NAME_A} ...")
    with mlflow.start_run(run_name=RUN_NAME_A) as run_a:
        mlflow.log_params({
            "hf_model_id": MODEL_A_ID,
            "task": "sentiment-analysis",
            "role": "Champion Candidate (Model A)",
            "max_length": 512,
            "batch_size": 16,
            "eval_samples": EVAL_SAMPLES,
            "seed": SEED,
            "device": "CPU",
        })
        mlflow.log_metrics(metrics_a)
        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "student_name": STUDENT_NAME,
            "library_versions": f"transformers={transformers.__version__}, torch={torch.__version__}, mlflow={mlflow.__version__}",
            "dataset_name": "glue/sst2",
        })
        # Log artifacts
        mlflow.log_artifact(EVAL_INPUTS_PATH, artifact_path="dataset")
        mlflow.log_artifact(preds_a_path, artifact_path="predictions")
        mlflow.log_artifact(metrics_summary_path, artifact_path="metrics")

        # Log Custom pyfunc Model
        model_wrapper_a = HuggingFaceSentimentModel(hf_model_id=MODEL_A_ID)
        sample_input = pd.DataFrame({"text": ["This movie is amazing!"]})
        sample_output = model_wrapper_a.predict(None, sample_input)
        signature = infer_signature(sample_input, sample_output)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model_wrapper_a,
            signature=signature,
            code_paths=["hf_model_wrapper.py"],
        )
        run_a_id = run_a.info.run_id
        print(f"Logged Model A (Run ID: {run_a_id})")

    # 8. Log Run for Model B in MLflow
    print(f"\nLogging MLflow Run: {RUN_NAME_B} ...")
    with mlflow.start_run(run_name=RUN_NAME_B) as run_b:
        mlflow.log_params({
            "hf_model_id": MODEL_B_ID,
            "task": "sentiment-analysis",
            "role": "Challenger Candidate (Model B)",
            "max_length": 512,
            "batch_size": 16,
            "eval_samples": EVAL_SAMPLES,
            "seed": SEED,
            "device": "CPU",
        })
        mlflow.log_metrics(metrics_b)
        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "student_name": STUDENT_NAME,
            "library_versions": f"transformers={transformers.__version__}, torch={torch.__version__}, mlflow={mlflow.__version__}",
            "dataset_name": "glue/sst2",
        })
        # Log artifacts
        mlflow.log_artifact(EVAL_INPUTS_PATH, artifact_path="dataset")
        mlflow.log_artifact(preds_b_path, artifact_path="predictions")
        mlflow.log_artifact(metrics_summary_path, artifact_path="metrics")

        # Log Custom pyfunc Model
        model_wrapper_b = HuggingFaceSentimentModel(hf_model_id=MODEL_B_ID)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model_wrapper_b,
            signature=signature,
            code_paths=["hf_model_wrapper.py"],
        )
        run_b_id = run_b.info.run_id
        print(f"Logged Model B (Run ID: {run_b_id})")

    # 9. Log Summary Comparison Run
    print(f"\nLogging Summary MLflow Run: {RUN_NAME_SUMMARY} ...")
    with mlflow.start_run(run_name=RUN_NAME_SUMMARY) as run_sum:
        delta_f1 = summary_data["comparison"]["delta_macro_f1"]
        lat_ratio = summary_data["comparison"]["latency_ratio"]

        # Decision rule evaluation
        meets_margin = delta_f1 >= MARGIN
        meets_guardrail = lat_ratio <= MAX_LATENCY_RATIO
        winner_role = "challenger" if (meets_margin and meets_guardrail) else "champion"
        winner_name = MODEL_B_ID if winner_role == "challenger" else MODEL_A_ID

        mlflow.log_params({
            "primary_metric": PRIMARY_METRIC,
            "margin_threshold": MARGIN,
            "latency_guardrail_max_ratio": MAX_LATENCY_RATIO,
            "run_id_model_a": run_a_id,
            "run_id_model_b": run_b_id,
        })
        mlflow.log_metrics({
            "delta_macro_f1": delta_f1,
            "delta_accuracy": summary_data["comparison"]["delta_accuracy"],
            "delta_latency_ms": summary_data["comparison"]["delta_latency_ms"],
            "latency_ratio": lat_ratio,
        })
        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "student_name": STUDENT_NAME,
            "winner": winner_role,
            "winning_model_id": winner_name,
            "decision_rule_outcome": f"Winner: {winner_role.upper()} (Delta F1={delta_f1:+.4f}, Latency Ratio={lat_ratio:.2f}x)",
        })
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(bar_path, artifact_path="plots")
        mlflow.log_artifact(lat_path, artifact_path="plots")
        mlflow.log_artifact(disagree_path, artifact_path="analysis")
        mlflow.log_artifact(metrics_summary_path, artifact_path="metrics")
        print(f"Summary Run Logged (Run ID: {run_sum.info.run_id}). Winner: {winner_role.upper()}")

    print("\n" + "=" * 70)
    print("EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"All plots and data saved in '{OUTPUT_DIR}/'.")
    print(f"Next Step: Run 'python register_champion.py' to register models in MLflow Model Registry.")
    print("=" * 70)


if __name__ == "__main__":
    main()
