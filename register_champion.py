"""
register_champion.py
====================
Registers candidate models in the MLflow Model Registry and assigns
the '@champion' and '@challenger' aliases based on the pre-defined decision rule.

Follows Lab 11 (Model Registry with MLflow) principles.

Decision Rule:
1. Primary Metric: Macro-F1 (Higher is better)
2. Improvement Margin: Challenger must beat Champion by at least 0.01 (+1.0 percentage point)
3. Latency Guardrail: Challenger average latency must not exceed 2.0x Champion latency
4. Outcome: If conditions are met, Challenger becomes Champion; otherwise Champion retains title.
"""

import sys
import mlflow
from mlflow.tracking import MlflowClient

# Connect to the tracking database
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"

MARGIN = 0.01
MAX_LATENCY_RATIO = 2.0


def main():
    print("=" * 70)
    print("MLFLOW MODEL REGISTRY: CHAMPION - CHALLENGER PROMOTION")
    print("=" * 70)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Retrieve experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"Error: Experiment '{EXPERIMENT_NAME}' not found. Please run evaluate.py first.")
        sys.exit(1)

    # Search for runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
    )

    run_a = None
    run_b = None

    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "")
        if "A_sentiment" in run_name and run_a is None:
            run_a = run
        elif "B_sentiment" in run_name and run_b is None:
            run_b = run

    if not run_a or not run_b:
        print("Error: Could not locate runs for Model A and Model B.")
        sys.exit(1)

    print(f"Located Run A: {run_a.info.run_id} ({run_a.data.tags.get('mlflow.runName')})")
    print(f"Located Run B: {run_b.info.run_id} ({run_b.data.tags.get('mlflow.runName')})")

    # Register Model A
    print(f"\nRegistering Model A under '{REGISTERED_MODEL_NAME}'...")
    mv_a = mlflow.register_model(
        model_uri=f"runs:/{run_a.info.run_id}/model",
        name=REGISTERED_MODEL_NAME,
        tags={
            "hf_model_id": run_a.data.params.get("hf_model_id", ""),
            "candidate_role": "Model A (Baseline)",
            "run_id": run_a.info.run_id,
        },
    )

    # Register Model B
    print(f"Registering Model B under '{REGISTERED_MODEL_NAME}'...")
    mv_b = mlflow.register_model(
        model_uri=f"runs:/{run_b.info.run_id}/model",
        name=REGISTERED_MODEL_NAME,
        tags={
            "hf_model_id": run_b.data.params.get("hf_model_id", ""),
            "candidate_role": "Model B (Challenger)",
            "run_id": run_b.info.run_id,
        },
    )

    # Extract metrics for decision rule
    f1_a = run_a.data.metrics.get("macro_f1", 0.0)
    f1_b = run_b.data.metrics.get("macro_f1", 0.0)
    lat_a = run_a.data.metrics.get("avg_latency_ms", 1.0)
    lat_b = run_b.data.metrics.get("avg_latency_ms", 1.0)

    delta_f1 = f1_b - f1_a
    latency_ratio = lat_b / lat_a if lat_a > 0 else 1.0

    print("\n--- DECISION RULE EVALUATION ---")
    print(f"Model A (DistilBERT): Macro-F1 = {f1_a:.4f}, Avg Latency = {lat_a:.2f} ms")
    print(f"Model B (BERT SST-2):  Macro-F1 = {f1_b:.4f}, Avg Latency = {lat_b:.2f} ms")
    print(f"Delta Macro-F1 (B - A): {delta_f1:+.4f} (Required Margin: >= +{MARGIN:.2f})")
    print(f"Latency Ratio (B / A):  {latency_ratio:.2f}x (Allowed Guardrail: <= {MAX_LATENCY_RATIO:.2f}x)")

    # Evaluate pre-declared rule
    meets_f1_margin = delta_f1 >= MARGIN
    meets_latency_guardrail = latency_ratio <= MAX_LATENCY_RATIO

    if meets_f1_margin and meets_latency_guardrail:
        winner = "Model B (Challenger)"
        champion_version = mv_b.version
        challenger_version = mv_a.version
        reason = f"Challenger outperformed Champion by {delta_f1 * 100:.2f}% (>= {MARGIN * 100:.1f}%) within latency guardrail."
    else:
        winner = "Model A (Champion)"
        champion_version = mv_a.version
        challenger_version = mv_b.version
        if not meets_f1_margin:
            reason = f"Challenger did not achieve the required +{MARGIN * 100:.1f}% Macro-F1 margin (actual delta: {delta_f1 * 100:+.2f}%)."
        else:
            reason = f"Challenger violated the latency guardrail ({latency_ratio:.2f}x > {MAX_LATENCY_RATIO:.2f}x)."

    print(f"\nDecision Result: {winner} is declared the CHAMPION!")
    print(f"Reason: {reason}")

    # Set aliases in MLflow Model Registry
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", champion_version)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", challenger_version)

    print("\n--- MODEL REGISTRY ALIASES UPDATED ---")
    print(f"Registered Model: '{REGISTERED_MODEL_NAME}'")
    print(f"  • Alias '@champion'   -> Version {champion_version}")
    print(f"  • Alias '@challenger' -> Version {challenger_version}")
    print("=" * 70)


if __name__ == "__main__":
    main()
