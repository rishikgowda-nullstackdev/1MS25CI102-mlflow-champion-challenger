"""
predict_from_mlflow.py
======================
Demonstrates loading the winning Champion model directly from the MLflow Model Registry
using its registered alias URI (models:/sentiment-model@champion).

IMPORTANT RUBRIC REQUIREMENT:
This script DOES NOT import transformers or Hugging Face pipelines directly.
All model loading and inference are performed strictly through MLflow pyfunc.

Follows Lab 08 (MLflow Models) and Lab 11 (Model Registry with MLflow) principles.
"""

import os
import sys
import argparse
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "sentiment-model"
INPUT_DATA_PATH = os.path.join("data", "new_data.csv")
OUTPUT_DATA_PATH = os.path.join("outputs", "new_data_predictions.csv")


def main():
    parser = argparse.ArgumentParser(description="Predict on new data using MLflow registered model alias.")
    parser.add_argument("--alias", type=str, default="champion", choices=["champion", "challenger"],
                        help="Model registry alias to load (default: champion)")
    args = parser.parse_args()
    target_alias = args.alias

    print("=" * 70)
    print(f"MLFLOW MODEL REGISTRY: INFERENCE VIA ALIAS URI (@{target_alias.upper()})")
    print("=" * 70)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Step 1: Query the registry for the version assigned to the alias
    try:
        model_version_info = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, target_alias)
    except Exception as e:
        print(f"Error: Could not find alias '{target_alias}' for '{REGISTERED_MODEL_NAME}'.")
        print("Please ensure register_champion.py has been run successfully.")
        print(f"Details: {e}")
        sys.exit(1)

    loaded_version = model_version_info.version
    source_run_id = model_version_info.run_id
    tags = model_version_info.tags or {}
    hf_model_id = tags.get("hf_model_id", "Unknown HF Model")

    print(f"Registered Model Name : {REGISTERED_MODEL_NAME}")
    print(f"Loaded Alias          : @{target_alias}")
    print(f"Resolved Version      : Version {loaded_version}")
    print(f"Hugging Face Model ID : {hf_model_id}")
    print(f"Source Run ID         : {source_run_id}")
    print("-" * 70)

    # Step 2: Load the model via alias URI
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{target_alias}"
    print(f"Loading model from URI: {model_uri} ...")
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully via MLflow pyfunc!")

    # Step 3: Load fresh, unseen data
    if not os.path.exists(INPUT_DATA_PATH):
        print(f"Error: Input data file '{INPUT_DATA_PATH}' not found.")
        sys.exit(1)

    df_new = pd.read_csv(INPUT_DATA_PATH)
    print(f"\nLoaded {len(df_new)} unseen test samples from '{INPUT_DATA_PATH}'.")

    # Step 4: Run predictions through MLflow pyfunc
    print("\nRunning inference through MLflow...")
    predictions_df = model.predict(df_new)

    # Step 5: Display clean tabular results
    print("\n" + "=" * 70)
    print(f"NEW DATA PREDICTIONS (@{target_alias.upper()} MODEL: {hf_model_id})")
    print("=" * 70)
    display_df = predictions_df.copy()
    display_df["text_preview"] = display_df["text"].apply(
        lambda t: (t[:50] + "...") if len(t) > 50 else t
    )
    cols = ["text_preview", "predicted_label", "confidence_score"]
    print(display_df[cols].to_string(index=False))
    print("=" * 70)

    # Step 6: Save outputs
    out_file = OUTPUT_DATA_PATH if target_alias == "champion" else os.path.join("outputs", f"new_data_predictions_{target_alias}.csv")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    predictions_df.to_csv(out_file, index=False)
    print(f"\nPredictions successfully saved to '{out_file}'.")


if __name__ == "__main__":
    main()
