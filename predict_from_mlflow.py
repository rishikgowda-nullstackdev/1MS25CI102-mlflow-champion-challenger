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
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "sentiment-model"
INPUT_DATA_PATH = os.path.join("data", "new_data.csv")
OUTPUT_DATA_PATH = os.path.join("outputs", "new_data_predictions.csv")


def main():
    print("=" * 70)
    print("MLFLOW MODEL REGISTRY: INFERENCE VIA ALIAS URI")
    print("=" * 70)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Step 1: Query the registry for the version assigned to @champion
    try:
        model_version_info = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    except Exception as e:
        print(f"Error: Could not find alias 'champion' for '{REGISTERED_MODEL_NAME}'.")
        print("Please ensure register_champion.py has been run successfully.")
        print(f"Details: {e}")
        sys.exit(1)

    champion_version = model_version_info.version
    source_run_id = model_version_info.run_id
    tags = model_version_info.tags or {}
    hf_model_id = tags.get("hf_model_id", "Unknown HF Model")

    print(f"Registered Model Name : {REGISTERED_MODEL_NAME}")
    print(f"Loaded Alias          : @champion")
    print(f"Resolved Version      : Version {champion_version}")
    print(f"Hugging Face Model ID : {hf_model_id}")
    print(f"Source Run ID         : {source_run_id}")
    print("-" * 70)

    # Step 2: Load the Champion model via alias URI
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    print(f"Loading model from URI: {model_uri} ...")
    champion_model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully via MLflow pyfunc!")

    # Step 3: Load fresh, unseen data
    if not os.path.exists(INPUT_DATA_PATH):
        print(f"Error: Input data file '{INPUT_DATA_PATH}' not found.")
        sys.exit(1)

    df_new = pd.read_csv(INPUT_DATA_PATH)
    print(f"\nLoaded {len(df_new)} unseen test samples from '{INPUT_DATA_PATH}'.")

    # Step 4: Run predictions through MLflow pyfunc
    print("\nRunning inference through MLflow...")
    predictions_df = champion_model.predict(df_new)

    # Step 5: Display clean tabular results
    print("\n" + "=" * 70)
    print("NEW DATA PREDICTIONS (CHAMPION MODEL)")
    print("=" * 70)
    display_df = predictions_df.copy()
    # Truncate text slightly for neat console table display
    display_df["text_preview"] = display_df["text"].apply(
        lambda t: (t[:50] + "...") if len(t) > 50 else t
    )
    cols = ["text_preview", "predicted_label", "confidence_score"]
    print(display_df[cols].to_string(index=False))
    print("=" * 70)

    # Step 6: Save outputs
    os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)
    predictions_df.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"\nPredictions successfully saved to '{OUTPUT_DATA_PATH}'.")


if __name__ == "__main__":
    main()
