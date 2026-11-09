"""
hf_model_wrapper.py
===================
Custom MLflow PythonModel wrapper for Hugging Face Sentiment Analysis.
Follows Lab 09 (Customized Models with MLflow) principles.

This wrapper wraps a Hugging Face text-classification pipeline so that
it can be saved, logged, and loaded through MLflow pyfunc, returning
clean and standardized prediction DataFrames.
"""

import pandas as pd
import mlflow.pyfunc
from transformers import pipeline


class HuggingFaceSentimentModel(mlflow.pyfunc.PythonModel):
    """
    Beginner-friendly custom MLflow wrapper for Hugging Face Sentiment Analysis models.
    Handles label normalization across different model formats (e.g., POSITIVE vs LABEL_1).
    """

    def __init__(self, hf_model_id: str):
        self.hf_model_id = hf_model_id
        self.pipeline = None

    def __getstate__(self):
        """
        Keep serialized model artifact extremely small by excluding
        live in-memory pipeline objects during pickling.
        """
        state = self.__dict__.copy()
        state["pipeline"] = None
        return state

    def load_context(self, context=None):
        """
        Loads the Hugging Face pipeline when the model is restored via mlflow.pyfunc.load_model.
        """
        # If the model ID was provided in init, use it; otherwise read from artifacts if present
        model_id = self.hf_model_id
        self.pipeline = pipeline(
            task="sentiment-analysis",
            model=model_id,
            tokenizer=model_id,
            truncation=True,
            max_length=512,
        )

    def _normalize_prediction(self, raw_label: str) -> tuple[str, int]:
        """
        Normalizes label strings from different model outputs:
        - 'POSITIVE' or 'LABEL_1' -> ('POSITIVE', 1)
        - 'NEGATIVE' or 'LABEL_0' -> ('NEGATIVE', 0)
        """
        cleaned = raw_label.strip().upper()
        if cleaned in ["POSITIVE", "LABEL_1", "POS", "1"]:
            return "POSITIVE", 1
        elif cleaned in ["NEGATIVE", "LABEL_0", "NEG", "0"]:
            return "NEGATIVE", 0
        else:
            # Fallback for unexpected label formats
            return cleaned, -1

    def predict(self, context, model_input):
        """
        Run inference on the provided input texts.
        Accepts:
            - pandas DataFrame (with a 'text' column or single column)
            - pandas Series
            - list of strings
        Returns:
            pandas DataFrame with columns:
            ['text', 'predicted_label', 'label_id', 'confidence_score']
        """
        # Ensure pipeline is initialized
        if self.pipeline is None:
            self.load_context(context)

        # Convert input to a Python list of strings
        if isinstance(model_input, pd.DataFrame):
            if "text" in model_input.columns:
                texts = model_input["text"].astype(str).tolist()
            else:
                texts = model_input.iloc[:, 0].astype(str).tolist()
        elif isinstance(model_input, pd.Series):
            texts = model_input.astype(str).tolist()
        elif isinstance(model_input, (list, tuple)):
            texts = [str(t) for t in model_input]
        else:
            texts = [str(model_input)]

        # Run Hugging Face pipeline
        raw_outputs = self.pipeline(texts)

        # Format output cleanly
        results = []
        for text, out in zip(texts, raw_outputs):
            raw_label = out.get("label", "")
            score = float(out.get("score", 0.0))
            norm_label, label_id = self._normalize_prediction(raw_label)
            results.append({
                "text": text,
                "predicted_label": norm_label,
                "label_id": label_id,
                "confidence_score": round(score, 4),
            })

        return pd.DataFrame(results)
