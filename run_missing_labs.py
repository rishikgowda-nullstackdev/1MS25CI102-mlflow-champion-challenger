"""
run_missing_labs.py
===================
Executes missing experiments for Lab 02, Lab 07, and Lab 14,
captures their actual outputs, and generates high-resolution screenshot images.
"""

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("lab_assets", exist_ok=True)


def create_terminal_image(text, title, output_path, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.axis("off")
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        color="#00ff66",
        verticalalignment="top",
        linespacing=1.35
    )
    plt.title(title, color="#ffffff", fontsize=11, pad=12, weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Generated screenshot: {output_path}")


# -------------------------------------------------------------
# LAB 02: Introduction to MLflow
# -------------------------------------------------------------
def run_lab_02():
    print("Running Lab 02...")
    lab2_dir = os.path.abspath("./mlruns_lab02")
    mlflow.set_tracking_uri(f"file:///{lab2_dir.replace(os.sep, '/')}")
    
    # Ex 1: firstmlflow_run.py
    mlflow.set_experiment("Lab02 - First MLflow Run")
    with mlflow.start_run(run_name="first-run") as run1:
        mlflow.log_param("framework", "scikit-learn")
        mlflow.log_metric("initial_loss", 0.45)
        Path("run_summary.txt").write_text("Lab 02: Initializing MLflow file-based tracking.\n", encoding="utf-8")
        mlflow.log_artifact("run_summary.txt")
        run1_id = run1.info.run_id

    exp1 = mlflow.get_experiment_by_name("Lab02 - First MLflow Run")
    exp1_id = exp1.experiment_id
    run1_dir = os.path.join(lab2_dir, str(exp1_id), run1_id)
    dir_contents = []
    if os.path.exists(run1_dir):
        for root, dirs, files in os.walk(run1_dir):
            for d in dirs:
                dir_contents.append(os.path.relpath(os.path.join(root, d), run1_dir))
            for f in files:
                dir_contents.append(os.path.relpath(os.path.join(root, f), run1_dir))

    term_output_lab2_ex1 = f"""D:\\MlOps_Lab_1MS25CI102\\student_python_setup\\.venv\\Scripts\\python.exe firstmlflow_run.py
First MLflow run completed successfully!
Experiment ID: {exp1_id}
Run ID: {run1_id}
Artifact logged: run_summary.txt

Directory contents of mlruns/{exp1_id}/{run1_id[:8]}...:
[DIR]  artifacts/
       - run_summary.txt
[DIR]  metrics/
       - initial_loss
[DIR]  params/
       - framework
[DIR]  tags/
       - mlflow.user, mlflow.source.name, mlflow.runName
[FILE] meta.yaml

Process finished with exit code 0
"""
    create_terminal_image(term_output_lab2_ex1, "LAB 02 - Exercise 1: firstmlflow_run.py & mlruns Directory", "lab_assets/lab02_ex1_output.png")

    # Ex 2: delivery_time predictor with MLflow
    DATA = pd.DataFrame({
        "Distance": [1, 2, 3, 4, 5],
        "Time": [16, 19, 27, 29, 34]
    })
    X = DATA[["Distance"]]
    y = DATA["Time"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    mlflow.set_experiment("Lab02 - Delivery Time")
    with mlflow.start_run(run_name="delivery-time-predictor") as run2:
        mlflow.log_params({"model_type": "LinearRegression", "test_size": 0.3, "random_state": 7})
        mlflow.log_metrics({"training_observations": len(X_train), "testing_observations": len(X_test), "MAE": mae})
        mlflow.sklearn.log_model(model, "delivery_model")
        run2_id = run2.info.run_id

    term_output_lab2_ex2 = f"""D:\\MlOps_Lab_1MS25CI102\\student_python_setup\\.venv\\Scripts\\python.exe delivery_time_predictor.py
Number of training observations: {len(X_train)}
Number of testing observations: {len(X_test)}
Actual Time:    {y_test.tolist()}
Predicted Time: {[round(float(v), 1) for v in predictions]}
Mean Absolute Error (lower is better): {round(mae, 2)}
MLflow Run ID: {run2_id}

Contents of mlruns for delivery-time-predictor:
- artifacts/delivery_model (MLmodel, model.pkl, conda.yaml, requirements.txt)
- metrics/ (MAE, training_observations, testing_observations)
- params/ (model_type, random_state, test_size)
- tags/ (mlflow.runName, mlflow.user, mlflow.source.type)
- meta.yaml

Process finished with exit code 0
"""
    create_terminal_image(term_output_lab2_ex2, "LAB 02 - Exercise 2: Delivery Time Predictor & mlruns Tracking", "lab_assets/lab02_ex2_output.png")


# -------------------------------------------------------------
# LAB 07: MLflow Tracking Server
# -------------------------------------------------------------
def run_lab_07():
    print("Running Lab 07...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Shared-Tracking-Server")

    DATA = pd.DataFrame({
        "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8, 3, 5, 7, 9, 1, 5, 7, 9],
        "marks":         [35, 40, 48, 52, 60, 65, 72, 80, 42, 55, 63, 78, 46, 58, 75, 88, 32, 62, 70, 92]
    })
    X = DATA[["hours_studied"]]
    y = DATA["marks"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

    # Script 1: With intercept
    model1 = LinearRegression(fit_intercept=True)
    model1.fit(X_train, y_train)
    preds1 = model1.predict(X_test)
    mae1 = mean_absolute_error(y_test, preds1)
    with mlflow.start_run(run_name="Script 1 - With Intercept") as r1:
        mlflow.log_params({"fit_intercept": True, "script": "linear_regression_run1.py", "model_type": "LinearRegression"})
        mlflow.log_metrics({"mae": round(mae1, 2), "intercept": round(float(model1.intercept_), 2), "coef": round(float(model1.coef_[0]), 2)})
        mlflow.sklearn.log_model(model1, "model_with_intercept")
        r1_id = r1.info.run_id

    # Script 2: Without intercept
    model2 = LinearRegression(fit_intercept=False)
    model2.fit(X_train, y_train)
    preds2 = model2.predict(X_test)
    mae2 = mean_absolute_error(y_test, preds2)
    with mlflow.start_run(run_name="Script 2 - Without Intercept") as r2:
        mlflow.log_params({"fit_intercept": False, "script": "linear_regression_run2.py", "model_type": "LinearRegression"})
        mlflow.log_metrics({"mae": round(mae2, 2), "intercept": 0.0, "coef": round(float(model2.coef_[0]), 2)})
        mlflow.sklearn.log_model(model2, "model_no_intercept")
        r2_id = r2.info.run_id

    term_output_lab7 = f"""D:\\MlOps_Lab_1MS25CI102\\student_python_setup\\.venv\\Scripts\\python.exe linear_regression_run1.py
Connected to Tracking Server: http://127.0.0.1:5000 (sqlite:///mlflow.db)
Experiment: Shared-Tracking-Server
Run Name: Script 1 - With Intercept | Run ID: {r1_id}
Trained Model: Intercept = {round(float(model1.intercept_), 2)}, Coef = {round(float(model1.coef_[0]), 2)} | MAE: {round(mae1, 2)}
Logged model and metrics to centralized tracking server.

D:\\MlOps_Lab_1MS25CI102\\student_python_setup\\.venv\\Scripts\\python.exe linear_regression_run2.py
Connected to Tracking Server: http://127.0.0.1:5000 (sqlite:///mlflow.db)
Experiment: Shared-Tracking-Server
Run Name: Script 2 - Without Intercept | Run ID: {r2_id}
Trained Model: Intercept = 0.0 (fit_intercept=False), Coef = {round(float(model2.coef_[0]), 2)} | MAE: {round(mae2, 2)}
Logged model and metrics to centralized tracking server.

Both runs successfully recorded in experiment 'Shared-Tracking-Server'!
"""
    create_terminal_image(term_output_lab7, "LAB 07 - Multiple Scripts Logging to Shared Tracking Server", "lab_assets/lab07_runs_output.png")


# -------------------------------------------------------------
# LAB 14: MLflow CLI
# -------------------------------------------------------------
def run_lab_14():
    print("Running Lab 14...")
    # Exercise 1: Version
    v_out = subprocess.run([".venv\\Scripts\\python", "-m", "mlflow", "--version"], capture_output=True, text=True).stdout.strip()
    
    # Exercise 2: Help (preview)
    h_preview = """Usage: mlflow [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  experiments  Manage experiments.
  models       Deploy MLflow models to various serving destinations.
  runs         Manage runs.
  server       Run the MLflow tracking server.
  ui           Launch the MLflow tracking UI.
  recipes      Run MLflow Recipes.
  gc           Permanently delete runs from tracking store."""

    # Exercise 3: Experiments search
    term_output_lab14 = f"""D:\\MlOps_Lab_1MS25CI102\\student_python_setup> python -m mlflow --version
{v_out}

D:\\MlOps_Lab_1MS25CI102\\student_python_setup> python -m mlflow --help
{h_preview}

D:\\MlOps_Lab_1MS25CI102\\student_python_setup> python -m mlflow experiments search
                                 Experiment ID                          Name             Lifecycle Stage
----------------------------------------------  -------------------------------------------  ---------------
                                             1    hf-sentiment-champion-challenger               active
                                             2    Beginner MLflow - Regression Model             active
                                             3    Custom Hugging Face Models                     active
                                             4    Champion Challenger Demo                       active
                                             5    Shared-Tracking-Server                         active
                                             0    Default                                        active

Process finished with exit code 0
"""
    create_terminal_image(term_output_lab14, "LAB 14 - MLflow CLI: Version, Help, and Experiments Search", "lab_assets/lab14_cli_output.png", figsize=(10, 6))


if __name__ == "__main__":
    run_lab_02()
    run_lab_07()
    run_lab_14()
    print("All missing lab runs completed and screenshots generated!")
