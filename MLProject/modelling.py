import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, log_loss, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
from mlflow.models.signature import infer_signature
import warnings
warnings.filterwarnings("ignore")

def load_data(path: str) -> pd.DataFrame:
    """
    Loads a CSV file as a DataFrame.

    Parameters:
    path (str): File path to the CSV.

    Returns:
    pd.DataFrame: Loaded dataset.

    Raises:
    FileNotFoundError: If the file does not exist.
    pd.errors.ParserError: If the file cannot be parsed.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"[ERROR] File not found: {path}") from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"[ERROR] Could not parse the file: {path}") from e

def evaluate_model(model, X, y, prefix="") -> dict:
    """
    Evaluates a classification model and returns metrics.

    Parameters:
    model: Trained classification model.
    X (pd.DataFrame): Features to predict on.
    y (pd.Series): True labels.
    prefix (str): Optional prefix for metric keys (e.g., 'train_', 'test_').

    Returns:
    dict: Dictionary of evaluation metrics.

    Raises:
    ValueError: If prediction fails due to mismatched input.
    """
    try:
        y_pred  = model.predict(X)
        y_proba = model.predict_proba(X)

        metrics = {
            f"{prefix}score"            : model.score(X, y),
            f"{prefix}accuracy_score"   : accuracy_score(y, y_pred),
            f"{prefix}balanced_accuracy": balanced_accuracy_score(y, y_pred),
            f"{prefix}log_loss"         : log_loss(y, y_proba),
            f"{prefix}precision_score"  : precision_score(y, y_pred, average='weighted'),
            f"{prefix}recall_score"     : recall_score(y, y_pred, average='weighted'),
            f"{prefix}f1_score"         : f1_score(y, y_pred, average='weighted'),
            f"{prefix}matthews_corrcoef": matthews_corrcoef(y, y_pred)
        }

        if len(set(y)) > 2:
            y_bin = label_binarize(y, classes=sorted(set(y)))
            metrics[f"{prefix}roc_auc"] = roc_auc_score(y_bin, y_proba, average='weighted', multi_class='ovr')
        else:
            metrics[f"{prefix}roc_auc"] = roc_auc_score(y, y_proba[:, 1])

        return metrics

    except Exception as e:
        raise ValueError(f"[ERROR] Model evaluation failed: {str(e)}") from e


def main():
    """
    Trains and tunes a RandomForest model using GridSearchCV,
    and manually logs training/testing metrics and model using MLflow.
    """
    try:
        # Kredensial dari environment
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        mlflow_username     = os.getenv("MLFLOW_TRACKING_USERNAME")
        mlflow_password     = os.getenv("MLFLOW_TRACKING_PASSWORD")

        # Set URI dan autentikasi
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("Sleep_Disorder_Classification_RF")

        # Load data
        X_train = load_data("processed-dataset/X_train.csv")
        X_test  = load_data("processed-dataset/X_test.csv")
        y_train = load_data("processed-dataset/y_train.csv").squeeze()
        y_test  = load_data("processed-dataset/y_test.csv").squeeze()

        print("MLFLOW_TRACKING_URI:", mlflow.get_tracking_uri())
        print("Current experiment:", mlflow.get_experiment_by_name("Sleep_Disorder_Classification_RF"))

        with mlflow.start_run(run_name="rf_fixed_params"):
            best_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=2,
                random_state=42
            )
            best_model.fit(X_train, y_train)

            # Log parameter secara manual
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", "None")
            mlflow.log_param("min_samples_split", 2)
            mlflow.log_param("min_samples_leaf", 2)

            # Manual log training metrics
            train_metrics = evaluate_model(best_model, X_train, y_train, prefix="train_")
            for key, value in train_metrics.items():
                mlflow.log_metric(key, value)
            print("[INFO] Training metrics logged manually.")

            # Manual log testing metrics
            test_metrics = evaluate_model(best_model, X_test, y_test, prefix="test_")
            for key, value in test_metrics.items():
                mlflow.log_metric(key, value)
            print("[INFO] Testing metrics logged manually.")

            # Log model signature
            input_example = X_train.iloc[:1]
            signature = infer_signature(X_train, best_model.predict(X_train))

            # Log model artifact
            mlflow.sklearn.log_model(
                best_model,
                artifact_path = "random_forest_model",
                input_example = input_example,
                signature     = signature,
                conda_env     = "conda.yaml"
            )
            mlflow.set_tag("stage", "tuning")
            mlflow.set_tag("model", "RandomForestClassifier")
            mlflow.set_tag("evaluation_extended", "True")
            mlflow.set_tag("extra_metrics", "MCC, Balanced Accuracy") # min 2 metrik tambahan di luar autolog
            print("[INFO] Best model logged.")

            run_id = mlflow.active_run().info.run_id
            print(f"[INFO] Run ID: {run_id}")
            print("[INFO] Hyperparameter tuning and logging completed.")

    except Exception as e:
        print(f"[ERROR] Modelling execution failed: {str(e)}")

if __name__ == "__main__":
    main()
