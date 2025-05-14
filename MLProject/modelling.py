import os
import pandas as pd
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, log_loss, precision_score, recall_score, f1_score
)
from mlflow.models.signature import infer_signature
import warnings
warnings.filterwarnings("ignore")


def load_data(path: str):
    return pd.read_csv(path)

def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.01,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted')
    }

def main():
    # Kredensial dari environment
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    # Set URI dan autentikasi
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("Sleep_Disorder_Classification_XGBoost")

    # Load data
    df = load_data("clean_sleep_health_and_lifestyle_dataset.csv")
    X = df.drop(columns=["sleep_disorder"])
    y = df["sleep_disorder"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="xgb_fixed_run"):
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        # Logging parameter
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("learning_rate", 0.01)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.set_tag("stage", "production")
        mlflow.set_tag("model", "XGBClassifier")

        # ✨ Tambah signature
        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, model.predict(X_test))

        # Save model with signature
        mlflow.sklearn.log_model(
            model,
            artifact_path="xgb_model",
            input_example=input_example,
            signature=signature
        )

        print("[Run] Model trained and logged with fixed hyperparameters")

if __name__ == "__main__":
    main()
