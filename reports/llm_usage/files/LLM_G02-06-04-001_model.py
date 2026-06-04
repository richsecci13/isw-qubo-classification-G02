import os
import time
import json
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

def train(
    classifier: str,
    reducedTrain_csv: str,
    target_column: str,
    model_path: str,
    metrics_json: str,
    seed: int = 42
):
    """
    Trains the specified binary classifier on the reduced dataset and saves the model.
    """
    # --- 1. Data Loading ---
    start_input_time = time.time()
    df = pd.read_csv(reducedTrain_csv)
    dataset_input_time = time.time() - start_input_time

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # --- 2. Classifier Selection ---
    # Convert input to lowercase to handle varying user inputs gracefully
    clf_name = classifier.lower().strip()
    
    if clf_name in ["random_forest", "randomforest"]:
        model = RandomForestClassifier(random_state=seed)
    elif clf_name in ["xgb", "xgboost"]:
        model = XGBClassifier(random_state=seed, eval_metric='logloss')
    elif clf_name in ["logistic_regression", "logisticregression"]:
        model = LogisticRegression(random_state=seed, max_iter=1000)
    else:
        raise ValueError(f"Unsupported classifier: {classifier}. Choose from: random_forest, xgboost, logistic_regression.")

    # --- 3. Model Training ---
    start_train_time = time.time()
    model.fit(X, y)
    training_time = time.time() - start_train_time

    # --- 4. Save Artifacts ---
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(metrics_json) or '.', exist_ok=True)

    # Save the trained model
    joblib.dump(model, model_path)

    # Calculate statistics for the JSON report
    n_samples = len(df)
    target_1_percentage = (y.sum() / n_samples) * 100

    stats = {
        "classifier": classifier,
        "seed": seed,
        "training_dataset": os.path.basename(reducedTrain_csv),
        "target_column": target_column,
        "model_path": os.path.basename(model_path),
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "target_1_percentage": round(target_1_percentage, 2),
        "dataset_input_time": round(dataset_input_time, 2),
        "training_time": round(training_time, 2)
    }

    with open(metrics_json, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Training completed for {classifier}. Model saved to {model_path}.")


def predict(
    reduced_Test_csv: str,
    target_column: str,
    model_path: str,
    predictions_csv: str,
    classif_stats_json: str
):
    """
    Loads a trained model, generates predictions on test data, and computes metrics.
    """
    # --- 1. Data Loading ---
    df = pd.read_csv(reduced_Test_csv)
    y_true = df[target_column]
    X = df.drop(columns=[target_column])

    # --- 2. Model Loading & Prediction ---
    model = joblib.load(model_path)
    
    # Get the class predictions and the probabilities for the positive class (1)
    y_pred = model.predict(X)
    
    # Handle edge case if a model doesn't support predict_proba natively
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = y_pred

    # --- 3. Save Predictions CSV ---
    os.makedirs(os.path.dirname(predictions_csv) or '.', exist_ok=True)
    
    # The output format explicitly requires: row_n, target, prediction, score
    pred_df = pd.DataFrame({
        "row_n": df.index,
        "target": y_true,
        "prediction": y_pred,
        "score": np.round(scores, 4) if 'np' in globals() else [round(s, 4) for s in scores]
    })
    pred_df.to_csv(predictions_csv, index=False)

    # --- 4. Calculate Statistics & Save JSON ---
    os.makedirs(os.path.dirname(classif_stats_json) or '.', exist_ok=True)

    n_samples = len(y_true)
    target_1_count = int(y_true.sum())
    target_1_percentage = (target_1_count / n_samples) * 100

    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, scores)
    
    # Ensure zero_division is handled gracefully
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    stats = {
        "classifier": type(model).__name__,
        "n_samples": n_samples,
        "target_1_count": target_1_count,
        "target_1_percentage": round(target_1_percentage, 2),
        "accuracy": accuracy,
        "class_0": {
            "precision": precision[0],
            "recall": recall[0],
            "f1": f1[0],
            "support": int(support[0])
        },
        "class_1": {
            "precision": precision[1],
            "recall": recall[1],
            "f1": f1[1],
            "support": int(support[1])
        },
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "labels": [1, 0],
            "matrix": cm.tolist()
        }
    }

    with open(classif_stats_json, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Prediction completed. Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training and Prediction CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train Subparser ---
    parser_train = subparsers.add_parser("train", help="Train a classifier")
    parser_train.add_argument("--classifier", required=True, help="Classifier to use (e.g., random_forest)")
    parser_train.add_argument("--in-reduced", required=True, help="Input reduced training dataset")
    parser_train.add_argument("--target", required=True, help="Target column name")
    parser_train.add_argument("--out-model", required=True, help="Path to save the trained model")
    parser_train.add_argument("--out-metrics", required=True, help="Path to save training metrics JSON")
    parser_train.add_argument("--seed", type=int, default=42, help="Random seed")

    # --- Predict Subparser ---
    parser_predict = subparsers.add_parser("predict", help="Generate predictions using a trained model")
    parser_predict.add_argument("--input-testset", required=True, help="Input reduced test dataset")
    parser_predict.add_argument("--target", required=True, help="Target column name")
    parser_predict.add_argument("--model", required=True, help="Path to the trained model file")
    parser_predict.add_argument("--out-predictions", required=True, help="Path to save predictions CSV")
    parser_predict.add_argument("--out-stats", required=True, help="Path to save classification stats JSON")

    args = parser.parse_args()

    # Route the command appropriately
    if args.command == "train":
        train(
            classifier=args.classifier,
            reducedTrain_csv=args.in_reduced,
            target_column=args.target,
            model_path=args.out_model,
            metrics_json=args.out_metrics,
            seed=args.seed
        )
    elif args.command == "predict":
        predict(
            reduced_Test_csv=args.input_testset,
            target_column=args.target,
            model_path=args.model,
            predictions_csv=args.out_predictions,
            classif_stats_json=args.out_stats
        )