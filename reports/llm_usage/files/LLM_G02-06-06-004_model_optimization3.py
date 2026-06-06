import os
import time
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

def train(
    classifier: str,
    reducedTrain_csv: str,
    target_column: str,
    model_path: str,
    metrics_json: str,
    seed: int = 42
):
    """
    Trains the specified binary classifier on the reduced dataset.
    Incorporates Class Weighting, Hyperparameter Optimization, and Threshold Tuning.
    """
    # --- 1. Data Loading ---
    start_input_time = time.time()
    df = pd.read_csv(reducedTrain_csv)
    dataset_input_time = time.time() - start_input_time

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # --- 2. Calculate Class Imbalance ---
    num_class_0 = (y == 0).sum()
    num_class_1 = (y == 1).sum()
    scale_pos_w = float(num_class_0 / num_class_1) if num_class_1 > 0 else 1.0

    # --- 3. Classifier Setup & Hyperparameter Grids (Optimization #3) ---
    clf_name = classifier.lower().strip()
    
    if clf_name in ["random_forest", "randomforest"]:
        base_model = RandomForestClassifier(random_state=seed, class_weight='balanced')
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    elif clf_name in ["xgb", "xgboost"]:
        base_model = XGBClassifier(random_state=seed, eval_metric='logloss', scale_pos_weight=scale_pos_w)
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0]
        }
    elif clf_name in ["logistic_regression", "logisticregression"]:
        base_model = LogisticRegression(random_state=seed, max_iter=2000, class_weight='balanced')
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'liblinear']
        }
    else:
        raise ValueError(f"Unsupported classifier: {classifier}. Choose from: random_forest, xgboost, logistic_regression.")

    # Execute Randomized Search targeting the F1 score
    start_train_time = time.time()
    print(f"Starting Hyperparameter Optimization for {classifier}...")
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=10,        # Number of parameter settings that are sampled
        scoring='f1',     # Explicitly optimize for F1 score
        cv=3,             # 3-fold cross-validation
        random_state=seed,
        n_jobs=-1         # Use all available CPU cores
    )
    search.fit(X, y)
    
    # Extract the absolute best model found during the search
    model = search.best_estimator_
    training_time = time.time() - start_train_time

    # --- 4. Dynamic Threshold Tuning (Optimization #2) ---
    best_threshold = 0.5
    if hasattr(model, "predict_proba"):
        train_probs = model.predict_proba(X)[:, 1]
        thresholds = np.arange(0.1, 0.95, 0.05)
        best_f1 = 0
        
        for t in thresholds:
            preds_t = (train_probs >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y, preds_t, labels=[0, 1], zero_division=0)
            
            # Maximize F1 for class 1
            if f1[1] > best_f1:
                best_f1 = f1[1]
                best_threshold = t

    # --- 5. Save Artifacts ---
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(metrics_json) or '.', exist_ok=True)

    artifact = {
        "model": model,
        "optimal_threshold": float(best_threshold)
    }
    joblib.dump(artifact, model_path)

    n_samples = len(df)
    target_1_percentage = (num_class_1 / n_samples) * 100

    # Ensure numpy types are converted to native Python types for JSON serialization
    best_params_clean = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) 
                         for k, v in search.best_params_.items()}

    stats = {
        "classifier": classifier,
        "seed": seed,
        "training_dataset": os.path.basename(reducedTrain_csv),
        "target_column": target_column,
        "model_path": os.path.basename(model_path),
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "target_1_percentage": round(target_1_percentage, 2),
        "best_hyperparameters": best_params_clean,
        "optimal_threshold": round(best_threshold, 3),
        "dataset_input_time": round(dataset_input_time, 2),
        "training_time": round(training_time, 2)
    }

    with open(metrics_json, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Training completed. Best Params: {best_params_clean}. Optimal Threshold: {best_threshold:.2f}.")

def predict(
    reduced_Test_csv: str,
    target_column: str,
    model_path: str,
    predictions_csv: str,
    classif_stats_json: str
):
    """
    Loads a trained model, generates predictions on test data using the tuned threshold, 
    and computes metrics.
    """
    # --- 1. Data Loading ---
    df = pd.read_csv(reduced_Test_csv)
    y_true = df[target_column]
    X = df.drop(columns=[target_column])

    # --- 2. Model Loading & Prediction ---
    artifact = joblib.load(model_path)
    
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        threshold = artifact.get("optimal_threshold", 0.5)
    else:
        model = artifact
        threshold = 0.5
    
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
        y_pred = (scores >= threshold).astype(int)
    else:
        y_pred = model.predict(X)
        scores = y_pred

    # --- 3. Save Predictions CSV ---
    os.makedirs(os.path.dirname(predictions_csv) or '.', exist_ok=True)
    
    pred_df = pd.DataFrame({
        "row_n": df.index,
        "target": y_true,
        "prediction": y_pred,
        "score": np.round(scores, 4)
    })
    pred_df.to_csv(predictions_csv, index=False)

    # --- 4. Calculate Statistics & Save JSON ---
    os.makedirs(os.path.dirname(classif_stats_json) or '.', exist_ok=True)

    n_samples = len(y_true)
    target_1_count = int(y_true.sum())
    target_1_percentage = (target_1_count / n_samples) * 100

    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, scores)
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    stats = {
        "classifier": type(model).__name__,
        "n_samples": n_samples,
        "target_1_count": target_1_count,
        "target_1_percentage": round(target_1_percentage, 2),
        "applied_threshold": threshold,
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
            "labels": [0, 1],
            "matrix": cm.tolist()
        }
    }

    with open(classif_stats_json, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Prediction completed. Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training and Prediction CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser("train", help="Train a classifier")
    parser_train.add_argument("--classifier", required=True, help="Classifier to use (e.g., random_forest)")
    parser_train.add_argument("--in-reduced", required=True, help="Input reduced training dataset")
    parser_train.add_argument("--target", required=True, help="Target column name")
    parser_train.add_argument("--out-model", required=True, help="Path to save the trained model")
    parser_train.add_argument("--out-metrics", required=True, help="Path to save training metrics JSON")
    parser_train.add_argument("--seed", type=int, default=42, help="Random seed")

    parser_predict = subparsers.add_parser("predict", help="Generate predictions using a trained model")
    parser_predict.add_argument("--input-testset", required=True, help="Input reduced test dataset")
    parser_predict.add_argument("--target", required=True, help="Target column name")
    parser_predict.add_argument("--model", required=True, help="Path to the trained model file")
    parser_predict.add_argument("--out-predictions", required=True, help="Path to save predictions CSV")
    parser_predict.add_argument("--out-stats", required=True, help="Path to save classification stats JSON")

    args = parser.parse_args()

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