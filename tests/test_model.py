import os
import json
import pytest
import pandas as pd
from src.qubo_project.model import train, predict

# Hardcoded path to the physical sample dataset as per specifications
SAMPLE_DATA_PATH = os.path.join("data", "sample_test_dataset.csv")
TARGET_COLUMN = "target"

@pytest.fixture
def phase3_4_setup(tmp_path):
    """
    Reads the physical sample dataset and creates mock 'reduced' training 
    and testing datasets. This isolates the model tests from the heavy 
    QUBO optimization while still relying on real physical data structure.
    """
    if not os.path.exists(SAMPLE_DATA_PATH):
        pytest.skip(f"Physical test file missing: {SAMPLE_DATA_PATH}. Please ensure it exists.")

    # Load the physical sample data
    df = pd.read_csv(SAMPLE_DATA_PATH)
    
    # Ensure we only use numeric columns to simulate the output of Phase 1 & 2
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if TARGET_COLUMN in numeric_cols:
        numeric_cols.remove(TARGET_COLUMN)
    
    # Take a small subset of features (e.g., first 3) + target to simulate Phase 2 output
    selected_features = numeric_cols[:3]
    final_cols = selected_features + [TARGET_COLUMN]
    
    # Fill any remaining NaNs just in case to ensure model training doesn't fail on raw sample data
    df_reduced = df[final_cols].fillna(0) 
    
    # Split into train and test
    split_idx = int(len(df_reduced) * 0.7)
    df_train = df_reduced.iloc[:split_idx]
    df_test = df_reduced.iloc[split_idx:]
    
    train_csv = str(tmp_path / "reduced_train.csv")
    test_csv = str(tmp_path / "reduced_test.csv")
    
    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)
    
    return train_csv, test_csv, tmp_path

def test_model_training_produces_saved_model(phase3_4_setup):
    """
    Test 6: Verify that the training produces a saved model file.
    """
    train_csv, _, tmp_path = phase3_4_setup
    model_path = str(tmp_path / "model.joblib")
    metrics_json = str(tmp_path / "training_metrics.json")
    
    # Run Phase 3 Training (using random_forest as mandated by specs)
    train(
        classifier="random_forest",
        reducedTrain_csv=train_csv,
        target_column=TARGET_COLUMN,
        model_path=model_path,
        metrics_json=metrics_json,
        seed=42
    )
    
    # Assert the model file was created
    assert os.path.exists(model_path), "Trained model file was not saved to disk."
    
    # Assert the model file actually contains data
    assert os.path.getsize(model_path) > 0, "Trained model file was created but is empty."
    
    # Verify the metrics JSON was also created
    assert os.path.exists(metrics_json), "Training metrics JSON was not saved."

def test_model_prediction_csv_columns(phase3_4_setup):
    """
    Test 7: Verify that the prediction produces a CSV file with the requested columns.
    """
    train_csv, test_csv, tmp_path = phase3_4_setup
    model_path = str(tmp_path / "model.joblib")
    metrics_json = str(tmp_path / "training_metrics.json")
    predictions_csv = str(tmp_path / "predictions.csv")
    stats_json = str(tmp_path / "classification_stats.json")
    
    # Step A: Train the model so we have an artifact to predict with
    train(
        classifier="random_forest",
        reducedTrain_csv=train_csv,
        target_column=TARGET_COLUMN,
        model_path=model_path,
        metrics_json=metrics_json,
        seed=42
    )
    
    # Step B: Run Phase 4 Prediction
    predict(
        reduced_Test_csv=test_csv,
        target_column=TARGET_COLUMN,
        model_path=model_path,
        predictions_csv=predictions_csv,
        classif_stats_json=stats_json
    )
    
    # Assert the predictions CSV was created
    assert os.path.exists(predictions_csv), "Predictions CSV was not saved to disk."
    
    # Load the predictions and verify the exact columns requested in the specifications
    df_preds = pd.read_csv(predictions_csv)
    expected_columns = ["row_n", "target", "prediction", "score"]
    
    assert list(df_preds.columns) == expected_columns, (
        f"Prediction CSV columns do not match specs. "
        f"Expected: {expected_columns}. Got: {list(df_preds.columns)}"
    )