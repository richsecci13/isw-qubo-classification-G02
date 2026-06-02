import os
import pytest
import pandas as pd
import numpy as np
from src.qubo_project.preprocessing import fit_normalize

@pytest.fixture
def sample_dataset(tmp_path):
    """
    Uses the real sample dataset 'data/sample_test_dataset.csv' created by the group,
    as strictly required by the specifications.
    """
    input_csv = "data/sample_test_dataset.csv"
    # Ensure the file exists before running tests
    assert os.path.exists(input_csv), f"Il file {input_csv} non esiste!"
    return input_csv, tmp_path

def test_preprocessing_only_numeric_columns(sample_dataset):
    """
    Test 1: Verify that the preprocessing produces only numerical columns.
    """
    input_csv, tmp_path = sample_dataset
    out_csv = str(tmp_path / "normalized.csv")
    out_json = str(tmp_path / "stats.json")
    
    # Run the pipeline
    fit_normalize(input_csv, "target", out_csv, out_json, minPercValid=0.5)
    
    # Load the output
    df_out = pd.read_csv(out_csv)
    
    # Assert every column is of a numeric data type
    for col in df_out.columns:
        assert pd.api.types.is_numeric_dtype(df_out[col]), f"Column '{col}' is not numeric"

def test_preprocessing_handles_missing_values(sample_dataset):
    """
    Test 2: Verify that the preprocessing successfully handles missing values.
    """
    input_csv, tmp_path = sample_dataset
    out_csv = str(tmp_path / "normalized.csv")
    out_json = str(tmp_path / "stats.json")
    
    # Run the pipeline
    fit_normalize(input_csv, "target", out_csv, out_json, minPercValid=0.5)
    
    # Load the output
    df_out = pd.read_csv(out_csv)
    
    # Assert there are absolutely zero missing (NaN) values left in the entire dataframe
    total_missing = df_out.isna().sum().sum()
    assert total_missing == 0, f"Output dataset still contains {total_missing} missing values"

def test_preprocessing_valid_normalization(sample_dataset):
    """
    Test 3: Verify that normalization produces a valid dataset (mean ~ 0, std ~ 1).
    """
    input_csv, tmp_path = sample_dataset
    out_csv = str(tmp_path / "normalized.csv")
    out_json = str(tmp_path / "stats.json")
    
    # Run the pipeline
    fit_normalize(input_csv, "target", out_csv, out_json, minPercValid=0.5)
    
    # Load the output
    df_out = pd.read_csv(out_csv)
    
    # Isolate features (exclude the target column)
    features = df_out.drop(columns=["target"])
    
    # Assert Z-score properties for each feature
    for col in features.columns:
        mean_val = features[col].mean()
        # scikit-learn's StandardScaler uses population standard deviation (ddof=0)
        std_val = features[col].std(ddof=0) 
        
        # We use np.isclose to account for minor floating-point inaccuracies
        assert np.isclose(mean_val, 0, atol=1e-5), f"Mean of '{col}' is {mean_val}, expected ~0"
        
        # If a column is perfectly constant in the dataset (like 'policy_code'), 
        # its scaled standard deviation will be 0.0, not 1.0.
        if not np.isclose(std_val, 0, atol=1e-5):
            assert np.isclose(std_val, 1, atol=1e-5), f"Std dev of '{col}' is {std_val}, expected ~1"