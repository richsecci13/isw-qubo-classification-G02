import time
import json
import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def fit_normalize(
    input_csv: str,
    target_column: str,
    normalized_csv: str,
    outInitalRes_json: str,
    minPercValid: float = 0.05
):
    """
    Reads the dataset, drops features with too many missing/zero values,
    normalizes the remaining features, and saves the outputs.
    """
    # --- 1. Data Ingestion ---
    start_input_time = time.time()
    df = pd.read_csv(input_csv)
    dataset_input_time = time.time() - start_input_time

    # --- 2. Data Processing ---
    start_processing_time = time.time()
    
    # Store initial metrics
    dataset_size = df.shape[0]
    n_input_features = df.shape[1] - 1  # Excluding the target column

    # Calculate the percentage of valid (non-null AND non-zero) data per column
    valid_mask = df.notna() & (df != 0)
    valid_percentages = valid_mask.mean()

    # Identify features to drop (strictly less than minPercValid)
    features_to_drop = valid_percentages[valid_percentages < minPercValid].index.tolist()
    
    # Ensure we never drop the target column, regardless of its distribution
    if target_column in features_to_drop:
        features_to_drop.remove(target_column)

    # Drop the invalid features
    df_reduced = df.drop(columns=features_to_drop)

    # Separate target and features for scaling
    y = df_reduced[target_column]
    X = df_reduced.drop(columns=[target_column])

    # Normalize features using Z-score (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reconstruct the DataFrame with scaled features and original column names
    df_normalized = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Re-attach the target column (keeping it unmodified)
    df_normalized[target_column] = y
    
    # Final feature count
    n_kept_features = df_normalized.shape[1] - 1

    dataset_processing_time = time.time() - start_processing_time

    # --- 3. Save Outputs ---
    # Ensure output directories exist before writing
    os.makedirs(os.path.dirname(normalized_csv) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(outInitalRes_json) or '.', exist_ok=True)

    # Save the normalized CSV
    df_normalized.to_csv(normalized_csv, index=False)

    # Compile and save JSON statistics
    stats = {
        "n_input_features": n_input_features,
        "n_kept_features": n_kept_features,
        "dataset_size": dataset_size,
        "dataset_input_time": round(dataset_input_time, 2),
        "dataset_processing_time": round(dataset_processing_time, 2),
        "dropped_feature_names": features_to_drop
    }
    
    with open(outInitalRes_json, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Preprocessing completed. Kept {n_kept_features}/{n_input_features} features.")

if __name__ == "__main__":
    # Command Line Interface parsing matching the specification
    parser = argparse.ArgumentParser(description="Preprocess and normalize the dataset")
    parser.add_argument("--input", required=True, help="Input dataset CSV path")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--out-data", required=True, help="Output normalized CSV path")
    parser.add_argument("--out-json", required=True, help="Output JSON results path")
    parser.add_argument("--min-perc-valid", type=float, default=0.05, help="Minimum percentage of valid non-zero data")
    
    args = parser.parse_args()
    
    fit_normalize(
        input_csv=args.input,
        target_column=args.target,
        normalized_csv=args.out_data,
        outInitalRes_json=args.out_json,
        minPercValid=args.min_perc_valid
    )