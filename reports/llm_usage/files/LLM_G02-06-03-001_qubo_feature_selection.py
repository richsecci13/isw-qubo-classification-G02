import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import dimod
import neal

def select_features(
    normalized_csv: str,
    reducedTrain_csv: str,
    reducedTest_csv: str,
    output_ottim_csv: str,
    output_json: str,
    target_column: str,
    percTest: float = 0.30,
    percSelected: float = 0.20,
    allowance: int = 1,
    seed: int = 42,
    alpha_computations: int = 100
):
    """
    Selects optimal features by formulating a QUBO problem and varying the alpha parameter.
    Balances target influence (maximization) and feature independence (minimization).
    """
    np.random.seed(seed)
    
    # --- 1. Data Loading & Setup ---
    df = pd.read_csv(normalized_csv)
    y = df[target_column]
    X = df.drop(columns=[target_column])
    feature_names = X.columns.tolist()
    m = len(feature_names)
    
    # Calculate the target number of features (K)
    target_k = int(round(percSelected * m))
    
    # --- 2. Correlation Matrices Calculation ---
    start_q_time = time.time()
    
    # Spearman correlation between features (|rho_jk|)
    corr_features_df = X.corr(method='spearman').abs()
    corr_features = corr_features_df.values
    np.fill_diagonal(corr_features, 0) # Diagonal is handled by the linear terms
    corr_features = np.nan_to_num(corr_features)
    
    # Spearman correlation with the target (|rho_Vj|)
    corr_target_df = X.corrwith(y, method='spearman').abs()
    corr_target = np.nan_to_num(corr_target_df.values)
    
    q_matrix_creation_time = time.time() - start_q_time

    # --- 3. QUBO Optimization Loop (Binary Search on Alpha) ---
    sampler = neal.SimulatedAnnealingSampler()
    
    low_alpha = 0.0
    high_alpha = 1.0
    best_alpha = 0.5
    best_selected_vector = []
    best_selected_names = []
    
    min_diff = float('inf') # To track the iteration closest to target K
    
    optimization_history = []
    optimization_times = []
    
    for step in range(alpha_computations):
        alpha = (low_alpha + high_alpha) / 2.0
        
        # Build QUBO model: minimize -x^T Q x
        # Linear terms: -alpha * |rho_Vj|
        linear = {i: -alpha * corr_target[i] for i in range(m)}
        
        # Quadratic terms: 2 * (1 - alpha) * |rho_ij| 
        # (Multiplied by 2 to account for symmetry since we only loop upper-triangular)
        quadratic = {
            (i, j): 2 * (1 - alpha) * corr_features[i, j] 
            for i in range(m) for j in range(i + 1, m)
        }
        
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, 'BINARY')
        
        # Sampling (Simulated Annealing)
        opt_start = time.time()
        sampleset = sampler.sample(bqm, num_reads=10, seed=seed)
        best_sample = sampleset.first.sample
        energy = sampleset.first.energy
        opt_time = time.time() - opt_start
        
        optimization_times.append(opt_time)
        
        # Extract selected features
        selected_vector = [best_sample[i] for i in range(m)]
        n_selected = sum(selected_vector)
        
        # Log the current iteration
        optimization_history.append({
            "alpha": alpha,
            "opt_time": opt_time,
            "n_selected": n_selected,
            "energy": energy
        })
        
        current_diff = abs(n_selected - target_k)
        
        # Stopping condition & Search update
        if current_diff <= allowance:
            best_alpha = alpha
            best_selected_vector = selected_vector
            best_selected_names = [feature_names[i] for i in range(m) if selected_vector[i] == 1]
            break
        elif n_selected < target_k:
            # Too few features -> we need more weight on target influence (increase alpha)
            low_alpha = alpha
        else:
            # Too many features -> we need more weight on independence penalty (decrease alpha)
            high_alpha = alpha
            
        # Fallback save: track the best state in case iterations exhaust without a perfect match
        if current_diff < min_diff:
            min_diff = current_diff
            best_alpha = alpha
            best_selected_vector = selected_vector
            best_selected_names = [feature_names[i] for i in range(m) if selected_vector[i] == 1]

    # Calculate optimization time statistics
    mean_opt_time = np.mean(optimization_times)
    std_opt_time = np.std(optimization_times) if len(optimization_times) > 1 else 0.0

    # Sort history strictly by alpha ascending as per specifications
    optimization_history.sort(key=lambda x: x["alpha"])

    # --- 4. Sequential Dataset Split ---
    final_columns = best_selected_names + [target_column]
    df_reduced = df[final_columns]
    
    # Split sequentially: first M samples for training, remaining for testing
    dataset_size = len(df_reduced)
    test_size_n = int(round(dataset_size * percTest))
    M = dataset_size - test_size_n
    
    df_train = df_reduced.iloc[:M]
    df_test = df_reduced.iloc[M:]

    # --- 5. Output Saving ---
    os.makedirs(os.path.dirname(reducedTrain_csv) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(output_ottim_csv) or '.', exist_ok=True)
    
    # Save CSV Datasets
    df_train.to_csv(reducedTrain_csv, index=False)
    df_test.to_csv(reducedTest_csv, index=False)
    
    # Save Optimization History CSV
    pd.DataFrame(optimization_history).to_csv(
        output_ottim_csv, 
        index=False, 
        header=["alpha", "optimization_time", "n_features", "cost_value"]
    )
    
    # Save JSON Statistics
    stats = {
        "n_features": m,
        "target_ratio": percSelected,
        "target_k": target_k,
        "allowance": allowance,
        "n_selected": sum(best_selected_vector),
        "alpha": round(best_alpha, 4),
        "selected_vector": best_selected_vector,
        "selected_feature_names": best_selected_names,
        "algorithm": "simulated_annealing",
        "seed": seed,
        "alpha_computations": len(optimization_history),
        "percTest": percTest,
        "training_dataset_size": len(df_train),
        "test_dataset_size": len(df_test),
        "q_matrix_creation_time": round(q_matrix_creation_time, 4),
        "mean optimization time": round(mean_opt_time, 4),
        "std_dev_optimization_time": round(std_opt_time, 4)
    }
    
    with open(output_json, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"QUBO Selection completed. Selected {sum(best_selected_vector)}/{m} features at alpha={best_alpha:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QUBO Feature Selection")
    parser.add_argument("--in-normalized", required=True, help="Input normalized CSV path")
    parser.add_argument("--out-train", required=True, help="Output reduced training set path")
    parser.add_argument("--out-test", required=True, help="Output reduced testing set path")
    parser.add_argument("--out-optimizations", required=True, help="Output optimization CSV path")
    parser.add_argument("--out-json", required=True, help="Output JSON results path")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--perc-selected", type=float, default=0.20, help="Target ratio of features to select")
    parser.add_argument("--allowance", type=int, default=1, help="Feature tolerance allowance")
    parser.add_argument("--perc-test", type=float, default=0.30, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha-computations", type=int, default=100, help="Max alpha iterations")
    
    args = parser.parse_args()
    
    select_features(
        normalized_csv=args.in_normalized,
        reducedTrain_csv=args.out_train,
        reducedTest_csv=args.out_test,
        output_ottim_csv=args.out_optimizations,
        output_json=args.out_json,
        target_column=args.target,
        percTest=args.perc_test,
        percSelected=args.perc_selected,
        allowance=args.allowance,
        seed=args.seed,
        alpha_computations=args.alpha_computations
    )