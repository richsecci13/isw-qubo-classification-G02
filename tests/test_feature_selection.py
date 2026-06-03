import os
import json
import pytest
from src.qubo_project.preprocessing import fit_normalize
from src.qubo_project.feature_selection import select_features

# Hardcoded path to the physical sample dataset as per specifications
SAMPLE_DATA_PATH = os.path.join("data", "sample_test_dataset.csv")
TARGET_COLUMN = "target"  # Ensure your sample CSV uses this target name

@pytest.fixture
def phase2_setup(tmp_path):
    """
    Fixture to prepare the normalized dataset required by Phase 2.
    It reads the physical sample CSV, runs Phase 1 normalization, 
    and returns the paths needed for Phase 2 testing.
    """
    if not os.path.exists(SAMPLE_DATA_PATH):
        pytest.skip(f"Physical test file missing: {SAMPLE_DATA_PATH}. Please ensure it exists.")

    normalized_csv = str(tmp_path / "normalized.csv")
    prep_json = str(tmp_path / "prep_stats.json")

    # Run Phase 1 to generate the required input for Phase 2
    fit_normalize(
        input_csv=SAMPLE_DATA_PATH,
        target_column=TARGET_COLUMN,
        normalized_csv=normalized_csv,
        outInitalRes_json=prep_json,
        minPercValid=0.05
    )
    
    return normalized_csv, tmp_path

def test_feature_selection_binary_vector(phase2_setup):
    """
    Test 4: Verify that the feature selection produces a valid binary vector (only 0s and 1s).
    """
    normalized_csv, tmp_path = phase2_setup
    
    reduced_train = str(tmp_path / "training_reduced.csv")
    reduced_test = str(tmp_path / "test_reduced.csv")
    optim_csv = str(tmp_path / "optimizations.csv")
    out_json = str(tmp_path / "feature_selection_result.json")

    # Run Phase 2
    # Note: We use a lower alpha_computations (e.g., 10) to keep the test suite fast
    select_features(
        normalized_csv=normalized_csv,
        reducedTrain_csv=reduced_train,
        reducedTest_csv=reduced_test,
        output_ottim_csv=optim_csv,
        output_json=out_json,
        target_column=TARGET_COLUMN,
        percTest=0.30,
        percSelected=0.20,
        allowance=1,
        seed=42,
        alpha_computations=10 
    )

    # Load the JSON output
    with open(out_json, 'r') as f:
        stats = json.load(f)

    selected_vector = stats.get("selected_vector")
    
    # Assert the vector exists and contains exclusively 0s and 1s
    assert selected_vector is not None, "JSON output is missing 'selected_vector'"
    for val in selected_vector:
        assert val in [0, 1], f"Vector contains invalid non-binary value: {val}"

def test_feature_selection_percentage(phase2_setup):
    """
    Test 5: Verify that the number of features selected is approximately 20% 
    of the original features, within the specified allowance.
    """
    normalized_csv, tmp_path = phase2_setup
    
    reduced_train = str(tmp_path / "training_reduced.csv")
    reduced_test = str(tmp_path / "test_reduced.csv")
    optim_csv = str(tmp_path / "optimizations.csv")
    out_json = str(tmp_path / "feature_selection_result.json")

    target_percentage = 0.20
    allowance = 1

    # Run Phase 2
    select_features(
        normalized_csv=normalized_csv,
        reducedTrain_csv=reduced_train,
        reducedTest_csv=reduced_test,
        output_ottim_csv=optim_csv,
        output_json=out_json,
        target_column=TARGET_COLUMN,
        percTest=0.30,
        percSelected=target_percentage,
        allowance=allowance,
        seed=42,
        alpha_computations=15
    )

    # Load the JSON output
    with open(out_json, 'r') as f:
        stats = json.load(f)

    total_features = stats.get("n_features")
    n_selected = stats.get("n_selected")
    
    # Calculate the exact mathematical target K
    expected_k = int(round(total_features * target_percentage))
    
    # Assert the selected amount falls within the allowed mathematical tolerance
    difference = abs(n_selected - expected_k)
    assert difference <= allowance, (
        f"Selected {n_selected} features, expected {expected_k} "
        f"(±{allowance}) out of {total_features} total features."
    )