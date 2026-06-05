import os
import sys
import json
import pandas as pd
import streamlit as st

# Add the project root to sys.path so 'src' can be found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import our pipeline functions
from src.qubo_project.preprocessing import fit_normalize
from src.qubo_project.feature_selection import select_features
from src.qubo_project.model import train, predict

# --- Configuration & Initialization ---
st.set_page_config(page_title="QUBO Credit Risk Classification", layout="wide")

# Initialize default output paths in session state
if 'paths' not in st.session_state:
    st.session_state.paths = {
        "uploaded_data": "data/uploaded_dataset.csv",
        "normalized_csv": "outputs/normalized.csv",
        "prep_json": "outputs/preprocessing_result.json",
        "reduced_train": "outputs/training_reduced.csv",
        "reduced_test": "outputs/test_reduced.csv",
        "optimizations_csv": "outputs/optimizations.csv",
        "fs_json": "outputs/feature_selection_result.json",
        "model_path": "outputs/model.joblib",
        "train_metrics": "outputs/training_metrics.json",
        "predictions_csv": "outputs/predictions.csv",
        "classif_stats": "outputs/classification_stats.json"
    }

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

st.title("QUBO Feature Reduction & Classification Pipeline")
st.markdown("Execute the complete machine learning lifecycle, from data ingestion to QUBO-optimized feature selection and prediction.")

# --- UI Layout: Pipeline Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Dataset", 
    "2. Preprocessing", 
    "3. QUBO Selection", 
    "4. Training", 
    "5. Prediction & Metrics"
])

# --- Tab 1: Dataset Selection ---
with tab1:
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV dataset", type="csv")
    
    if uploaded_file is not None:
        try:
            # Save the uploaded file to the data directory
            df = pd.read_csv(uploaded_file)
            df.to_csv(st.session_state.paths["uploaded_data"], index=False)
            st.success(f"Dataset uploaded successfully! Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
            st.dataframe(df.head(10))
            st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"Error reading dataset: {e}")
    else:
        st.info("Awaiting dataset upload. (e.g., input_dataset.csv)")

# --- Tab 2: Preprocessing ---
with tab2:
    st.header("Data Preprocessing")
    
    col1, col2 = st.columns(2)
    with col1:
        target_col_prep = st.text_input("Target Column Name", value="target", key="target_prep")
    with col2:
        min_perc_valid = st.number_input("Minimum Valid Data %", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
        
    if st.button("Run Preprocessing", type="primary"):
        if not os.path.exists(st.session_state.paths["uploaded_data"]):
            st.warning("Please upload a dataset in Step 1 first.")
        else:
            with st.spinner("Cleaning and normalizing data..."):
                try:
                    fit_normalize(
                        input_csv=st.session_state.paths["uploaded_data"],
                        target_column=target_col_prep,
                        normalized_csv=st.session_state.paths["normalized_csv"],
                        outInitalRes_json=st.session_state.paths["prep_json"],
                        minPercValid=min_perc_valid
                    )
                    st.success("Preprocessing Complete!")
                    
                    # Parse and display JSON visually
                    with open(st.session_state.paths["prep_json"], 'r') as f:
                        prep_stats = json.load(f)
                        
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Original Features", prep_stats["n_input_features"])
                    m2.metric("Kept Features", prep_stats["n_kept_features"])
                    m3.metric("Processing Time (s)", prep_stats["dataset_processing_time"])
                    
                    if prep_stats["dropped_feature_names"]:
                        st.subheader("Dropped Features")
                        st.dataframe(pd.DataFrame({"Feature Name": prep_stats["dropped_feature_names"]}))
                        
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")

# --- Tab 3: QUBO Feature Selection ---
with tab3:
    st.header("QUBO Optimization")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        perc_selected = st.slider("Target Feature %", min_value=0.05, max_value=1.0, value=0.20, step=0.05)
        allowance = st.number_input("Tolerance Allowance (Features)", min_value=0, value=1, step=1)
    with col2:
        perc_test = st.slider("Test Set %", min_value=0.1, max_value=0.5, value=0.30, step=0.05)
        alpha_comp = st.number_input("Max Alpha Computations", min_value=10, value=100, step=10)
    with col3:
        target_col_qubo = st.text_input("Target Column Name", value="target", key="target_qubo")
        seed_qubo = st.number_input("Random Seed", value=42, step=1)
        
    if st.button("Run Feature Selection", type="primary"):
        if not os.path.exists(st.session_state.paths["normalized_csv"]):
            st.warning("Please run Preprocessing in Step 2 first.")
        else:
            with st.spinner("Optimizing feature set using Simulated Annealing..."):
                try:
                    select_features(
                        normalized_csv=st.session_state.paths["normalized_csv"],
                        reducedTrain_csv=st.session_state.paths["reduced_train"],
                        reducedTest_csv=st.session_state.paths["reduced_test"],
                        output_ottim_csv=st.session_state.paths["optimizations_csv"],
                        output_json=st.session_state.paths["fs_json"],
                        target_column=target_col_qubo,
                        percTest=perc_test,
                        percSelected=perc_selected,
                        allowance=allowance,
                        seed=seed_qubo,
                        alpha_computations=alpha_comp
                    )
                    st.success("QUBO Optimization Complete!")
                    
                    # Read Optimization History CSV for graphing
                    opt_df = pd.read_csv(st.session_state.paths["optimizations_csv"])
                    
                    # Plot Alpha vs Number of Features
                    st.subheader("Optimization Trajectory: Alpha vs Features Selected")
                    chart_data = opt_df.set_index("alpha")["n_features"]
                    st.line_chart(chart_data)
                    
                    # Parse and display JSON visually
                    with open(st.session_state.paths["fs_json"], 'r') as f:
                        fs_stats = json.load(f)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Final Alpha", fs_stats["alpha"])
                    m2.metric("Features Selected", f'{fs_stats["n_selected"]} / {fs_stats["n_features"]}')
                    m3.metric("Train/Test Split", f'{fs_stats["training_dataset_size"]} / {fs_stats["test_dataset_size"]}')
                    
                    st.subheader("Selected Features")
                    st.dataframe(pd.DataFrame({"Feature": fs_stats["selected_feature_names"]}))

                except Exception as e:
                    st.error(f"QUBO Selection failed: {e}")

# --- Tab 4: Training ---
with tab4:
    st.header("Model Training")
    
    col1, col2 = st.columns(2)
    with col1:
        classifier_choice = st.selectbox("Select Classifier", ["Random_Forest", "XGBoost", "Logistic_Regression"])
        target_col_train = st.text_input("Target Column Name", value="target", key="target_train")
    with col2:
        seed_train = st.number_input("Model Seed", value=42, step=1)
        
    if st.button("Train Classifier", type="primary"):
        if not os.path.exists(st.session_state.paths["reduced_train"]):
            st.warning("Please run Feature Selection in Step 3 first.")
        else:
            with st.spinner(f"Training {classifier_choice}..."):
                try:
                    if classifier_choice == "Random_Forest":
                        suffix = "_rf"
                    elif classifier_choice == "XGBoost":
                        suffix = "_xgb"
                    else:
                        suffix = "_lr"
                    
                    st.session_state.paths["model_path"] = f"outputs/model{suffix}.joblib"
                    st.session_state.paths["train_metrics"] = f"outputs/training_metrics{suffix}.json"
                    st.session_state.paths["predictions_csv"] = f"outputs/predictions{suffix}.csv"
                    st.session_state.paths["classif_stats"] = f"outputs/classification_stats{suffix}.json"

                    train(
                        classifier=classifier_choice,
                        reducedTrain_csv=st.session_state.paths["reduced_train"],
                        target_column=target_col_train,
                        model_path=st.session_state.paths["model_path"],
                        metrics_json=st.session_state.paths["train_metrics"],
                        seed=seed_train
                    )
                    st.success("Model Training Complete!")
                    
                    # Parse and display JSON visually
                    with open(st.session_state.paths["train_metrics"], 'r') as f:
                        train_stats = json.load(f)
                        
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Algorithm", train_stats["classifier"])
                    m2.metric("Training Time (s)", train_stats["training_time"])
                    m3.metric("Dataset Size", train_stats["n_samples"])
                except Exception as e:
                    st.error(f"Training failed: {e}")

# --- Tab 5: Prediction & Metrics ---
with tab5:
    st.header("Inference & Performance Evaluation")
    
    target_col_pred = st.text_input("Target Column Name", value="target", key="target_pred")
    
    if st.button("Run Predictions", type="primary"):
        if not os.path.exists(st.session_state.paths["model_path"]) or not os.path.exists(st.session_state.paths["reduced_test"]):
            st.warning("Ensure the model is trained (Step 4) and test data exists (Step 3).")
        else:
            with st.spinner("Generating predictions..."):
                try:
                    predict(
                        reduced_Test_csv=st.session_state.paths["reduced_test"],
                        target_column=target_col_pred,
                        model_path=st.session_state.paths["model_path"],
                        predictions_csv=st.session_state.paths["predictions_csv"],
                        classif_stats_json=st.session_state.paths["classif_stats"]
                    )
                    st.success("Predictions Complete!")
                    
                    # Parse JSON into visually appealing formats
                    with open(st.session_state.paths["classif_stats"], 'r') as f:
                        stats = json.load(f)
                        
                    # Top Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("Accuracy", f"{stats['accuracy']:.4f}")
                    m2.metric("ROC-AUC Score", f"{stats['roc_auc']:.4f}")
                    
                    st.markdown("---")
                    
                    # Graphical representation of metrics using a Bar Chart
                    st.subheader("Precision, Recall, and F1 by Class")
                    metrics_data = {
                        "Class": ["Class 0", "Class 1"],
                        "Precision": [stats["class_0"]["precision"], stats["class_1"]["precision"]],
                        "Recall": [stats["class_0"]["recall"], stats["class_1"]["recall"]],
                        "F1 Score": [stats["class_0"]["f1"], stats["class_1"]["f1"]]
                    }
                    metrics_df = pd.DataFrame(metrics_data).set_index("Class")
                    st.bar_chart(metrics_df)
                    
                    st.markdown("---")
                    
                    # Table representation of Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm_matrix = stats["confusion_matrix"]["matrix"]
                    labels = stats["confusion_matrix"]["labels"]
                    
                    cm_df = pd.DataFrame(
                        cm_matrix, 
                        index=[f"True {labels[0]}", f"True {labels[1]}"], 
                        columns=[f"Pred {labels[0]}", f"Pred {labels[1]}"]
                    )
                    st.table(cm_df)
                    
                    # Show a snippet of the raw prediction output
                    st.subheader("Predictions Preview")
                    preds_df = pd.read_csv(st.session_state.paths["predictions_csv"])
                    st.dataframe(preds_df.head(10))

                except Exception as e:
                    st.error(f"Prediction failed: {e}")