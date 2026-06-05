import os
import sys
import json
import pandas as pd
import streamlit as st
import altair as alt

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
    
    st.subheader("⚙️ Hyperparameters")
    col1, col2 = st.columns(2)
    with col1:
        target_col_prep = st.text_input("Target Column Name", value="target", key="target_prep")
    with col2:
        min_perc_valid = st.number_input("Minimum Valid Data %", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
        
    st.subheader("📁 File Paths")
    col3, col4 = st.columns(2)
    with col3:
        out_norm_csv = st.text_input("Output Normalized CSV", value=st.session_state.paths["normalized_csv"])
    with col4:
        out_prep_json = st.text_input("Output JSON", value=st.session_state.paths["prep_json"])
        
    if st.button("Run Preprocessing", type="primary"):
        if not os.path.exists(st.session_state.paths["uploaded_data"]):
            st.warning("Please upload a dataset in Step 1 first.")
        else:
            with st.spinner("Cleaning and normalizing data..."):
                try:
                    # Update session state with user paths
                    st.session_state.paths["normalized_csv"] = out_norm_csv
                    st.session_state.paths["prep_json"] = out_prep_json
                    
                    fit_normalize(
                        input_csv=st.session_state.paths["uploaded_data"],
                        target_column=target_col_prep,
                        normalized_csv=out_norm_csv,
                        outInitalRes_json=out_prep_json,
                        minPercValid=min_perc_valid
                    )
                    st.success("Preprocessing Complete!")
                    
                    with open(out_prep_json, 'r') as f:
                        prep_stats = json.load(f)
                        
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Original Features", prep_stats["n_input_features"])
                    m2.metric("Kept Features", prep_stats["n_kept_features"])
                    m3.metric("Processing Time (s)", prep_stats["dataset_processing_time"])
                    
                    if prep_stats["dropped_feature_names"]:
                        st.write("**Dropped Features:**")
                        st.dataframe(pd.DataFrame({"Feature Name": prep_stats["dropped_feature_names"]}))
                        
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")

# --- Tab 3: QUBO Feature Selection ---
with tab3:
    st.header("QUBO Optimization")
    
    st.subheader("⚙️ Hyperparameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        perc_selected = st.slider("Target Feature %", min_value=0.05, max_value=1.0, value=0.20, step=0.05)
        allowance = st.number_input("Tolerance Allowance (Features)", min_value=0, value=1, step=1)
    with col2:
        perc_test = st.slider("Test Set %", min_value=0.1, max_value=0.5, value=0.30, step=0.05)
        alpha_comp = st.number_input("Max Alpha Computations", min_value=10, value=100, step=10)
    with col3:
        target_col_qubo = st.text_input("Target Column Name", value="target", key="target_qubo")
        seed_qubo = st.number_input("Random Seed (QUBO)", value=42, step=1)

    st.subheader("📁 File Paths")
    col4, col5 = st.columns(2)
    with col4:
        in_norm_csv = st.text_input("Input Normalized CSV", value=st.session_state.paths["normalized_csv"], key="in_norm_qubo")
        out_train_csv = st.text_input("Output Reduced Train CSV", value=st.session_state.paths["reduced_train"])
        out_test_csv = st.text_input("Output Reduced Test CSV", value=st.session_state.paths["reduced_test"])
    with col5:
        out_optim_csv = st.text_input("Output Optimizations CSV", value=st.session_state.paths["optimizations_csv"])
        out_fs_json = st.text_input("Output FS JSON", value=st.session_state.paths["fs_json"])
        
    if st.button("Run Feature Selection", type="primary"):
        if not os.path.exists(in_norm_csv):
            st.warning(f"Input file not found: {in_norm_csv}. Please run Step 2.")
        else:
            with st.spinner("Optimizing feature set using Simulated Annealing..."):
                try:
                    # Update paths
                    st.session_state.paths["reduced_train"] = out_train_csv
                    st.session_state.paths["reduced_test"] = out_test_csv
                    st.session_state.paths["optimizations_csv"] = out_optim_csv
                    st.session_state.paths["fs_json"] = out_fs_json

                    select_features(
                        normalized_csv=in_norm_csv,
                        reducedTrain_csv=out_train_csv,
                        reducedTest_csv=out_test_csv,
                        output_ottim_csv=out_optim_csv,
                        output_json=out_fs_json,
                        target_column=target_col_qubo,
                        percTest=perc_test,
                        percSelected=perc_selected,
                        allowance=allowance,
                        seed=seed_qubo,
                        alpha_computations=alpha_comp
                    )
                    st.success("QUBO Optimization Complete!")
                    
                    opt_df = pd.read_csv(out_optim_csv)
                    st.write("**Optimization Trajectory: Alpha vs Features Selected**")
                    st.line_chart(opt_df.set_index("alpha")["n_features"])
                    
                    with open(out_fs_json, 'r') as f:
                        fs_stats = json.load(f)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Final Alpha", fs_stats["alpha"])
                    m2.metric("Features Selected", f'{fs_stats["n_selected"]} / {fs_stats["n_features"]}')
                    m3.metric("Train/Test Split", f'{fs_stats["training_dataset_size"]} / {fs_stats["test_dataset_size"]}')
                    
                    st.write("**Selected Features:**")
                    st.dataframe(pd.DataFrame({"Feature": fs_stats["selected_feature_names"]}))

                except Exception as e:
                    st.error(f"QUBO Selection failed: {e}")

# --- Tab 4: Training ---
with tab4:
    st.header("Model Training")
    
    st.subheader("⚙️ Hyperparameters")
    col1, col2 = st.columns(2)
    with col1:
        classifier_choice = st.selectbox("Select Classifier", ["Random_Forest", "XGBoost", "Logistic_Regression"])
        target_col_train = st.text_input("Target Column Name", value="target", key="target_train")
    with col2:
        seed_train = st.number_input("Model Seed", value=42, step=1)
        
    st.subheader("📁 File Paths")
    
    # Auto-generate suggestion names based on classifier choice, but let the user edit them
    suffix_map = {"Random_Forest": "_rf", "XGBoost": "_xgb", "Logistic_Regression": "_lr"}
    sfx = suffix_map[classifier_choice]
    
    col3, col4 = st.columns(2)
    with col3:
        in_train_csv = st.text_input("Input Reduced Train CSV", value=st.session_state.paths["reduced_train"], key="in_train_model")
        out_model_path = st.text_input("Output Model File (.joblib)", value=f"outputs/model{sfx}.joblib")
        
    # Extract model base name dynamically
    model_basename_train = os.path.basename(out_model_path).replace(".joblib", "")
    
    with col4:
        out_train_metrics = st.text_input("Output Training Metrics JSON", value=f"outputs/training_metrics{sfx}_{model_basename_train}.json")
        
    if st.button("Train Classifier", type="primary"):
        if not os.path.exists(in_train_csv):
            st.warning(f"Input file not found: {in_train_csv}. Please run Step 3.")
        else:
            with st.spinner(f"Training {classifier_choice}..."):
                try:
                    # Update session state paths based on training
                    st.session_state.paths["model_path"] = out_model_path
                    st.session_state.paths["train_metrics"] = out_train_metrics

                    train(
                        classifier=classifier_choice,
                        reducedTrain_csv=in_train_csv,
                        target_column=target_col_train,
                        model_path=out_model_path,
                        metrics_json=out_train_metrics,
                        seed=seed_train
                    )
                    st.success("Model Training Complete!")
                    
                    with open(out_train_metrics, 'r') as f:
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
    
    st.subheader("⚙️ Settings")
    target_col_pred = st.text_input("Target Column Name", value="target", key="target_pred")
    
    st.subheader("📁 File Paths")
    col1, col2 = st.columns(2)
    with col1:
        in_test_csv = st.text_input("Input Reduced Test CSV", value=st.session_state.paths["reduced_test"], key="in_test_pred")
        in_model_path = st.text_input("Input Model (.joblib)", value=st.session_state.paths["model_path"], key="in_model_pred")
        
    # Extract model name dynamically for output files
    model_name = os.path.basename(in_model_path).replace(".joblib", "")
            
    with col2:
        out_preds_csv = st.text_input("Output Predictions CSV", value=f"outputs/predictions_{model_name}.csv")
        out_stats_json = st.text_input("Output Classification Stats JSON", value=f"outputs/classification_stats_{model_name}.json")
    
    if st.button("Run Predictions", type="primary"):
        if not os.path.exists(in_model_path) or not os.path.exists(in_test_csv):
            st.warning("Ensure the model file and test dataset exist at the specified paths.")
        else:
            with st.spinner("Generating predictions..."):
                try:
                    # Update session state
                    st.session_state.paths["predictions_csv"] = out_preds_csv
                    st.session_state.paths["classif_stats"] = out_stats_json

                    predict(
                        reduced_Test_csv=in_test_csv,
                        target_column=target_col_pred,
                        model_path=in_model_path,
                        predictions_csv=out_preds_csv,
                        classif_stats_json=out_stats_json
                    )
                    st.success("Predictions Complete!")
                    
                    with open(out_stats_json, 'r') as f:
                        stats = json.load(f)
                        
                    m1, m2 = st.columns(2)
                    m1.metric("Accuracy", f"{stats['accuracy']:.4f}")
                    m2.metric("ROC-AUC Score", f"{stats['roc_auc']:.4f}")
                    
                    st.markdown("---")
                    
                    # Graphical representation: Grouped Bar Chart using Altair
                    st.subheader("Precision, Recall, and F1 by Class")
                    metrics_data = {
                        "Class": ["Class 0", "Class 1"],
                        "Precision": [stats["class_0"]["precision"], stats["class_1"]["precision"]],
                        "Recall": [stats["class_0"]["recall"], stats["class_1"]["recall"]],
                        "F1 Score": [stats["class_0"]["f1"], stats["class_1"]["f1"]]
                    }
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    # Melt dataframe for Altair to create the grouped bar effect
                    df_melted = metrics_df.melt(id_vars='Class', var_name='Metric', value_name='Score')
                    
                    # Create the Altair chart: 3 vertical bars per metric grouped by Class
                    grouped_bar_chart = alt.Chart(df_melted).mark_bar().encode(
                        x=alt.X('Class:N', title=None, axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color('Metric:N', legend=alt.Legend(title="Metrics")),
                        xOffset='Metric:N'
                    ).properties(height=400)
                    
                    st.altair_chart(grouped_bar_chart, use_container_width=True)
                    
                    st.markdown("---")
                    
                    st.subheader("Confusion Matrix")
                    cm_matrix = stats["confusion_matrix"]["matrix"]
                    labels = stats["confusion_matrix"]["labels"]
                    
                    cm_df = pd.DataFrame(
                        cm_matrix, 
                        index=[f"True {labels[0]}", f"True {labels[1]}"], 
                        columns=[f"Pred {labels[0]}", f"Pred {labels[1]}"]
                    )
                    st.table(cm_df)
                    
                    st.subheader("Predictions Preview")
                    preds_df = pd.read_csv(out_preds_csv)
                    st.dataframe(preds_df.head(10))

                except Exception as e:
                    st.error(f"Prediction failed: {e}")