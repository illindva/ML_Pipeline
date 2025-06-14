import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import pickle
import base64
from io import BytesIO

# Import custom modules
from config import get_database_manager, get_database_info, APP_TITLE, APP_DESCRIPTION
from data_processor import DataProcessor
from model_builder import ModelBuilder
from visualizations import Visualizer
from utils import Utils

# Page configuration
st.set_page_config(
    page_title="Fraud Detection ML Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'dataset_id' not in st.session_state:
    st.session_state.dataset_id = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

# Initialize components
@st.cache_resource
def initialize_components():
    db_manager = get_database_manager()
    data_processor = DataProcessor()
    model_builder = ModelBuilder()
    visualizer = Visualizer()
    utils = Utils()
    return db_manager, data_processor, model_builder, visualizer, utils

db_manager, data_processor, model_builder, visualizer, utils = initialize_components()

def main():
    st.title("🔍 Fraud Detection ML Pipeline")
    st.markdown("**A comprehensive machine learning platform for fraud detection and analysis**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    steps = [
        "1. Data Upload & Storage",
        "2. Data Exploration & Cleaning",
        "3. Feature Engineering",
        "4. Model Building",
        "5. Model Evaluation",
        "6. Predictions & Deployment",
        "7. Database Management"
    ]
    
    selected_step = st.sidebar.selectbox("Select Step", steps, index=st.session_state.current_step - 1)
    st.session_state.current_step = steps.index(selected_step) + 1
    
    # Display current datasets
    st.sidebar.subheader("Available Datasets")
    datasets = db_manager.get_all_datasets()
    if datasets:
        dataset_names = [f"{row[1]} ({row[0]})" for row in datasets]
        selected_dataset = st.sidebar.selectbox("Select Dataset", ["None"] + dataset_names)
        if selected_dataset != "None":
            st.session_state.dataset_id = int(selected_dataset.split("(")[1].split(")")[0])
    else:
        st.sidebar.info("No datasets uploaded yet")
    
    # Main content based on selected step
    if st.session_state.current_step == 1:
        step_1_data_upload()
    elif st.session_state.current_step == 2:
        step_2_data_exploration()
    elif st.session_state.current_step == 3:
        step_3_feature_engineering()
    elif st.session_state.current_step == 4:
        step_4_model_building()
    elif st.session_state.current_step == 5:
        step_5_model_evaluation()
    elif st.session_state.current_step == 6:
        step_6_predictions()
    elif st.session_state.current_step == 7:
        step_7_database_management()

def step_1_data_upload():
    st.header("📂 Step 1: Data Upload & Storage")
    
    # Upload mode selection
    upload_mode = st.radio(
        "Select upload mode:",
        ["Single File", "Multiple CSV Files"],
        help="Choose whether to upload one file or merge multiple CSV files"
    )
    
    if upload_mode == "Single File":
        # Single file upload
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Dataset")
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload your dataset for fraud detection analysis"
            )
            
            if uploaded_file is not None:
                try:
                    # Read the file
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    # Fix Arrow compatibility
                    df = data_processor._fix_arrow_compatibility(df)
                    
                    st.success(f"File loaded successfully! Shape: {df.shape}")
                    
                    # Display basic info
                    st.subheader("Dataset Preview")
                    st.dataframe(df.head(10))
                    
                    # Dataset metadata
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.metric("Rows", df.shape[0])
                        st.metric("Columns", df.shape[1])
                    
                    with col_info2:
                        st.metric("Missing Values", df.isnull().sum().sum())
                        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                    
                    # Save dataset
                    st.subheader("Save Dataset")
                    dataset_name = st.text_input("Dataset Name", value=uploaded_file.name.split('.')[0])
                    dataset_description = st.text_area("Description", placeholder="Brief description of the dataset...")
                    
                    if st.button("Save Dataset", type="primary"):
                        try:
                            dataset_id = db_manager.save_dataset(
                                name=dataset_name,
                                description=dataset_description,
                                data=df
                            )
                            st.session_state.dataset_id = dataset_id
                            st.session_state.current_data = df
                            st.success(f"Dataset saved successfully with ID: {dataset_id}")
                            st.balloons()
                        except Exception as e:
                            st.error(f"Error saving dataset: {str(e)}")
                            
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        with col2:
            st.subheader("Dataset Statistics")
            if st.session_state.dataset_id:
                try:
                    df = db_manager.load_dataset(st.session_state.dataset_id)
                    stats = utils.get_dataset_statistics(df)
                    
                    for key, value in stats.items():
                        st.metric(key.replace('_', ' ').title(), value)
                        
                except Exception as e:
                    st.error(f"Error loading dataset statistics: {str(e)}")
    
    else:
        # Multiple CSV files upload
        st.subheader("Multiple CSV Files Merge")
        
        uploaded_files = st.file_uploader(
            "Choose CSV files to merge", 
            type=['csv'],
            accept_multiple_files=True,
            help="Upload multiple CSV files that will be merged based on common columns"
        )
        
        if uploaded_files and len(uploaded_files) > 1:
            try:
                # Read all CSV files
                dataframes = []
                file_names = []
                
                st.subheader("Files Overview")
                files_info = []
                
                for uploaded_file in uploaded_files:
                    df = pd.read_csv(uploaded_file)
                    dataframes.append(df)
                    file_names.append(uploaded_file.name)
                    
                    files_info.append({
                        'File': uploaded_file.name,
                        'Rows': df.shape[0],
                        'Columns': df.shape[1],
                        'Column Names': ', '.join(df.columns[:5]) + ('...' if len(df.columns) > 5 else '')
                    })
                
                files_df = pd.DataFrame(files_info)
                st.dataframe(files_df, use_container_width=True)
                
                # Merge strategy selection
                st.subheader("Merge Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    merge_strategy = st.selectbox(
                        "Merge Strategy",
                        ["auto", "inner", "outer", "concatenate"],
                        help="""
                        - Auto: Automatically determine best merge approach
                        - Inner: Keep only rows with matching keys in all files
                        - Outer: Keep all rows, fill missing with NaN
                        - Concatenate: Stack files vertically with source identifier
                        """
                    )
                
                with col2:
                    # Show common columns
                    all_columns = [set(df.columns) for df in dataframes]
                    common_columns = set.intersection(*all_columns) if all_columns else set()
                    
                    st.write("**Common Columns:**")
                    if common_columns:
                        st.write(f"Found {len(common_columns)} common columns")
                        for col in sorted(common_columns):
                            st.write(f"• {col}")
                    else:
                        st.warning("No common columns found")
                
                # Merge files
                if st.button("Merge Files", type="primary"):
                    with st.spinner("Merging files..."):
                        try:
                            merged_df, merge_report = data_processor.merge_csv_files(
                                dataframes, file_names, merge_strategy
                            )
                            
                            st.session_state.current_data = merged_df
                            st.session_state.merge_report = merge_report
                            
                            # Show merge results
                            st.success("Files merged successfully!")
                            
                            # Merge report
                            st.subheader("Merge Report")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Files Merged", merge_report["total_files"])
                            with col2:
                                st.metric("Final Rows", merge_report["final_shape"][0])
                            with col3:
                                st.metric("Final Columns", merge_report["final_shape"][1])
                            
                            if merge_report.get("warnings"):
                                st.warning("Merge Warnings:")
                                for warning in merge_report["warnings"]:
                                    st.write(f"• {warning}")
                            
                            st.write(f"**Merge Type:** {merge_report.get('merge_type', 'Unknown')}")
                            
                            if merge_report.get("merge_keys"):
                                st.write(f"**Merge Keys:** {', '.join(merge_report['merge_keys'])}")
                            
                            # Dataset naming
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                dataset_name = st.text_input(
                                    "Dataset Name", 
                                    value=f"merged_dataset_{len(file_names)}_files",
                                    help="Give your merged dataset a meaningful name"
                                )
                            
                            with col2:
                                dataset_description = st.text_area(
                                    "Dataset Description",
                                    value=f"Merged dataset from {len(file_names)} files: {', '.join(file_names)}",
                                    help="Describe the merged dataset"
                                )
                            
                            # Save merged dataset
                            if st.button("Save Merged Dataset", type="secondary"):
                                if dataset_name:
                                    try:
                                        dataset_id = db_manager.save_dataset(
                                            name=dataset_name,
                                            description=dataset_description,
                                            data=merged_df
                                        )
                                        st.session_state.dataset_id = dataset_id
                                        st.success(f"Merged dataset saved! Dataset ID: {dataset_id}")
                                        st.balloons()
                                        
                                    except Exception as e:
                                        st.error(f"Error saving dataset: {str(e)}")
                                else:
                                    st.error("Please provide a dataset name")
                            
                            # Show merged dataset preview
                            st.subheader("Merged Dataset Preview")
                            st.dataframe(merged_df.head(10))
                            
                        except Exception as e:
                            st.error(f"Error merging files: {str(e)}")
                            st.info("Try using 'concatenate' merge strategy if files have different structures")
                
            except Exception as e:
                st.error(f"Error reading files: {str(e)}")
                st.info("Please ensure all files are valid CSV format")
        
        elif uploaded_files and len(uploaded_files) == 1:
            st.info("You've uploaded only one file. For single file upload, please use 'Single File' mode above.")
        
        else:
            st.info("Please upload at least 2 CSV files to merge them together.")
    
    # Show existing datasets
    st.subheader("Existing Datasets")
    _show_existing_datasets()

def _show_existing_datasets():
    """Show existing datasets section"""
    datasets = db_manager.get_all_datasets()
    
    if datasets:
        # Create a DataFrame for better display
        datasets_df = pd.DataFrame(datasets, columns=['ID', 'Name', 'Description', 'Upload Date', 'File Size', 'Columns Info'])
        
        # Display datasets table
        st.dataframe(datasets_df, use_container_width=True)
        
        # Load existing dataset
        selected_dataset = st.selectbox(
            "Select a dataset to load:",
            options=[None] + [f"{row[0]} - {row[1]}" for row in datasets],
            format_func=lambda x: "Choose a dataset..." if x is None else x
        )
        
        if selected_dataset and st.button("Load Selected Dataset"):
            dataset_id = int(selected_dataset.split(' - ')[0])
            try:
                df = db_manager.load_dataset(dataset_id)
                st.session_state.current_data = df
                st.session_state.dataset_id = dataset_id
                st.success("Dataset loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
    else:
        st.info("No datasets found. Upload your first dataset above!")

def step_2_data_exploration():
    st.header("🔍 Step 2: Data Exploration & Cleaning")
    
    if not st.session_state.dataset_id:
        st.warning("Please upload and select a dataset first!")
        return
    
    try:
        # Use processed data if available, otherwise use original
        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            st.info("📊 Showing processed data. Use 'Load Original Data' to reset.")
            if st.button("🔄 Load Original Data"):
                st.session_state.processed_data = None
                st.rerun()
        else:
            df = db_manager.load_dataset(st.session_state.dataset_id)
        
        # Refresh button for current view
        col_refresh1, col_refresh2 = st.columns([6, 1])
        with col_refresh2:
            if st.button("🔄 Refresh", key="refresh_exploration"):
                st.rerun()
        
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Data Quality", "Distributions", "Correlations"])
        
        with tab1:
            st.subheader("Dataset Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Basic Information")
                st.dataframe(df.describe())
                
            with col2:
                st.subheader("Data Types")
                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Unique Values': df.nunique()
                })
                st.dataframe(dtype_df)
        
        with tab2:
            st.subheader("Data Quality Analysis")
            
            # Missing values analysis
            missing_data = utils.analyze_missing_data(df)
            if not missing_data.empty:
                st.subheader("Missing Values")
                st.dataframe(missing_data)
                
                # Visualize missing data
                fig = visualizer.plot_missing_data(df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found!")
            
            # Data cleaning options
            st.subheader("Data Cleaning Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                handle_missing = st.selectbox(
                    "Handle Missing Values",
                    ["None", "Drop rows", "Fill with mean/mode", "Forward fill", "Backward fill"]
                )
                
                remove_duplicates = st.checkbox("Remove duplicate rows")
                
            with col2:
                outlier_method = st.selectbox(
                    "Outlier Detection",
                    ["None", "IQR Method", "Z-Score", "Isolation Forest"]
                )
                
                fix_dtypes = st.checkbox("Auto-fix data types", value=True)
                
            with col3:
                drop_empty_columns = st.checkbox("Drop columns with all null values")
                modify_categorical = st.checkbox("Modify incorrect categorical values")
            
            # Categorical value modification interface
            value_mappings = {}
            if modify_categorical:
                st.subheader("Categorical Value Modification")
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                if categorical_cols:
                    selected_cat_col = st.selectbox("Select categorical column to modify", categorical_cols)
                    
                    if selected_cat_col:
                        unique_values = df[selected_cat_col].dropna().unique()
                        st.write(f"Current unique values in '{selected_cat_col}': {list(unique_values[:10])}{'...' if len(unique_values) > 10 else ''}")
                        
                        # Value mapping interface
                        st.write("**Map incorrect values to correct ones:**")
                        
                        col_map1, col_map2 = st.columns(2)
                        with col_map1:
                            st.write("Original Value")
                        with col_map2:
                            st.write("New Value")
                        
                        for i, value in enumerate(unique_values[:10]):  # Limit to first 10 for UI
                            col_map1, col_map2 = st.columns(2)
                            with col_map1:
                                st.text(str(value))
                            with col_map2:
                                new_value = st.text_input(f"", value=str(value), key=f"map_{selected_cat_col}_{i}", label_visibility="collapsed")
                                if new_value != str(value):
                                    value_mappings[value] = new_value
                        
                        if len(unique_values) > 10:
                            st.info(f"Showing first 10 values. Total unique values: {len(unique_values)}")
                        
                        if value_mappings:
                            st.write("**Mappings to apply:**")
                            for old_val, new_val in value_mappings.items():
                                st.write(f"'{old_val}' → '{new_val}'")
            
            if st.button("Apply Data Cleaning", type="primary"):
                try:
                    cleaned_df = data_processor.clean_data(
                        df,
                        handle_missing=handle_missing,
                        remove_duplicates=remove_duplicates,
                        outlier_method=outlier_method,
                        fix_dtypes=fix_dtypes,
                        drop_empty_columns=drop_empty_columns,
                        value_mappings=value_mappings if modify_categorical else {}
                    )
                    
                    # Save cleaned data
                    analysis_id = db_manager.save_analysis(
                        st.session_state.dataset_id,
                        "data_cleaning",
                        {
                            "original_shape": df.shape,
                            "cleaned_shape": cleaned_df.shape,
                            "cleaning_methods": {
                                "missing_values": handle_missing,
                                "duplicates": remove_duplicates,
                                "outliers": outlier_method,
                                "dtypes": fix_dtypes
                            }
                        }
                    )
                    
                    st.session_state.processed_data = cleaned_df
                    st.success(f"Data cleaned successfully! Shape changed from {df.shape} to {cleaned_df.shape}")
                    
                except Exception as e:
                    st.error(f"Error cleaning data: {str(e)}")
        
        with tab3:
            st.subheader("Data Distributions")
            
            # Select columns for visualization
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols:
                st.subheader("Numerical Distributions")
                selected_numeric = st.multiselect("Select numerical columns", numeric_cols, default=numeric_cols[:3])
                
                if selected_numeric:
                    for col in selected_numeric:
                        fig = visualizer.plot_distribution(df, col)
                        st.plotly_chart(fig, use_container_width=True)
            
            if categorical_cols:
                st.subheader("Categorical Distributions")
                selected_categorical = st.multiselect("Select categorical columns", categorical_cols, default=categorical_cols[:3])
                
                if selected_categorical:
                    for col in selected_categorical:
                        fig = visualizer.plot_categorical_distribution(df, col)
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Correlation Analysis")
            
            if numeric_cols:
                correlation_matrix = df[numeric_cols].corr()
                fig = visualizer.plot_correlation_matrix(correlation_matrix)
                st.plotly_chart(fig, use_container_width=True)
                
                # High correlation pairs
                high_corr_pairs = utils.find_high_correlation_pairs(correlation_matrix, threshold=0.8)
                if not high_corr_pairs.empty:
                    st.subheader("High Correlation Pairs (>0.8)")
                    st.dataframe(high_corr_pairs)
            else:
                st.info("No numerical columns found for correlation analysis")
                
    except Exception as e:
        st.error(f"Error in data exploration: {str(e)}")

def step_3_feature_engineering():
    st.header("⚙️ Step 3: Feature Engineering")
    
    if not st.session_state.dataset_id:
        st.warning("Please upload and select a dataset first!")
        return
    
    try:
        # Use processed data if available, otherwise use original
        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            st.info("📊 Using processed data from previous steps.")
        else:
            df = db_manager.load_dataset(st.session_state.dataset_id)
        
        # Refresh button for current view
        col_refresh1, col_refresh2 = st.columns([6, 1])
        with col_refresh2:
            if st.button("🔄 Refresh", key="refresh_feature_eng"):
                st.rerun()
        
        tab1, tab2, tab3, tab4 = st.tabs(["Target Selection", "Feature Creation", "Feature Selection", "Preprocessing"])
        
        with tab1:
            st.subheader("Target Variable Selection")
            
            # Target column selection
            potential_targets = df.columns.tolist()
            target_col = st.selectbox(
                "Select Target Column",
                potential_targets,
                help="Choose the column you want to predict"
            )
            
            if target_col:
                st.session_state.target_column = target_col
                
                # Analyze target variable
                st.subheader("Target Variable Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Target Distribution:**")
                    target_counts = df[target_col].value_counts()
                    st.dataframe(target_counts)
                    
                    # Check for class imbalance
                    if len(target_counts) == 2:
                        imbalance_ratio = target_counts.min() / target_counts.max()
                        if imbalance_ratio < 0.3:
                            st.warning(f"Class imbalance detected! Ratio: {imbalance_ratio:.2f}")
                        else:
                            st.success("Classes are reasonably balanced")
                
                with col2:
                    fig = visualizer.plot_target_distribution(df, target_col)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Feature Creation")
            
            # Date feature engineering
            date_cols = df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
            potential_date_cols = [col for col in date_cols if any(keyword in col.lower() for keyword in ['date', 'time'])]
            
            if potential_date_cols:
                st.subheader("Date Feature Engineering")
                selected_date_cols = st.multiselect("Select date columns", potential_date_cols)
                
                if selected_date_cols:
                    date_features = st.multiselect(
                        "Select date features to create",
                        ["year", "month", "day", "dayofweek", "quarter", "is_weekend", "days_since"]
                    )
                    
                    if st.button("Create Date Features"):
                        try:
                            df = data_processor.create_date_features(df, selected_date_cols, date_features)
                            st.success("Date features created successfully!")
                            st.session_state.processed_data = df
                        except Exception as e:
                            st.error(f"Error creating date features: {str(e)}")
            
            # Numerical feature engineering
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if st.session_state.target_column in numeric_cols:
                numeric_cols.remove(st.session_state.target_column)
            
            if numeric_cols:
                st.subheader("Numerical Feature Engineering")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    create_polynomial = st.checkbox("Create polynomial features")
                    if create_polynomial:
                        poly_degree = st.slider("Polynomial degree", 2, 3, 2)
                        poly_cols = st.multiselect("Select columns for polynomial features", numeric_cols)
                
                with col2:
                    create_interactions = st.checkbox("Create interaction features")
                    if create_interactions:
                        interaction_cols = st.multiselect("Select columns for interactions", numeric_cols)
                
                if st.button("Create Numerical Features"):
                    try:
                        if create_polynomial and poly_cols:
                            df = data_processor.create_polynomial_features(df, poly_cols, poly_degree)
                        
                        if create_interactions and len(interaction_cols) >= 2:
                            df = data_processor.create_interaction_features(df, interaction_cols)
                        
                        st.success("Numerical features created successfully!")
                        st.session_state.processed_data = df
                        
                    except Exception as e:
                        st.error(f"Error creating numerical features: {str(e)}")
        
        with tab3:
            st.subheader("Feature Selection")
            
            if st.session_state.target_column:
                # Separate features and target
                feature_cols = [col for col in df.columns if col != st.session_state.target_column]
                
                st.write(f"**Total Features Available:** {len(feature_cols)}")
                
                # Feature selection methods
                selection_method = st.selectbox(
                    "Feature Selection Method",
                    ["Manual Selection", "Correlation-based", "Univariate Statistical", "Recursive Feature Elimination"]
                )
                
                if selection_method == "Manual Selection":
                    selected_features = st.multiselect(
                        "Select features manually",
                        feature_cols,
                        default=feature_cols[:10] if len(feature_cols) > 10 else feature_cols
                    )
                    st.session_state.feature_columns = selected_features
                
                elif selection_method == "Correlation-based":
                    threshold = st.slider("Correlation threshold", 0.1, 0.9, 0.5)
                    
                    if st.button("Apply Correlation-based Selection"):
                        try:
                            selected_features = data_processor.select_features_by_correlation(
                                df, st.session_state.target_column, threshold
                            )
                            st.session_state.feature_columns = selected_features
                            st.success(f"Selected {len(selected_features)} features based on correlation")
                            st.write("Selected features:", selected_features)
                        except Exception as e:
                            st.error(f"Error in feature selection: {str(e)}")
                
                elif selection_method == "Univariate Statistical":
                    k_features = st.slider("Number of top features", 5, min(50, len(feature_cols)), 20)
                    
                    if st.button("Apply Statistical Selection"):
                        try:
                            selected_features = data_processor.select_features_statistical(
                                df, feature_cols, st.session_state.target_column, k_features
                            )
                            st.session_state.feature_columns = selected_features
                            st.success(f"Selected {len(selected_features)} features based on statistical tests")
                            st.write("Selected features:", selected_features)
                        except Exception as e:
                            st.error(f"Error in feature selection: {str(e)}")
                
                # Display selected features
                if st.session_state.feature_columns:
                    st.subheader("Selected Features")
                    st.write(f"**Number of selected features:** {len(st.session_state.feature_columns)}")
                    st.write(st.session_state.feature_columns)
        
        with tab4:
            st.subheader("Data Preprocessing")
            
            if st.session_state.feature_columns and st.session_state.target_column:
                
                # Encoding options
                st.subheader("Categorical Encoding")
                categorical_cols = df[st.session_state.feature_columns].select_dtypes(include=['object']).columns.tolist()
                
                if categorical_cols:
                    encoding_method = st.selectbox(
                        "Encoding method for categorical variables",
                        ["One-Hot Encoding", "Label Encoding", "Target Encoding"]
                    )
                    
                    # Handle high cardinality categories
                    handle_high_cardinality = st.checkbox("Handle high cardinality categories")
                    if handle_high_cardinality:
                        cardinality_threshold = st.slider("Cardinality threshold", 5, 50, 20)
                else:
                    st.info("No categorical columns found in selected features")
                
                # Scaling options
                st.subheader("Feature Scaling")
                scaling_method = st.selectbox(
                    "Scaling method",
                    ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
                )
                
                # Class imbalance handling
                st.subheader("Class Imbalance Handling")
                target_counts = df[st.session_state.target_column].value_counts()
                
                if len(target_counts) == 2:
                    imbalance_ratio = target_counts.min() / target_counts.max()
                    
                    if imbalance_ratio < 0.3:
                        st.warning(f"Class imbalance detected! Minority class ratio: {imbalance_ratio:.2f}")
                        
                        resampling_method = st.selectbox(
                            "Resampling method",
                            ["None", "SMOTE", "Random Over Sampling", "Random Under Sampling", "ADASYN"]
                        )
                    else:
                        resampling_method = "None"
                        st.success("Classes are reasonably balanced")
                else:
                    resampling_method = "None"
                
                # Apply preprocessing
                if st.button("Apply Preprocessing", type="primary"):
                    try:
                        X = df[st.session_state.feature_columns].copy()
                        y = df[st.session_state.target_column].copy()
                        
                        # Apply preprocessing
                        X_processed, y_processed, preprocessing_info = data_processor.preprocess_data(
                            X, y,
                            categorical_encoding=encoding_method if categorical_cols else None,
                            scaling_method=scaling_method,
                            resampling_method=resampling_method,
                            handle_high_cardinality=handle_high_cardinality if categorical_cols else False,
                            cardinality_threshold=cardinality_threshold if categorical_cols and handle_high_cardinality else 20
                        )
                        
                        # Save preprocessing info
                        analysis_id = db_manager.save_analysis(
                            st.session_state.dataset_id,
                            "preprocessing",
                            preprocessing_info
                        )
                        
                        # Store processed data
                        st.session_state.processed_data = pd.concat([X_processed, y_processed], axis=1)
                        st.session_state.feature_columns = X_processed.columns.tolist()
                        
                        st.success("Preprocessing completed successfully!")
                        st.write(f"**Final dataset shape:** {X_processed.shape}")
                        st.write(f"**Features after preprocessing:** {len(X_processed.columns)}")
                        
                        # Show preprocessing summary
                        st.json(preprocessing_info)
                        
                    except Exception as e:
                        st.error(f"Error in preprocessing: {str(e)}")
            else:
                st.warning("Please select features and target column first!")
                
    except Exception as e:
        st.error(f"Error in feature engineering: {str(e)}")

def step_4_model_building():
    st.header("🤖 Step 4: Model Building")
    
    if not st.session_state.processed_data is not None or not st.session_state.feature_columns or not st.session_state.target_column:
        st.warning("Please complete data preprocessing first!")
        return
    
    try:
        df = st.session_state.processed_data
        X = df[st.session_state.feature_columns]
        y = df[st.session_state.target_column]
        
        # Refresh button for current view
        col_refresh1, col_refresh2 = st.columns([6, 1])
        with col_refresh2:
            if st.button("🔄 Refresh", key="refresh_model_building"):
                st.rerun()
        
        tab1, tab2, tab3 = st.tabs(["Train-Test Split", "Model Training", "Hyperparameter Tuning"])
        
        with tab1:
            st.subheader("Train-Test Split")
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_size = st.slider("Test size ratio", 0.1, 0.5, 0.3)
                random_state = st.number_input("Random state", 0, 1000, 42)
                stratify = st.checkbox("Stratify split", value=True)
            
            with col2:
                st.write("**Dataset Information:**")
                st.write(f"Total samples: {len(df)}")
                st.write(f"Features: {len(st.session_state.feature_columns)}")
                st.write(f"Target: {st.session_state.target_column}")
                
                if stratify:
                    st.write("**Class distribution:**")
                    st.write(y.value_counts())
            
            if st.button("Create Train-Test Split", type="primary"):
                try:
                    X_train, X_test, y_train, y_test = model_builder.create_train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=stratify
                    )
                    
                    # Store in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    st.success("Train-test split created successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Training set:**")
                        st.write(f"Shape: {X_train.shape}")
                        st.write(y_train.value_counts())
                    
                    with col2:
                        st.write("**Test set:**")
                        st.write(f"Shape: {X_test.shape}")
                        st.write(y_test.value_counts())
                        
                except Exception as e:
                    st.error(f"Error creating train-test split: {str(e)}")
        
        with tab2:
            st.subheader("Model Training")
            
            if not all(key in st.session_state for key in ['X_train', 'X_test', 'y_train', 'y_test']):
                st.warning("Please create train-test split first!")
            else:
                # Model selection
                models_to_train = st.multiselect(
                    "Select models to train",
                    ["Logistic Regression", "Random Forest", "XGBoost", "SVM", "Naive Bayes"],
                    default=["Logistic Regression", "Random Forest"]
                )
                
                # Training configuration
                st.subheader("Training Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    cross_validation = st.checkbox("Use cross-validation", value=True)
                    if cross_validation:
                        cv_folds = st.slider("CV folds", 3, 10, 5)
                
                with col2:
                    calculate_feature_importance = st.checkbox("Calculate feature importance", value=True)
                    save_models = st.checkbox("Save trained models", value=True)
                
                if st.button("Train Models", type="primary"):
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        trained_models = {}
                        
                        for i, model_name in enumerate(models_to_train):
                            status_text.text(f"Training {model_name}...")
                            progress_bar.progress((i + 1) / len(models_to_train))
                            
                            # Train model
                            model_info = model_builder.train_model(
                                model_name,
                                st.session_state.X_train,
                                st.session_state.y_train,
                                cross_validation=cross_validation,
                                cv_folds=cv_folds if cross_validation else None
                            )
                            
                            trained_models[model_name] = model_info
                            
                            # Calculate feature importance if requested
                            if calculate_feature_importance:
                                try:
                                    importance = model_builder.get_feature_importance(
                                        model_info['model'], st.session_state.feature_columns
                                    )
                                    model_info['feature_importance'] = importance
                                except:
                                    pass
                        
                        # Store models in session state
                        st.session_state.models = trained_models
                        
                        status_text.text("Training completed!")
                        st.success(f"Successfully trained {len(models_to_train)} models!")
                        
                        # Display training results
                        st.subheader("Training Results")
                        
                        for model_name, model_info in trained_models.items():
                            with st.expander(f"{model_name} Results"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if 'cv_scores' in model_info:
                                        st.write("**Cross-validation scores:**")
                                        st.write(f"Mean: {model_info['cv_scores'].mean():.4f}")
                                        st.write(f"Std: {model_info['cv_scores'].std():.4f}")
                                
                                with col2:
                                    if 'feature_importance' in model_info:
                                        st.write("**Top 10 Important Features:**")
                                        top_features = model_info['feature_importance'].head(10)
                                        st.dataframe(top_features)
                        
                        # Save models to database if requested
                        if save_models:
                            for model_name, model_info in trained_models.items():
                                analysis_id = db_manager.save_analysis(
                                    st.session_state.dataset_id,
                                    f"model_{model_name.lower().replace(' ', '_')}",
                                    {
                                        'model_name': model_name,
                                        'cv_scores': model_info.get('cv_scores', []).tolist() if 'cv_scores' in model_info else [],
                                        'feature_importance': model_info.get('feature_importance', pd.DataFrame()).to_dict() if 'feature_importance' in model_info else {}
                                    }
                                )
                        
                    except Exception as e:
                        st.error(f"Error training models: {str(e)}")
        
        with tab3:
            st.subheader("Hyperparameter Tuning")
            
            if not st.session_state.models:
                st.warning("Please train models first!")
            else:
                # Select model for tuning
                model_to_tune = st.selectbox(
                    "Select model for hyperparameter tuning",
                    list(st.session_state.models.keys())
                )
                
                if model_to_tune:
                    st.subheader(f"Tuning {model_to_tune}")
                    
                    # Get hyperparameter space for the selected model
                    param_space = model_builder.get_hyperparameter_space(model_to_tune)
                    
                    # Display parameter options
                    st.write("**Available parameters:**")
                    for param, values in param_space.items():
                        st.write(f"- {param}: {values}")
                    
                    # Tuning configuration
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        tuning_method = st.selectbox(
                            "Tuning method",
                            ["Grid Search", "Random Search"]
                        )
                        
                        scoring_metric = st.selectbox(
                            "Scoring metric",
                            ["accuracy", "precision", "recall", "f1", "roc_auc"]
                        )
                    
                    with col2:
                        cv_folds_tuning = st.slider("CV folds for tuning", 3, 10, 5)
                        
                        if tuning_method == "Random Search":
                            n_iter = st.slider("Number of iterations", 10, 100, 50)
                        else:
                            n_iter = None
                    
                    if st.button("Start Hyperparameter Tuning", type="primary"):
                        try:
                            with st.spinner("Tuning hyperparameters... This may take a while."):
                                best_model, best_params, best_score = model_builder.tune_hyperparameters(
                                    model_to_tune,
                                    st.session_state.X_train,
                                    st.session_state.y_train,
                                    param_space,
                                    method=tuning_method.lower().replace(' ', '_'),
                                    scoring=scoring_metric,
                                    cv=cv_folds_tuning,
                                    n_iter=n_iter
                                )
                            
                            # Update model in session state
                            st.session_state.models[model_to_tune]['tuned_model'] = best_model
                            st.session_state.models[model_to_tune]['best_params'] = best_params
                            st.session_state.models[model_to_tune]['best_score'] = best_score
                            
                            st.success(f"Hyperparameter tuning completed!")
                            st.write(f"**Best {scoring_metric} score:** {best_score:.4f}")
                            st.write("**Best parameters:**")
                            st.json(best_params)
                            
                            # Save tuning results
                            analysis_id = db_manager.save_analysis(
                                st.session_state.dataset_id,
                                f"tuning_{model_to_tune.lower().replace(' ', '_')}",
                                {
                                    'model_name': model_to_tune,
                                    'best_params': best_params,
                                    'best_score': best_score,
                                    'tuning_method': tuning_method,
                                    'scoring_metric': scoring_metric
                                }
                            )
                            
                        except Exception as e:
                            st.error(f"Error in hyperparameter tuning: {str(e)}")
                            
    except Exception as e:
        st.error(f"Error in model building: {str(e)}")

def step_5_model_evaluation():
    st.header("📊 Step 5: Model Evaluation")
    
    if not st.session_state.models or not all(key in st.session_state for key in ['X_test', 'y_test']):
        st.warning("Please train models first!")
        return
    
    try:
        # Refresh button for current view
        col_refresh1, col_refresh2 = st.columns([6, 1])
        with col_refresh2:
            if st.button("🔄 Refresh", key="refresh_model_eval"):
                st.rerun()
        
        tab1, tab2, tab3, tab4 = st.tabs(["Performance Metrics", "ROC & Precision-Recall", "Feature Importance", "Model Comparison"])
        
        with tab1:
            st.subheader("Model Performance Metrics")
            
            # Evaluate all models
            evaluation_results = {}
            
            for model_name, model_info in st.session_state.models.items():
                # Use tuned model if available, otherwise use original model
                model = model_info.get('tuned_model', model_info['model'])
                
                # Make predictions
                y_pred = model.predict(st.session_state.X_test)
                y_pred_proba = None
                
                try:
                    y_pred_proba = model.predict_proba(st.session_state.X_test)[:, 1]
                except:
                    pass
                
                # Calculate metrics
                metrics = model_builder.evaluate_model(st.session_state.y_test, y_pred, y_pred_proba)
                evaluation_results[model_name] = {
                    'metrics': metrics,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
            
            # Display metrics comparison
            metrics_df = pd.DataFrame({
                model_name: results['metrics'] 
                for model_name, results in evaluation_results.items()
            }).T
            
            st.dataframe(metrics_df.round(4))
            
            # Visualize metrics comparison
            fig = visualizer.plot_metrics_comparison(metrics_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results for each model
            for model_name, results in evaluation_results.items():
                with st.expander(f"{model_name} Detailed Results"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Classification Metrics:**")
                        for metric, value in results['metrics'].items():
                            st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
                    
                    with col2:
                        # Confusion matrix
                        fig = visualizer.plot_confusion_matrix(
                            st.session_state.y_test, results['y_pred'], model_name
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("ROC Curves and Precision-Recall Curves")
            
            # ROC Curves
            st.subheader("ROC Curves")
            roc_data = []
            
            for model_name, results in evaluation_results.items():
                if results['y_pred_proba'] is not None:
                    roc_data.append({
                        'model_name': model_name,
                        'y_true': st.session_state.y_test,
                        'y_scores': results['y_pred_proba']
                    })
            
            if roc_data:
                fig = visualizer.plot_roc_curves(roc_data)
                st.plotly_chart(fig, use_container_width=True)
            
            # Precision-Recall Curves
            st.subheader("Precision-Recall Curves")
            if roc_data:
                fig = visualizer.plot_precision_recall_curves(roc_data)
                st.plotly_chart(fig, use_container_width=True)
            
            # Individual model analysis
            selected_model = st.selectbox(
                "Select model for detailed analysis",
                list(evaluation_results.keys())
            )
            
            if selected_model and evaluation_results[selected_model]['y_pred_proba'] is not None:
                st.subheader(f"{selected_model} - Threshold Analysis")
                
                # Threshold optimization
                thresholds, precisions, recalls, f1_scores = model_builder.analyze_thresholds(
                    st.session_state.y_test, evaluation_results[selected_model]['y_pred_proba']
                )
                
                fig = visualizer.plot_threshold_analysis(thresholds, precisions, recalls, f1_scores)
                st.plotly_chart(fig, use_container_width=True)
                
                # Optimal threshold
                optimal_threshold_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_threshold_idx]
                
                st.write(f"**Optimal threshold (max F1):** {optimal_threshold:.3f}")
                st.write(f"**F1 score at optimal threshold:** {f1_scores[optimal_threshold_idx]:.4f}")
                st.write(f"**Precision at optimal threshold:** {precisions[optimal_threshold_idx]:.4f}")
                st.write(f"**Recall at optimal threshold:** {recalls[optimal_threshold_idx]:.4f}")
        
        with tab3:
            st.subheader("Feature Importance Analysis")
            
            for model_name, model_info in st.session_state.models.items():
                if 'feature_importance' in model_info:
                    st.subheader(f"{model_name} - Feature Importance")
                    
                    importance_df = model_info['feature_importance']
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.dataframe(importance_df.head(15))
                    
                    with col2:
                        fig = visualizer.plot_feature_importance(importance_df.head(15))
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Model Comparison Summary")
            
            # Best model selection
            best_model_metric = st.selectbox(
                "Select metric for best model selection",
                ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
            )
            
            if best_model_metric in metrics_df.columns:
                best_model = metrics_df[best_model_metric].idxmax()
                best_score = metrics_df.loc[best_model, best_model_metric]
                
                st.success(f"**Best model based on {best_model_metric}:** {best_model}")
                st.write(f"**Score:** {best_score:.4f}")
            
            # Model summary table
            st.subheader("Complete Model Comparison")
            st.dataframe(metrics_df.round(4))
            
            # Save evaluation results
            analysis_id = db_manager.save_analysis(
                st.session_state.dataset_id,
                "model_evaluation",
                {
                    'evaluation_results': {
                        model_name: {
                            'metrics': results['metrics']
                        }
                        for model_name, results in evaluation_results.items()
                    },
                    'best_model': best_model,
                    'best_score': best_score,
                    'evaluation_metric': best_model_metric
                }
            )
            
    except Exception as e:
        st.error(f"Error in model evaluation: {str(e)}")

def step_6_predictions():
    st.header("🎯 Step 6: Predictions & Deployment")
    
    if not st.session_state.models:
        st.warning("Please train models first!")
        return
    
    try:
        # Refresh button for current view
        col_refresh1, col_refresh2 = st.columns([6, 1])
        with col_refresh2:
            if st.button("🔄 Refresh", key="refresh_predictions"):
                st.rerun()
        
        tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Export"])
        
        with tab1:
            st.subheader("Single Prediction")
            
            # Model selection
            selected_model = st.selectbox(
                "Select model for prediction",
                list(st.session_state.models.keys())
            )
            
            if selected_model:
                model_info = st.session_state.models[selected_model]
                model = model_info.get('tuned_model', model_info['model'])
                
                st.subheader("Enter Feature Values")
                
                # Create input fields for each feature
                feature_values = {}
                
                # Get feature information from the processed data
                sample_data = st.session_state.processed_data[st.session_state.feature_columns].iloc[0]
                
                cols = st.columns(3)
                
                for i, feature in enumerate(st.session_state.feature_columns):
                    col_idx = i % 3
                    
                    with cols[col_idx]:
                        sample_value = sample_data[feature]
                        
                        if isinstance(sample_value, (int, float)):
                            # Numerical feature
                            min_val = float(st.session_state.processed_data[feature].min())
                            max_val = float(st.session_state.processed_data[feature].max())
                            default_val = float(st.session_state.processed_data[feature].median())
                            
                            feature_values[feature] = st.number_input(
                                feature,
                                min_value=min_val,
                                max_value=max_val,
                                value=default_val,
                                key=f"feature_{i}"
                            )
                        else:
                            # Categorical feature (assuming it's already encoded)
                            unique_values = st.session_state.processed_data[feature].unique()
                            default_val = unique_values[0]
                            
                            feature_values[feature] = st.selectbox(
                                feature,
                                unique_values,
                                index=0,
                                key=f"feature_{i}"
                            )
                
                if st.button("Make Prediction", type="primary"):
                    try:
                        # Create input dataframe
                        input_df = pd.DataFrame([feature_values])
                        
                        # Make prediction
                        prediction = model.predict(input_df)[0]
                        
                        try:
                            prediction_proba = model.predict_proba(input_df)[0]
                            confidence = max(prediction_proba)
                        except:
                            prediction_proba = None
                            confidence = None
                        
                        # Display results
                        st.subheader("Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if prediction == 1:
                                st.error(f"**Prediction: FRAUDULENT**")
                            else:
                                st.success(f"**Prediction: LEGITIMATE**")
                        
                        with col2:
                            if confidence:
                                st.metric("Confidence", f"{confidence:.2%}")
                        
                        if prediction_proba is not None:
                            st.subheader("Prediction Probabilities")
                            prob_df = pd.DataFrame({
                                'Class': ['Legitimate', 'Fraudulent'],
                                'Probability': prediction_proba
                            })
                            
                            fig = visualizer.plot_prediction_probabilities(prob_df)
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
        
        with tab2:
            st.subheader("Batch Prediction")
            
            # File upload for batch prediction
            uploaded_file = st.file_uploader(
                "Upload file for batch prediction",
                type=['csv', 'xlsx', 'xls'],
                help="Upload a file with the same features as the training data"
            )
            
            if uploaded_file is not None:
                try:
                    # Read file
                    if uploaded_file.name.endswith('.csv'):
                        batch_df = pd.read_csv(uploaded_file)
                    else:
                        batch_df = pd.read_excel(uploaded_file)
                    
                    st.write("**Uploaded data preview:**")
                    st.dataframe(batch_df.head())
                    
                    # Check if required features are present
                    missing_features = set(st.session_state.feature_columns) - set(batch_df.columns)
                    
                    if missing_features:
                        st.error(f"Missing features: {missing_features}")
                    else:
                        # Model selection for batch prediction
                        batch_model = st.selectbox(
                            "Select model for batch prediction",
                            list(st.session_state.models.keys()),
                            key="batch_model"
                        )
                        
                        if st.button("Run Batch Prediction", type="primary"):
                            try:
                                model_info = st.session_state.models[batch_model]
                                model = model_info.get('tuned_model', model_info['model'])
                                
                                # Prepare data
                                X_batch = batch_df[st.session_state.feature_columns]
                                
                                # Make predictions
                                predictions = model.predict(X_batch)
                                
                                try:
                                    prediction_probas = model.predict_proba(X_batch)
                                    fraud_probabilities = prediction_probas[:, 1]
                                except:
                                    fraud_probabilities = None
                                
                                # Add predictions to dataframe
                                result_df = batch_df.copy()
                                result_df['Prediction'] = ['Fraudulent' if p == 1 else 'Legitimate' for p in predictions]
                                
                                if fraud_probabilities is not None:
                                    result_df['Fraud_Probability'] = fraud_probabilities
                                    result_df['Confidence'] = np.maximum(fraud_probabilities, 1 - fraud_probabilities)
                                
                                st.success(f"Batch prediction completed! {len(result_df)} records processed.")
                                
                                # Display results summary
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    fraud_count = sum(predictions)
                                    st.metric("Fraudulent Cases", fraud_count)
                                
                                with col2:
                                    legitimate_count = len(predictions) - fraud_count
                                    st.metric("Legitimate Cases", legitimate_count)
                                
                                with col3:
                                    fraud_rate = fraud_count / len(predictions) * 100
                                    st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
                                
                                # Show results
                                st.subheader("Prediction Results")
                                st.dataframe(result_df)
                                
                                # Download results
                                csv = result_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Results as CSV",
                                    data=csv,
                                    file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                                
                            except Exception as e:
                                st.error(f"Error in batch prediction: {str(e)}")
                                
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        with tab3:
            st.subheader("Model Export & Deployment")
            
            # Model selection for export
            export_model = st.selectbox(
                "Select model to export",
                list(st.session_state.models.keys()),
                key="export_model"
            )
            
            if export_model:
                model_info = st.session_state.models[export_model]
                model = model_info.get('tuned_model', model_info['model'])
                
                st.subheader("Model Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Model Type:** {export_model}")
                    st.write(f"**Features:** {len(st.session_state.feature_columns)}")
                    st.write(f"**Target:** {st.session_state.target_column}")
                
                with col2:
                    if 'best_params' in model_info:
                        st.write("**Hyperparameters:**")
                        st.json(model_info['best_params'])
                
                # Export options
                st.subheader("Export Options")
                
                export_format = st.selectbox(
                    "Export format",
                    ["Pickle", "Joblib"]
                )
                
                include_preprocessing = st.checkbox("Include preprocessing steps", value=True)
                include_feature_names = st.checkbox("Include feature names", value=True)
                
                if st.button("Export Model", type="primary"):
                    try:
                        # Prepare export data
                        export_data = {
                            'model': model,
                            'model_name': export_model,
                            'feature_columns': st.session_state.feature_columns,
                            'target_column': st.session_state.target_column,
                            'export_timestamp': datetime.now().isoformat()
                        }
                        
                        if 'best_params' in model_info:
                            export_data['best_params'] = model_info['best_params']
                        
                        if include_preprocessing:
                            # Add preprocessing information
                            preprocessing_analysis = db_manager.get_analysis(st.session_state.dataset_id, "preprocessing")
                            if preprocessing_analysis:
                                export_data['preprocessing_info'] = preprocessing_analysis
                        
                        # Serialize model
                        buffer = BytesIO()
                        if export_format == "Pickle":
                            pickle.dump(export_data, buffer)
                            file_extension = "pkl"
                        else:
                            import joblib
                            joblib.dump(export_data, buffer)
                            file_extension = "joblib"
                        
                        buffer.seek(0)
                        
                        # Create download
                        filename = f"{export_model.lower().replace(' ', '_')}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
                        
                        st.download_button(
                            label=f"Download {export_model} Model",
                            data=buffer.getvalue(),
                            file_name=filename,
                            mime="application/octet-stream"
                        )
                        
                        st.success("Model exported successfully!")
                        
                        # Display export summary
                        st.subheader("Export Summary")
                        st.write(f"**Filename:** {filename}")
                        st.write(f"**Format:** {export_format}")
                        st.write(f"**Size:** {len(buffer.getvalue()) / 1024:.2f} KB")
                        
                    except Exception as e:
                        st.error(f"Error exporting model: {str(e)}")
                
                # API endpoint information
                st.subheader("API Deployment")
                st.info("""
                **Deployment Instructions:**
                
                1. Download the exported model file
                2. Set up a Flask/FastAPI server
                3. Load the model and create prediction endpoints
                4. Ensure the same preprocessing steps are applied to new data
                
                **Example API endpoint:**
                ```python
                @app.post("/predict")
                def predict(features: dict):
                    # Load your exported model
                    model_data = pickle.load(open('model.pkl', 'rb'))
                    model = model_data['model']
                    
                    # Apply preprocessing and make prediction
                    prediction = model.predict([list(features.values())])
                    return {"prediction": int(prediction[0])}
                ```
                """)
                
    except Exception as e:
        st.error(f"Error in predictions: {str(e)}")

def step_7_database_management():
    st.header("🗄️ Step 7: Database Management")
    
    try:
        # Refresh button for current view
        col_refresh1, col_refresh2 = st.columns([6, 1])
        with col_refresh2:
            if st.button("🔄 Refresh", key="refresh_database"):
                st.rerun()
        
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Datasets", "Models", "Analytics"])
        
        with tab1:
            st.subheader("Database Overview")
            
            # Get database statistics
            stats = db_manager.get_database_stats()
            
            # Display statistics cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Datasets", stats['datasets'])
            
            with col2:
                st.metric("Total Models", stats['models'])
            
            with col3:
                st.metric("Total Analyses", stats['analyses'])
            
            with col4:
                st.metric("Total Predictions", stats['predictions'])
            
            # Database connection info
            st.subheader("Database Connection")
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                db_info = get_database_info()
                st.success(f"{db_info['type']} Database Connected")
                st.write(f"**Database Type:** {db_info['type']}")
                st.write(f"**Status:** {db_info['status']}")
                if 'file' in db_info:
                    st.write(f"**File:** {db_info['file']}")
                elif 'url' in db_info:
                    st.write("**Connection:** Configured")
                
            with col_info2:
                st.write("**Database Features:**")
                features = db_info.get('features', [])
                for feature in features:
                    st.write(f"• {feature}")
                
                st.write("**Switch Database:**")
                st.code("""
Current: SQLite (Local)
Available: PostgreSQL 

Set DATABASE_TYPE=postgresql 
to use PostgreSQL backend
                """)
            
            # Storage usage visualization
            if stats['datasets'] > 0 or stats['models'] > 0:
                st.subheader("Storage Distribution")
                
                chart_data = {
                    'Category': ['Datasets', 'Models', 'Analyses', 'Predictions'],
                    'Count': [stats['datasets'], stats['models'], stats['analyses'], stats['predictions']]
                }
                
                fig = visualizer.plot_categorical_distribution(pd.DataFrame(chart_data), 'Category')
                fig.update_layout(title="Database Content Distribution", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Dataset Management")
            
            # List all datasets
            datasets = db_manager.get_all_datasets()
            
            if datasets:
                st.write(f"**Found {len(datasets)} datasets in database:**")
                
                # Create dataframe for display
                df_datasets = pd.DataFrame(datasets, columns=[
                    'ID', 'Name', 'Description', 'Upload Date', 'File Size (bytes)', 'Shape'
                ])
                
                # Format the display
                df_datasets['Upload Date'] = pd.to_datetime(df_datasets['Upload Date']).dt.strftime('%Y-%m-%d %H:%M')
                df_datasets['File Size'] = df_datasets['File Size (bytes)'].apply(lambda x: f"{x/1024:.2f} KB" if x < 1024*1024 else f"{x/(1024*1024):.2f} MB")
                df_datasets['Rows x Cols'] = df_datasets['Shape'].apply(lambda x: f"{x[0]} x {x[1]}")
                
                # Display table
                display_df = df_datasets[['ID', 'Name', 'Description', 'Upload Date', 'File Size', 'Rows x Cols']]
                st.dataframe(display_df, use_container_width=True)
                
                # Dataset management actions
                st.subheader("Dataset Actions")
                
                col_action1, col_action2 = st.columns(2)
                
                with col_action1:
                    selected_dataset_id = st.selectbox(
                        "Select dataset for actions",
                        [row[0] for row in datasets],
                        format_func=lambda x: f"{dict([(row[0], f'{row[1]} (ID: {row[0]})') for row in datasets])[x]}"
                    )
                
                with col_action2:
                    action = st.selectbox("Action", ["View Details", "Download", "Delete"])
                
                if st.button("Execute Action", type="primary"):
                    if action == "View Details":
                        try:
                            dataset = db_manager.load_dataset(selected_dataset_id)
                            st.subheader(f"Dataset Details (ID: {selected_dataset_id})")
                            
                            col_det1, col_det2 = st.columns(2)
                            
                            with col_det1:
                                st.write("**Basic Information:**")
                                st.write(f"Shape: {dataset.shape}")
                                st.write(f"Memory Usage: {dataset.memory_usage(deep=True).sum() / 1024:.2f} KB")
                                st.write(f"Columns: {len(dataset.columns)}")
                                
                            with col_det2:
                                st.write("**Data Types:**")
                                st.write(dataset.dtypes.value_counts())
                            
                            st.write("**Sample Data:**")
                            st.dataframe(dataset.head(10))
                            
                            st.write("**Column Information:**")
                            info_df = pd.DataFrame({
                                'Column': dataset.columns,
                                'Type': dataset.dtypes,
                                'Non-Null': dataset.count(),
                                'Null Count': dataset.isnull().sum(),
                                'Unique Values': dataset.nunique()
                            })
                            st.dataframe(info_df)
                            
                        except Exception as e:
                            st.error(f"Error loading dataset: {str(e)}")
                    
                    elif action == "Download":
                        try:
                            dataset = db_manager.load_dataset(selected_dataset_id)
                            csv_data = dataset.to_csv(index=False)
                            
                            dataset_name = dict([(row[0], row[1]) for row in datasets])[selected_dataset_id]
                            filename = f"{dataset_name.lower().replace(' ', '_')}_{selected_dataset_id}.csv"
                            
                            st.download_button(
                                label="Download Dataset as CSV",
                                data=csv_data,
                                file_name=filename,
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Error preparing download: {str(e)}")
                    
                    elif action == "Delete":
                        if st.session_state.get(f'confirm_delete_{selected_dataset_id}', False):
                            try:
                                success = db_manager.delete_dataset(selected_dataset_id)
                                if success:
                                    st.success(f"Dataset {selected_dataset_id} deleted successfully!")
                                    st.session_state[f'confirm_delete_{selected_dataset_id}'] = False
                                    st.rerun()
                                else:
                                    st.error("Failed to delete dataset")
                            except Exception as e:
                                st.error(f"Error deleting dataset: {str(e)}")
                        else:
                            st.warning("⚠️ This action cannot be undone!")
                            if st.button("Confirm Deletion", key=f"confirm_{selected_dataset_id}"):
                                st.session_state[f'confirm_delete_{selected_dataset_id}'] = True
                                st.rerun()
            else:
                st.info("No datasets found in database. Upload some data in Step 1 to get started.")
        
        with tab3:
            st.subheader("Model Management")
            
            # List all models
            models = db_manager.get_all_models()
            
            if models:
                st.write(f"**Found {len(models)} models in database:**")
                
                # Create dataframe for display
                df_models = pd.DataFrame(models, columns=[
                    'ID', 'Dataset ID', 'Model Name', 'Model Type', 'Created Date', 'Accuracy'
                ])
                
                # Format the display
                df_models['Created Date'] = pd.to_datetime(df_models['Created Date']).dt.strftime('%Y-%m-%d %H:%M')
                df_models['Accuracy'] = df_models['Accuracy'].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else str(x))
                
                st.dataframe(df_models, use_container_width=True)
                
                # Model performance visualization
                if len(models) > 1:
                    st.subheader("Model Performance Comparison")
                    
                    # Filter models with numeric accuracy
                    numeric_models = df_models[pd.to_numeric(df_models['Accuracy'], errors='coerce').notna()].copy()
                    
                    if not numeric_models.empty:
                        numeric_models['Accuracy_Numeric'] = pd.to_numeric(numeric_models['Accuracy'])
                        
                        fig = visualizer.plot_distribution(numeric_models, 'Accuracy_Numeric')
                        fig.update_layout(title="Model Accuracy Distribution", xaxis_title="Accuracy")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Model management actions
                st.subheader("Model Actions")
                
                selected_model_id = st.selectbox(
                    "Select model",
                    [row[0] for row in models],
                    format_func=lambda x: f"{dict([(row[0], f'{row[2]} - {row[3]} (ID: {row[0]})') for row in models])[x]}"
                )
                
                if st.button("Load Model Details", type="primary"):
                    try:
                        model_data = db_manager.load_model(selected_model_id)
                        
                        st.subheader(f"Model Details (ID: {selected_model_id})")
                        
                        col_mod1, col_mod2 = st.columns(2)
                        
                        with col_mod1:
                            st.write("**Basic Information:**")
                            st.write(f"Name: {model_data['model_name']}")
                            st.write(f"Type: {model_data['model_type']}")
                            st.write(f"Dataset ID: {model_data['dataset_id']}")
                            st.write(f"Created: {model_data['created_date']}")
                        
                        with col_mod2:
                            if model_data['metrics']:
                                st.write("**Performance Metrics:**")
                                st.json(model_data['metrics'])
                        
                        if model_data['parameters']:
                            st.write("**Model Parameters:**")
                            st.json(model_data['parameters'])
                        
                        # Model export option
                        st.subheader("Export Model")
                        
                        if st.button("Export This Model"):
                            buffer = BytesIO()
                            pickle.dump(model_data, buffer)
                            buffer.seek(0)
                            
                            filename = f"{model_data['model_name'].lower().replace(' ', '_')}_exported.pkl"
                            
                            st.download_button(
                                label="Download Model File",
                                data=buffer.getvalue(),
                                file_name=filename,
                                mime="application/octet-stream"
                            )
                    
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
            else:
                st.info("No models found in database. Train some models in Step 4 to see them here.")
        
        with tab4:
            st.subheader("Analytics & Insights")
            
            # Database usage analytics
            if stats['datasets'] > 0:
                st.subheader("Usage Analytics")
                
                # Get all datasets for timeline
                datasets = db_manager.get_all_datasets()
                if datasets:
                    # Create timeline chart
                    df_timeline = pd.DataFrame(datasets, columns=[
                        'ID', 'Name', 'Description', 'Upload Date', 'File Size', 'Shape'
                    ])
                    
                    df_timeline['Upload Date'] = pd.to_datetime(df_timeline['Upload Date'])
                    df_timeline['Month'] = df_timeline['Upload Date'].dt.to_period('M')
                    
                    monthly_uploads = df_timeline.groupby('Month').size().reset_index(name='Uploads')
                    monthly_uploads['Month'] = monthly_uploads['Month'].astype(str)
                    
                    if len(monthly_uploads) > 0:
                        st.write("**Dataset Uploads Over Time:**")
                        fig = visualizer.plot_distribution(monthly_uploads, 'Uploads')
                        fig.update_layout(title="Monthly Dataset Uploads", xaxis_title="Month")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Storage analytics
                models = db_manager.get_all_models()
                if models:
                    df_models_analytics = pd.DataFrame(models, columns=[
                        'ID', 'Dataset ID', 'Model Name', 'Model Type', 'Created Date', 'Accuracy'
                    ])
                    
                    # Model type distribution
                    model_type_counts = df_models_analytics['Model Type'].value_counts().reset_index()
                    model_type_counts.columns = ['Model Type', 'Count']
                    
                    st.write("**Model Type Distribution:**")
                    fig = visualizer.plot_categorical_distribution(model_type_counts, 'Model Type')
                    fig.update_layout(title="Distribution of Model Types", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Database maintenance
            st.subheader("Database Maintenance")
            
            col_maint1, col_maint2 = st.columns(2)
            
            with col_maint1:
                st.write("**Cleanup Options:**")
                days_old = st.slider("Remove data older than (days)", 7, 90, 30)
                
                if st.button("Clean Old Data", type="secondary"):
                    try:
                        db_manager.cleanup_old_data(days_old)
                        st.success(f"Cleaned up data older than {days_old} days")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during cleanup: {str(e)}")
            
            with col_maint2:
                st.write("**Health Check:**")
                
                if st.button("Run Health Check"):
                    try:
                        # Test database connection
                        test_stats = db_manager.get_database_stats()
                        
                        st.success("✅ Database connection: OK")
                        st.success(f"✅ Tables accessible: {len(test_stats)} table types")
                        st.success("✅ Data integrity: OK")
                        
                        # Additional checks
                        total_records = sum(test_stats.values())
                        if total_records > 1000:
                            st.warning("⚠️ Large dataset detected - consider cleanup")
                        else:
                            st.success("✅ Database size: Optimal")
                    
                    except Exception as e:
                        st.error(f"❌ Health check failed: {str(e)}")
            
            # Advanced options
            st.subheader("Advanced Operations")
            
            with st.expander("Database Schema Information"):
                st.code("""
Database Schema:

1. datasets
   - id (Primary Key)
   - name, description
   - upload_date, file_size
   - columns_info (JSON)
   - data_blob (Binary)

2. analysis
   - id (Primary Key)
   - dataset_id (Foreign Key)
   - analysis_type, results (JSON)
   - created_date

3. models
   - id (Primary Key) 
   - dataset_id (Foreign Key)
   - model_name, model_type
   - model_data (Binary)
   - metrics, parameters (JSON)
   - created_date

4. predictions
   - id (Primary Key)
   - model_id (Foreign Key)
   - input_data, prediction_result (JSON)
   - confidence_score
   - created_date
                """)
            
            with st.expander("Raw SQL Query Interface"):
                st.warning("⚠️ Advanced users only. Incorrect queries may damage data.")
                
                sql_query = st.text_area(
                    "Enter SQL query:",
                    placeholder="SELECT * FROM datasets LIMIT 5;",
                    help="Execute raw SQL queries against the database"
                )
                
                if st.button("Execute Query", type="secondary"):
                    if sql_query.strip():
                        try:
                            # For safety, only allow SELECT queries
                            if sql_query.strip().upper().startswith('SELECT'):
                                import sqlite3
                                conn = sqlite3.connect('fraud_detection.db')
                                df_result = pd.read_sql_query(sql_query, conn)
                                conn.close()
                                st.dataframe(df_result)
                            else:
                                st.error("Only SELECT queries are allowed for safety")
                        except Exception as e:
                            st.error(f"Query error: {str(e)}")
                    else:
                        st.error("Please enter a query")
        
    except Exception as e:
        st.error(f"Error in database management: {str(e)}")

if __name__ == "__main__":
    main()
