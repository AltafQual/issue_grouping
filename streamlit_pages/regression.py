import asyncio
import json
import logging
import traceback
from typing import Any, Dict

import nest_asyncio
import pandas as pd
import requests
import streamlit as st

from src.constants import DataFrameKeys
from src.helpers import get_tc_ids_from_sql

# Apply nest_asyncio to avoid asyncio loop issues
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="Regression Analysis", page_icon="📊", layout="wide")

COL_TO_SHOW = ["tc_uuid", "soc_name", "reason", "model_name", "tags", "feature_name", DataFrameKeys.cluster_class]

# Initialize session state variables
if "tc_ids_options" not in st.session_state:
    try:
        with st.spinner("Getting TC IDs from SQL..."):
            st.session_state.tc_ids_options = get_tc_ids_from_sql()
    except Exception as e:
        traceback.print_exc()
        logger.info(f"Failed to load test case IDs: {str(e)}")
        st.session_state.tc_ids_options = pd.DataFrame()

if "regression_results" not in st.session_state:
    st.session_state.regression_results = None

if "last_run_ids" not in st.session_state:
    st.session_state.last_run_ids = {"run_id_a": None, "run_id_b": None}

st.title("Regression Analysis 📊")
with st.expander("How to use this tool"):
    st.markdown(
        """
    1. Select two Run IDs to compare:
       - The first Run ID serves as the base for comparison
       - The second Run ID is compared against the base
    
    2. Click "Analyze Regression" to start the analysis
    
    3. Results are organized in two ways:
       - **Type-based Analysis**: Shows clusters organized by type and runtime
       - **Model-based Analysis**: Shows clusters organized by model name
    
    4. For each cluster, you can see:
       - The cluster name and class
       - Details about the failures in that cluster
       - Differences between the two runs
    """
    )


# Function to call the API for regression analysis
def get_regression_data(run_id_a: str, run_id_b: str) -> Dict[str, Any]:
    print("making api call to get the cluster info")
    try:
        url = "http://hyd-e160-a18-05:8001/api/get_two_run_ids_cluster_info/"
        payload = {"run_id_a": run_id_a, "run_id_b": run_id_b}
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling regression API: {str(e)}")
        traceback.print_exc()
        return None


# Main form for input
with st.form("regression_form"):
    col1, col2 = st.columns(2)

    tc_options = []
    if not st.session_state.tc_ids_options.empty:
        try:
            tc_options = st.session_state.tc_ids_options["testplan_id"].tolist()
        except Exception:
            traceback.print_exc()
            pass

    with col1:
        run_id_a = st.selectbox("Select First Run ID (Base)", options=tc_options, index=None, key="run_id_a_select")

    with col2:
        run_id_b = st.selectbox("Select Second Run ID (Compare)", options=tc_options, index=None, key="run_id_b_select")

    analyze_button = st.form_submit_button("Analyze Regression")

# Process form submission
if analyze_button:
    if not run_id_a or not run_id_b:
        st.warning("Please select both Run IDs to continue.")
    elif run_id_a == run_id_b:
        st.warning("Please select different Run IDs for comparison.")
    else:
        # Check if we need to fetch new data or can use cached results
        if (
            run_id_a != st.session_state.last_run_ids["run_id_a"]
            or run_id_b != st.session_state.last_run_ids["run_id_b"]
        ):

            with st.spinner(f"Analyzing regression between {run_id_a} and {run_id_b}..."):
                try:
                    # Call API to get regression data
                    regression_data = get_regression_data(run_id_a, run_id_b)

                    if regression_data and regression_data.get("status") == 200:
                        st.session_state.regression_results = regression_data
                        st.session_state.last_run_ids = {"run_id_a": run_id_a, "run_id_b": run_id_b}
                        st.success(f"Analysis completed in {regression_data.get('time_taken', 0)} seconds!")
                    else:
                        if regression_data and regression_data.get("status") == 404:
                            st.warning("No regression data found between these two runs.")
                        else:
                            st.error("Failed to retrieve regression data. Please try again.")
                        st.session_state.regression_results = None
                except Exception as e:
                    st.error(f"Error during regression analysis: {str(e)}")
                    traceback.print_exc()
                    st.session_state.regression_results = None
        else:
            st.info("Using cached results for the same Run IDs.")

# Display results if available
if st.session_state.regression_results:
    results = st.session_state.regression_results

    st.header("Regression Analysis Results")
    st.markdown(f"**Base Run ID:** {results['run_id_a']}")
    st.markdown(f"**Compare Run ID:** {results['run_id_b']}")

    # Display Type-based results
    if results.get("type"):
        st.subheader("Type-based Analysis")

        # Create tabs for each type
        types = list(results["type"].keys())
        if types:
            type_tabs = st.tabs(types)

            for i, type_name in enumerate(types):
                with type_tabs[i]:
                    type_data = results["type"][type_name]

                    # Create tabs for each runtime within this type
                    runtimes = list(type_data.keys())
                    if runtimes:
                        runtime_tabs = st.tabs(runtimes)

                        for j, runtime in enumerate(runtimes):
                            with runtime_tabs[j]:
                                runtime_data = type_data[runtime]

                                # Display clusters for this runtime
                                clusters = list(runtime_data.keys())
                                if clusters:
                                    st.markdown(f"**Found {len(clusters)} clusters for runtime {runtime}**")

                                    for cluster_name in clusters:
                                        cluster_entries = runtime_data[cluster_name]

                                        # Convert to DataFrame for better display
                                        df = pd.DataFrame(cluster_entries)

                                        # Display cluster information
                                        st.markdown(f"### Cluster: {cluster_name}")

                                        # Get class name if available
                                        if DataFrameKeys.cluster_class in df.columns:
                                            class_name = df[DataFrameKeys.cluster_class].iloc[0]
                                            st.markdown(f"**Class:** {class_name}")

                                        # Display the data
                                        st.dataframe(df[[c for c in COL_TO_SHOW if c in df.columns]])
                                else:
                                    st.info(f"No clusters found for runtime {runtime}")
                    else:
                        st.info("No runtime data available for this type")
        else:
            st.info("No type-based data available")

    # Display Model-based results
    if results.get("model"):
        st.subheader("Model-based Analysis")

        models = list(results["model"].keys())
        if models:
            # Create tabs for each model
            model_tabs = st.tabs(models)

            for i, model_name in enumerate(models):
                with model_tabs[i]:
                    model_data = results["model"][model_name]

                    # Convert to DataFrame for better display
                    df = pd.DataFrame(model_data)

                    # Group by cluster name for better organization
                    if DataFrameKeys.cluster_name in df.columns:
                        clusters = df[DataFrameKeys.cluster_name].unique()

                        st.markdown(f"**Found {len(clusters)} clusters for model {model_name}**")

                        for cluster in clusters:
                            cluster_df = df[df[DataFrameKeys.cluster_name] == cluster]

                            st.markdown(f"### Cluster: {cluster}")

                            # Get class name if available
                            if DataFrameKeys.cluster_class in cluster_df.columns:
                                class_name = cluster_df[DataFrameKeys.cluster_class].iloc[0]
                                st.markdown(f"**Class:** {class_name}")

                            # Display the data
                            st.dataframe(cluster_df[[c for c in COL_TO_SHOW if c in cluster_df.columns]])
                    else:
                        st.dataframe(df[[c for c in COL_TO_SHOW if c in df.columns]])
        else:
            st.info("No model-based data available")
