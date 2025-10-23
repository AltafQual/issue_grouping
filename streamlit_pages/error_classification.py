import asyncio
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from src import helpers
from src.constants import FaissConfigurations
# Import your cluster search functionality
from src.custom_clustering import CustomEmbeddingCluster

st.title("Error Classification")


def get_available_types():
    base_path = FaissConfigurations.base_path
    types = []

    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            if item.endswith("_custom") and os.path.isdir(os.path.join(base_path, item)):
                type_name = item.replace("_custom", "")
                types.append(type_name)

    return types


# Create sidebar for configuration
st.sidebar.header("Configuration")
similarity_threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.5,
    max_value=1.0,
    value=0.90,
    step=0.01,
    help="Minimum similarity threshold for matching clusters",
)

with st.expander("How to use this tool"):
    st.markdown(
        """
    1. Select the type of classification from the dropdown
    2. Enter one query in the text area per test and make sure to change type accordingly
    3. Adjust the similarity threshold in the sidebar if needed
    4. Click the "Classify" button to see results
    5. Results will show the matched cluster and class for each query along with metadata about that cluster
    """
    )

# Get available types
available_types = get_available_types()
if not available_types:
    st.error(f"No classification types found. Please check the {FaissConfigurations.base_path} directory.")
    st.stop()

selected_type = st.selectbox("Select Type", available_types)

query_input = st.text_area(
    "Enter Error Reason", height=150, help="Please enter only one error reason, per classification session"
)

# Process button
if st.button("Classify"):
    if not query_input.strip():
        st.error("Please enter at least one query to classify")
    else:
        query = query_input.strip()
        query = helpers.preprocess_error_log(query)
        query = helpers.mask_numbers(query)
        query = helpers.trim(query)

        with st.spinner("Classifying queries..."):
            try:
                # Initialize the cluster search
                cluster_search = CustomEmbeddingCluster()

                # Run the async function in a synchronous context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                cluster_names, class_names = loop.run_until_complete(
                    cluster_search.batch_search(
                        type_=selected_type, queries=query, similarity_threshold=similarity_threshold
                    )
                )
                loop.close()

                # Display results
                st.subheader("Classification Results")
                all_clusters = cluster_search.get_all_clusters(selected_type)

                # Display metadata for each matched cluster
                unique_clusters = set([c for c in cluster_names if c != "non_grouped"])

                if unique_clusters:
                    for cluster in unique_clusters:
                        if cluster in all_clusters:
                            st.markdown(f"### Cluster: {cluster}")
                            st.markdown(f"**Class:** {all_clusters[cluster]['class']}")

                            # Format the metadata as JSON for display
                            cluster_metadata = all_clusters[cluster]
                            st.json(cluster_metadata)
                else:
                    st.info("No matching clusters found.")

            except Exception as e:
                st.error(f"An error occurred during classification: {str(e)}")
                st.exception(e)
