import logging
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import HDBSCAN

from src import helpers
from src.constants import DataFrameKeys
from src.data_loader import ExcelLoader
from src.embeddings import QGenieBGEM3Embedding
from src.qgenie import qgenie_post_processing


class FailureAnalyzer:
    def __init__(self, embedding_model=None):
        """Initialize the failure analyzer with configurable parameters."""
        self.logger = self._setup_logger()
        self.logger.info("loading model")
        self.embedding_model = QGenieBGEM3Embedding()

    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the analyzer."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def load_data(self, file_path: str = None, st_obj=None) -> pd.DataFrame:
        """Load data from the specified Excel file."""
        self.logger.info(f"Loading data")
        return ExcelLoader.load(path=file_path, st_obj=st_obj)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for the provided texts."""
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embedding_model.embed(texts)
        return np.array(embeddings)

    def cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster the embeddings using HDBSCAN."""
        self.logger.info("Clustering embeddings")
        cluster = HDBSCAN(min_cluster_size=2, min_samples=10, metric="cosine", n_jobs=-1)
        return cluster.fit_predict(embeddings)

    def analyze(
        self, file_path: str = None, st_object=None, dataframe=None, failure_column: str = "reason"
    ) -> pd.DataFrame:
        """Perform the complete analysis workflow."""
        # Load data
        if file_path:
            dataframe = self.load_data(file_path)
        elif st_object:
            dataframe = self.load_data(st_object=st_object)

        self.logger.info(f"Loaded {len(dataframe)} rows")

        with st.spinner("Pre processing Error logs"):
            # Generate embeddings
            failure_texts = dataframe[failure_column].astype(str).tolist()
            failure_texts = [helpers.preprocess_error_log(text) for text in failure_texts]
            failure_df = helpers.remove_empty_and_misc_rows(
                dataframe, failure_texts, DataFrameKeys.preprocessed_text_key
            )

        with st.spinner("Generating embeddings"):
            start = time.time()
            embeddings = self.generate_embeddings(failure_df[DataFrameKeys.preprocessed_text_key].tolist())
            failure_df[DataFrameKeys.embeddings_key] = list(embeddings)
            st.info(f"Embeddings generated in {round(time.time() - start,2)} seconds")

        # Cluster embeddings
        failure_df[DataFrameKeys.cluster_type_int] = self.cluster_embeddings(embeddings)
        # save copy of cluster into new column later will be replaced with the actual names from Qgeneie
        failure_df[DataFrameKeys.cluster_name] = failure_df[DataFrameKeys.cluster_type_int].copy()
        failure_df[DataFrameKeys.cluster_name] = failure_df[DataFrameKeys.cluster_name].astype("object")

        merged_groups = helpers.merge_similar_clusters(
            embeddings, list(failure_df[DataFrameKeys.cluster_type_int].unique())
        )
        failure_df = helpers.update_labels_with_merged_clusters(
            failure_df, merged_groups, DataFrameKeys.cluster_type_int
        )

        # Log cluster statistics
        cluster_counts = failure_df[DataFrameKeys.cluster_type_int].value_counts()
        self.logger.info(f"Found {len(cluster_counts) - (1 if -1 in cluster_counts else 0)} clusters")
        self.logger.info(f"Noise points: {cluster_counts.get(-1, 0)}")

        with st.spinner("QGenie Post Processing Cluster"):
            start = time.time()
            failure_df = qgenie_post_processing(failure_df)
            st.info(f"QGenie post processing completed in {round(time.time() - start,2)} seconds")
        return failure_df

    def save_results(self, data: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """Save the analysis results to a file."""
        if output_path is None:
            output_path = "failure_analysis_results.xlsx"

        # Create a copy without the embeddings column for saving
        save_data = data.drop(columns=["embeddings"])
        save_data.to_excel(output_path, index=False)
        self.logger.info(f"Results saved to {output_path}")
