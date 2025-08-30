import asyncio
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from src import helpers
from src.constants import ClusterSpecificKeys, DataFrameKeys
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

    def load_data(self, file_path: str = None, st_obj=None, tc_id=None) -> pd.DataFrame:
        """Load data from the specified Excel file."""
        self.logger.info(f"Loading data")
        if not tc_id:
            return ExcelLoader.load(path=file_path, st_obj=st_obj)

        return helpers.get_tc_id_df(tc_id)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for the provided texts."""
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embedding_model.embed(texts)
        return list(np.array(embeddings))

    async def agenerate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for the provided texts."""
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = await self.embedding_model.aembed(texts)
        return list(np.array(embeddings))

    async def agenerate_single_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for the provided texts."""
        if not isinstance(texts, list):
            texts = [texts]
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = [self.embedding_model.aembed_query(text) for text in texts]
        embeddings = await asyncio.gather(*embeddings)
        return list(np.array(embeddings))

    def cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster the embeddings using HDBSCAN."""
        self.logger.info("Clustering embeddings")
        cluster = HDBSCAN(min_cluster_size=2, min_samples=10, metric="cosine", n_jobs=-1)
        return cluster.fit_predict(embeddings)

    async def analyze(
        self, file_path: str = None, st_object=None, dataframe=None, failure_column: str = "reason"
    ) -> pd.DataFrame:
        """Perform the complete analysis workflow."""

        if file_path:
            dataframe = self.load_data(file_path)
        elif st_object:
            dataframe = self.load_data(st_object=st_object)

        self.logger.info(f"Loaded {len(dataframe)} rows")

        # Initialize all rows as non-grouped
        dataframe.loc[:, DataFrameKeys.cluster_name] = ClusterSpecificKeys.non_grouped_key
        dataframe.loc[:, DataFrameKeys.cluster_type_int] = ClusterSpecificKeys.non_grouped_key

        # Preprocess failure texts
        failure_texts = dataframe[failure_column].astype(str).tolist()
        failure_texts = [helpers.preprocess_error_log(text) for text in failure_texts]
        # Apply preprocessing steps
        failure_df = helpers.remove_empty_and_misc_rows(dataframe, failure_texts, DataFrameKeys.preprocessed_text_key)
        failure_df = helpers.trim_error_logs(failure_df)
        non_clustered_df = failure_df[
            failure_df[DataFrameKeys.cluster_name] == ClusterSpecificKeys.non_grouped_key
        ].reset_index(drop=True)
        empty_log_df = failure_df[
            failure_df[DataFrameKeys.cluster_name] != ClusterSpecificKeys.non_grouped_key
        ].reset_index(drop=True)
        non_clustered_df = helpers.fuzzy_cluster_grouping(non_clustered_df)
        failure_df = pd.concat([empty_log_df, non_clustered_df], axis=0).reset_index(drop=True)

        # Handle small datasets (10 or fewer rows)
        if failure_df.shape[0] <= 10:
            non_clustered_df = None
            if any(
                value == ClusterSpecificKeys.non_grouped_key
                for value in failure_df[DataFrameKeys.cluster_name].tolist()
            ):
                non_clustered_df = failure_df[
                    failure_df[DataFrameKeys.cluster_name] == ClusterSpecificKeys.non_grouped_key
                ]
                embeddings = await self.agenerate_single_embeddings(
                    non_clustered_df[DataFrameKeys.preprocessed_text_key].tolist()
                )
                non_clustered_df.loc[:, DataFrameKeys.embeddings_key] = embeddings
                non_clustered_df = qgenie_post_processing(non_clustered_df)

            if non_clustered_df is not None:
                failure_df = pd.concat([empty_log_df, non_clustered_df], axis=0).reset_index(drop=True)

            return failure_df

        # Split data into already clustered and non-clustered
        already_clustered_df = failure_df[
            failure_df[DataFrameKeys.cluster_name] != ClusterSpecificKeys.non_grouped_key
        ].reset_index(drop=True)
        non_clustered_df = failure_df[
            failure_df[DataFrameKeys.cluster_name] == ClusterSpecificKeys.non_grouped_key
        ].reset_index(drop=True)

        self.logger.info(
            f"{already_clustered_df.shape[0]} already clustered logs and {non_clustered_df.shape[0]} non-clustered logs"
        )

        # Process non-clustered data
        if non_clustered_df.shape[0] > 10:
            # For larger datasets, use clustering
            embeddings = await self.agenerate_embeddings(non_clustered_df[DataFrameKeys.preprocessed_text_key].tolist())
            non_clustered_df.loc[:, DataFrameKeys.embeddings_key] = embeddings

            # Cluster embeddings
            non_clustered_df.loc[:, DataFrameKeys.cluster_type_int] = self.cluster_embeddings(embeddings)

            # Merge similar clusters
            merged_groups = helpers.merge_similar_clusters(
                embeddings, list(non_clustered_df[DataFrameKeys.cluster_type_int].unique())
            )
            non_clustered_df = helpers.update_labels_with_merged_clusters(
                non_clustered_df, merged_groups, DataFrameKeys.cluster_type_int
            )

            # Log cluster statistics
            cluster_counts = non_clustered_df[DataFrameKeys.cluster_type_int].value_counts()
            num_clusters = len(cluster_counts) - (1 if -1 in cluster_counts else 0)
            self.logger.info(f"Found {num_clusters} clusters")
            self.logger.info(f"Noise points: {cluster_counts.get(-1, 0)}")
        else:
            # For smaller datasets, use single embeddings
            embeddings = await self.agenerate_single_embeddings(
                non_clustered_df[DataFrameKeys.preprocessed_text_key].tolist()
            )
            non_clustered_df.loc[:, DataFrameKeys.embeddings_key] = embeddings

        # Apply Qgenie post-processing
        non_clustered_df = qgenie_post_processing(non_clustered_df)
        # Combine results and reset index
        final_df = pd.concat([already_clustered_df, non_clustered_df], axis=0).reset_index(drop=True)
        final_df.loc[
            final_df[DataFrameKeys.cluster_name].isin(
                {ClusterSpecificKeys.non_grouped_key, str(ClusterSpecificKeys.non_grouped_key)}
            ),
            DataFrameKeys.cluster_name,
        ] = "Others"

        return final_df

    def save_results(self, data: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """Save the analysis results to a file."""
        if output_path is None:
            output_path = "failure_analysis_results.xlsx"

        # Create a copy without the embeddings column for saving
        save_data = data.drop(columns=["embeddings"])
        save_data.to_excel(output_path, index=False)
        self.logger.info(f"Results saved to {output_path}")
