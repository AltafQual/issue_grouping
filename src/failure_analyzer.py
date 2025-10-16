import asyncio
import threading
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import swifter
from sklearn.cluster import HDBSCAN

from src import helpers
from src.constants import ClusterSpecificKeys, DataFrameKeys, ErrorLogConfigurations
from src.custom_clustering import CustomEmbeddingCluster
from src.data_loader import ExcelLoader
from src.embeddings import BGEM3Embeddings, FallbackEmbeddings, QGenieBGEM3Embedding
from src.faiss_db import FaissIVFFlatIndex
from src.logger import AppLogger
from src.qgenie import generate_cluster_name, qgenie_post_processing, subcluster_verifier_failed

threading.Thread(target=helpers.faissdb_update_worker, daemon=True).start()


class FailureAnalyzer:
    def __init__(self, embedding_model=None):
        """Initialize the failure analyzer with configurable parameters."""
        self.logger = AppLogger().get_logger(__name__)
        self.logger.info("loading model")
        self.embedding_model = FallbackEmbeddings()

    def load_data(self, file_path: str = None, st_obj=None, tc_id=None) -> pd.DataFrame:
        """Load data from the specified Excel file."""
        self.logger.info(f"Loading data")
        if not tc_id:
            dataframe = ExcelLoader.load(path=file_path, st_obj=st_obj)
        else:
            dataframe = helpers.get_tc_id_df(tc_id)

        if isinstance(dataframe, pd.DataFrame):
            st.write(f"Total number of test cases: {dataframe.shape[0]}")
            return dataframe[~dataframe["result"].isin({"PASS", "NOT_RUN", "PARENT_NOT_RUN", "PARENT_FAIL"})]

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for the provided texts."""
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embedding_model.embed(texts)
        if isinstance(embeddings, np.ndarray):
            return list(embeddings)
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
        # TODO: convert everything ot self variables and use OOP properly, try extracting run id from dataframe if not provided
        if file_path:
            dataframe = self.load_data(file_path)
        elif st_object:
            dataframe = self.load_data(st_object=st_object)

        self.logger.info(f"Loaded {len(dataframe)} rows")
        current_type = dataframe.iloc[0]["type"]

        # Initialize all rows as non-grouped
        dataframe.loc[:, DataFrameKeys.cluster_name] = ClusterSpecificKeys.non_grouped_key
        dataframe.loc[:, DataFrameKeys.cluster_type_int] = ClusterSpecificKeys.non_grouped_key
        dataframe.loc[:, DataFrameKeys.grouped_from_faiss] = np.nan
        dataframe.loc[:, DataFrameKeys.embeddings_key] = np.nan

        # Preprocess failure texts
        failure_texts = dataframe[failure_column].astype(str).tolist()
        failure_texts = [helpers.preprocess_error_log(text) for text in failure_texts]

        # Apply preprocessing steps
        failure_df = helpers.remove_empty_and_misc_rows(dataframe, failure_texts, DataFrameKeys.preprocessed_text_key)
        failure_df = helpers.trim_error_logs(failure_df)
        failure_df_copy = failure_df.copy()

        # dataframe with empty/no logs
        empty_log_df = failure_df[failure_df[DataFrameKeys.cluster_name] != ClusterSpecificKeys.non_grouped_key]

        failure_df = failure_df[~failure_df.index.isin(empty_log_df.index)]
        failure_df = await helpers.check_if_issue_alread_grouped(failure_df)
        faiss_grouped = failure_df[
            (failure_df[DataFrameKeys.grouped_from_faiss] == True)
            & (failure_df[DataFrameKeys.cluster_name] != ClusterSpecificKeys.non_grouped_key)
        ]

        non_clustered_df = failure_df[
            failure_df[DataFrameKeys.cluster_name] == ClusterSpecificKeys.non_grouped_key
        ].reset_index(drop=True)
        fuzzy_clustered_df = pd.DataFrame()
        if not non_clustered_df.empty:
            non_clustered_df = await helpers.fuzzy_cluster_grouping(non_clustered_df)
            fuzzy_clustered_df = non_clustered_df[
                (non_clustered_df[DataFrameKeys.cluster_name] != ClusterSpecificKeys.non_grouped_key)
                & (
                    ~non_clustered_df[DataFrameKeys.cluster_name].isin(
                        {ErrorLogConfigurations.empty_error, ErrorLogConfigurations.no_error}
                    )
                    & (non_clustered_df[DataFrameKeys.grouped_from_faiss] != True)
                )
            ]

            if not fuzzy_clustered_df.empty:
                fuzzy_clustered_df[DataFrameKeys.embeddings_key] = pd.Series(
                    self.generate_embeddings(fuzzy_clustered_df[DataFrameKeys.preprocessed_text_key].tolist()),
                    index=fuzzy_clustered_df.index,
                )

                non_clustered_df = non_clustered_df[
                    (~non_clustered_df.index.isin(fuzzy_clustered_df.index))
                    & (non_clustered_df[DataFrameKeys.cluster_name] == ClusterSpecificKeys.non_grouped_key)
                ]

        self.logger.info(
            f"\nType: {current_type} \ntotal errors: {failure_df_copy.shape[0]}, \nEmpty logs grouped: {empty_log_df.shape[0]}, \nFuzzy grouped: {fuzzy_clustered_df.shape[0]}, \nFaiss Grouped: {faiss_grouped.shape[0]}, \nNot grouped: {non_clustered_df.shape[0]}"
        )
        failure_df = pd.concat([empty_log_df, fuzzy_clustered_df, non_clustered_df, faiss_grouped], axis=0)

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
                if not non_clustered_df.empty:
                    embeddings = self.generate_embeddings(
                        non_clustered_df[DataFrameKeys.preprocessed_text_key].tolist()
                    )
                    non_clustered_df.loc[:, DataFrameKeys.embeddings_key] = pd.Series(
                        embeddings, index=non_clustered_df.index
                    )
                    non_clustered_df = await qgenie_post_processing(non_clustered_df)
                else:
                    non_clustered_df = None

            if non_clustered_df is not None:
                failure_df = pd.concat(
                    [
                        failure_df[failure_df[DataFrameKeys.cluster_name] != ClusterSpecificKeys.non_grouped_key],
                        non_clustered_df,
                    ],
                    axis=0,
                )

            verifier_failed_df = subcluster_verifier_failed(
                failure_df[
                    (failure_df[DataFrameKeys.cluster_name] == "VerifierFailed")
                    & (failure_df[DataFrameKeys.grouped_from_faiss] != True)
                ]
            )
            if verifier_failed_df is not None and not verifier_failed_df.empty:
                failure_df = pd.concat(
                    [failure_df[failure_df[DataFrameKeys.cluster_name] != "VerifierFailed"], verifier_failed_df], axis=0
                ).reset_index(drop=True)

            mask = failure_df[DataFrameKeys.cluster_name].isin(
                {ClusterSpecificKeys.non_grouped_key, str(ClusterSpecificKeys.non_grouped_key)}
            )
            if not failure_df.loc[mask].empty:
                cluster_names = await helpers.generate_cluster_name_for_single_rows(failure_df.loc[mask])
                failure_df.loc[mask, DataFrameKeys.cluster_name] = cluster_names

            failure_df = await helpers.assign_cluster_class(failure_df)
            self.logger.info(f"Finished processing: {current_type}")
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

        if not non_clustered_df.empty:
            embeddings = self.generate_embeddings(non_clustered_df[DataFrameKeys.preprocessed_text_key].tolist())
            non_clustered_df.loc[:, DataFrameKeys.embeddings_key] = pd.Series(embeddings, index=non_clustered_df.index)

            # Process non-clustered data
            if non_clustered_df.shape[0] > 10:

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

            # Apply Qgenie post-processing
            non_clustered_df = await qgenie_post_processing(non_clustered_df)

        dfs_to_concat = []
        if not already_clustered_df.empty:
            verifier_failed_df = subcluster_verifier_failed(
                already_clustered_df[
                    (already_clustered_df[DataFrameKeys.cluster_name] == "VerifierFailed")
                    & (already_clustered_df[DataFrameKeys.grouped_from_faiss] != True)
                ]
            )
            # Combine results and reset index
            dfs_to_concat = [
                non_clustered_df,
                already_clustered_df[already_clustered_df[DataFrameKeys.cluster_name] != "VerifierFailed"],
            ]
            if verifier_failed_df is not None and not verifier_failed_df.empty:
                dfs_to_concat.append(verifier_failed_df)

        if dfs_to_concat:
            final_df = pd.concat(dfs_to_concat, axis=0).reset_index(drop=True)
        else:
            final_df = already_clustered_df.copy()

        mask = final_df[DataFrameKeys.cluster_name].isin(
            {ClusterSpecificKeys.non_grouped_key, str(ClusterSpecificKeys.non_grouped_key)}
        )

        if not final_df.loc[mask].empty:
            # Process rows with semaphore
            cluster_names = await helpers.generate_cluster_name_for_single_rows(final_df.loc[mask])
            # Update the dataframe with results
            final_df.loc[mask, DataFrameKeys.cluster_name] = cluster_names

        final_df = await helpers.assign_cluster_class(final_df)
        self.logger.info(f"Finished processing: {current_type}")
        return final_df

    def save_results(self, data: pd.DataFrame, output_path: Optional[str] = None) -> None:
        """Save the analysis results to a file."""
        if output_path is None:
            output_path = "failure_analysis_results.xlsx"

        # Create a copy without the embeddings column for saving
        save_data = data.drop(columns=["embeddings"])
        save_data.to_excel(output_path, index=False)
        self.logger.info(f"Results saved to {output_path}")

    def save_as_faiss(self, db: "FaissIVFFlatIndex", data: pd.DataFrame, run_id=None):
        CustomEmbeddingCluster().save_threaded(data, run_id=run_id)
