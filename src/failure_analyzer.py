import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from src.clustering.ranker import (
    ClusterCohesionAnalyzer,
    ClusterRanker,
    merge_similar_clusters,
    reassign_unclustered_logs,
    update_labels_with_merged_clusters
)
from src.clustering.splade_encoder import SPLADEEncoder
from src.constants import ClusterSpecificKeys, DataFrameKeys, ErrorLogConfigurations
from src.data.excel_loader import ExcelLoader
from src.data.mysql_client import get_tc_id_df
from src.embeddings import FallbackEmbeddings
from src.llm.cluster_classifier import assign_cluster_class
from src.llm.cluster_namer import generate_cluster_name_for_single_rows
from src.llm.deduplicator import (
    detect_and_merge_near_duplicate_clusters,
    qgenie_post_processing,
    subcluster_verifier_failed
)
from src.logger import AppLogger
from src.pipeline.pregroup_pipeline import check_if_issue_alread_grouped, fuzzy_cluster_grouping, splade_pregroup
from src.preprocessing.log_extractor import remove_empty_and_misc_rows
from src.preprocessing.normalizer import preprocess_error_log, trim_error_logs


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
            dataframe = get_tc_id_df(tc_id)

        if isinstance(dataframe, pd.DataFrame):
            if dataframe.empty:
                return dataframe
            self.logger.info(f"Total number of test cases: {dataframe.shape[0]}")
            return dataframe[~dataframe["result"].isin({"PASS", "NOT_RUN", "PARENT_NOT_RUN", "PARENT_FAIL"})]

    def cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster the embeddings using HDBSCAN."""
        self.logger.info("Clustering embeddings")
        cluster = HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric="cosine",
            cluster_selection_method="eom",
            cluster_selection_epsilon=0.12,
            n_jobs=-1,
        )
        return cluster.fit_predict(embeddings)

    async def analyze(
        self, file_path: str = None, st_object=None, dataframe=None, failure_column: str = "reason"
    ) -> pd.DataFrame:
        """Perform the complete analysis workflow.

        Computes embeddings and SPLADE vectors ONCE at the top, then passes
        pre-computed arrays to all downstream steps.  This eliminates the
        3-5x redundant embedding and 2x redundant SPLADE calls that previously
        made the pipeline take over an hour.
        """
        if file_path:
            dataframe = self.load_data(file_path)
        elif st_object:
            dataframe = self.load_data(st_object=st_object)

        self.logger.info(f"Loaded {len(dataframe)} rows")
        current_type = dataframe.iloc[0]["type"]
        # Initialize all rows as non-grouped
        dataframe[DataFrameKeys.cluster_name] = pd.array(
            [ClusterSpecificKeys.non_grouped_key] * len(dataframe), dtype=object
        )
        dataframe.loc[:, DataFrameKeys.cluster_type_int] = ClusterSpecificKeys.non_grouped_key
        dataframe[DataFrameKeys.grouped_from_faiss] = pd.array([np.nan] * len(dataframe), dtype=object)
        dataframe[DataFrameKeys.embeddings_key] = pd.array([np.nan] * len(dataframe), dtype=object)

        # Preprocess failure texts
        failure_texts = dataframe[failure_column].astype(str).tolist()
        failure_texts = [preprocess_error_log(text) for text in failure_texts]

        # Apply preprocessing steps
        failure_df = remove_empty_and_misc_rows(dataframe, failure_texts, DataFrameKeys.preprocessed_text_key)
        failure_df = trim_error_logs(failure_df)

        # ====================================================================
        # COMPUTE EMBEDDINGS + SPLADE ONCE — pass everywhere below
        # ====================================================================
        all_texts = failure_df[DataFrameKeys.preprocessed_text_key].astype(str).tolist()
        failure_df["_embed_pos"] = range(len(failure_df))

        # 1. Embed ALL texts ONCE
        self.logger.info(f"[OneShot] Computing embeddings for {len(all_texts)} texts")
        all_embeddings = np.array(await self.embedding_model.aembed(all_texts))
        failure_df[DataFrameKeys.embeddings_key] = pd.Series(all_embeddings.tolist(), index=failure_df.index)

        # 2. SPLADE encode ALL texts ONCE
        encoder = SPLADEEncoder()
        all_splade_vecs = None
        if encoder.is_available:
            self.logger.info(f"[OneShot] Computing SPLADE vectors for {len(all_texts)} texts")
            all_splade_vecs = await encoder.aencode(all_texts)

        self.logger.info(f"[OneShot] Embeddings and SPLADE computed for {len(all_texts)} texts")

        # ====================================================================
        # PIPELINE STEPS — use pre-computed vectors, no re-computation
        # ====================================================================

        # Separate empty/no-log rows (already assigned a cluster name by preprocessing)
        empty_log_df = failure_df[failure_df[DataFrameKeys.cluster_name] != ClusterSpecificKeys.non_grouped_key]
        failure_df = failure_df[~failure_df.index.isin(empty_log_df.index)]

        # Step 1: Check if errors already exist in vector DB
        failure_df = await check_if_issue_alread_grouped(
            failure_df,
            precomputed_embeddings=all_embeddings,
            precomputed_splade_vecs=all_splade_vecs,
        )
        faiss_grouped = failure_df[
            (failure_df[DataFrameKeys.grouped_from_faiss] == True)
            & (failure_df[DataFrameKeys.cluster_name] != ClusterSpecificKeys.non_grouped_key)
        ]

        # Step 2: Fuzzy pre-grouping for remaining unclustered
        non_clustered_df = failure_df[
            failure_df[DataFrameKeys.cluster_name] == ClusterSpecificKeys.non_grouped_key
        ].reset_index(drop=True)
        fuzzy_clustered_df = pd.DataFrame()
        if not non_clustered_df.empty:
            non_clustered_df = await fuzzy_cluster_grouping(non_clustered_df, precomputed_embeddings=all_embeddings)
            fuzzy_clustered_df = non_clustered_df[
                (non_clustered_df[DataFrameKeys.cluster_name] != ClusterSpecificKeys.non_grouped_key)
                & (
                    ~non_clustered_df[DataFrameKeys.cluster_name].isin(
                        {ErrorLogConfigurations.empty_error, ErrorLogConfigurations.no_error}
                    )
                    & (non_clustered_df[DataFrameKeys.grouped_from_faiss] != True)
                )
            ]
            # Embeddings already in DataFrame from one-shot computation — no generate_embeddings() needed
            if not fuzzy_clustered_df.empty:
                non_clustered_df = non_clustered_df[
                    (~non_clustered_df.index.isin(fuzzy_clustered_df.index))
                    & (non_clustered_df[DataFrameKeys.cluster_name] == ClusterSpecificKeys.non_grouped_key)
                ]

        # Step 3: SPLADE pre-grouping with pre-computed vectors
        splade_grouped_df = pd.DataFrame()
        if not non_clustered_df.empty:
            # Get subset of pre-computed vectors for unclustered rows
            if "_embed_pos" in non_clustered_df.columns:
                positions = non_clustered_df["_embed_pos"].tolist()
                subset_splade = all_splade_vecs[positions] if all_splade_vecs is not None else None
                subset_embeddings = all_embeddings[positions]
            else:
                subset_splade = None
                subset_embeddings = None

            non_clustered_df = await splade_pregroup(
                non_clustered_df,
                type_=current_type,
                precomputed_splade_vecs=subset_splade,
                precomputed_embeddings=subset_embeddings,
            )

            # Split SPLADE-grouped rows from still-unclustered
            splade_grouped_mask = non_clustered_df[DataFrameKeys.cluster_name] != ClusterSpecificKeys.non_grouped_key
            splade_grouped_df = non_clustered_df[splade_grouped_mask].copy()
            non_clustered_df = non_clustered_df[~splade_grouped_mask].reset_index(drop=True)

            self.logger.info(
                f"[SPLADESplit] type={current_type}: "
                f"{len(splade_grouped_df)} rows pre-grouped by SPLADE, "
                f"{len(non_clustered_df)} rows remain for HDBSCAN"
            )
            # Embeddings already in DataFrame — no generate_embeddings() needed

        self.logger.info(
            f"\nType: {current_type} \ntotal errors: {len(all_texts)}, "
            f"\nEmpty logs grouped: {empty_log_df.shape[0]}, "
            f"\nFuzzy grouped: {fuzzy_clustered_df.shape[0]}, "
            f"\nFaiss Grouped: {faiss_grouped.shape[0]}, "
            f"\nNot grouped: {non_clustered_df.shape[0]} "
            f"\nSplade Grouped: {splade_grouped_df.shape[0]}"
        )
        failure_df = pd.concat(
            [empty_log_df, fuzzy_clustered_df, splade_grouped_df, non_clustered_df, faiss_grouped], axis=0
        )

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
                    # Embeddings already in DataFrame
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

            verifier_failed_df = await subcluster_verifier_failed(
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
                cluster_names = await generate_cluster_name_for_single_rows(failure_df.loc[mask])
                failure_df.loc[mask, DataFrameKeys.cluster_name] = cluster_names

            failure_df = await assign_cluster_class(failure_df)
            failure_df = await detect_and_merge_near_duplicate_clusters(failure_df)
            failure_df = ClusterRanker().rank_dataframe(failure_df)
            failure_df = ClusterCohesionAnalyzer().analyze_dataframe(failure_df)
            failure_df.drop(columns=["_embed_pos"], inplace=True, errors="ignore")
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
            # Extract embeddings from DataFrame (already computed in one-shot)
            embeddings = non_clustered_df[DataFrameKeys.embeddings_key].tolist()
            # Filter out any NaN values and convert to proper array
            valid_embeddings = [e for e in embeddings if isinstance(e, (list, np.ndarray))]
            if valid_embeddings:
                embeddings = valid_embeddings

            # Process non-clustered data
            if not non_clustered_df.empty and non_clustered_df.shape[0] > 10:

                # Cluster embeddings
                non_clustered_df.loc[:, DataFrameKeys.cluster_type_int] = self.cluster_embeddings(embeddings)

                # Merge similar clusters
                merged_groups = merge_similar_clusters(
                    embeddings, list(non_clustered_df[DataFrameKeys.cluster_type_int].unique())
                )
                non_clustered_df = update_labels_with_merged_clusters(
                    non_clustered_df, merged_groups, DataFrameKeys.cluster_type_int
                )

                # Reassign noise points to nearest cluster before LLM post-processing
                non_clustered_df = reassign_unclustered_logs(non_clustered_df, threshold=0.82)

                # Log cluster statistics
                cluster_counts = non_clustered_df[DataFrameKeys.cluster_type_int].value_counts()
                num_clusters = len(cluster_counts) - (1 if -1 in cluster_counts else 0)
                self.logger.info(f"Found {num_clusters} clusters")
                self.logger.info(f"Noise points: {cluster_counts.get(-1, 0)}")

            # Apply Qgenie post-processing to remaining unclustered rows
            if not non_clustered_df.empty:
                non_clustered_df = await qgenie_post_processing(non_clustered_df)

        dfs_to_concat = []
        if not already_clustered_df.empty:
            verifier_failed_df = await subcluster_verifier_failed(
                already_clustered_df[
                    (already_clustered_df[DataFrameKeys.cluster_name] == "VerifierFailed")
                    & (already_clustered_df[DataFrameKeys.grouped_from_faiss] != True)
                ]
            )
            # Combine results and reset index
            dfs_to_concat = [
                non_clustered_df,
                splade_grouped_df,
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
            cluster_names = await generate_cluster_name_for_single_rows(final_df.loc[mask])
            # Update the dataframe with results
            final_df.loc[mask, DataFrameKeys.cluster_name] = cluster_names

        final_df = await assign_cluster_class(final_df)
        final_df = await detect_and_merge_near_duplicate_clusters(final_df)
        final_df = ClusterRanker().rank_dataframe(final_df)
        final_df = ClusterCohesionAnalyzer().analyze_dataframe(final_df)
        final_df.drop(columns=["_embed_pos"], inplace=True, errors="ignore")
        self.logger.info(f"Finished processing: {current_type}")
        return final_df
