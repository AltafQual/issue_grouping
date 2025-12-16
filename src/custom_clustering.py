import concurrent.futures
import json
import os
import time
import traceback
from threading import Lock
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.constants import ClusterSpecificKeys, DataFrameKeys, FaissConfigurations
from src.embeddings import FallbackEmbeddings
from src.logger import AppLogger

logger = AppLogger().get_logger()


class CustomEmbeddingCluster:
    """
    Custom clustering class that stores embeddings and metadata together,
    allowing for more control over search results and metadata handling.

    Designed to maintain compatibility with existing metadata structure
    while providing improved search capabilities.
    """

    def __init__(self, base_path: str = FaissConfigurations.base_path):
        self.base_path = base_path

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length for cosine similarity."""
        vectors = np.array(vectors)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        return vectors / norms

    def _compute_centroids(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Compute centroids for each cluster in the dataframe.
        Returns a dictionary mapping cluster names to their centroids and metadata.
        """
        centroids = {}

        for cluster_name, group in df.groupby(DataFrameKeys.cluster_name):

            # Compute centroid
            embeddings = np.vstack(group[DataFrameKeys.embeddings_key])
            centroid = np.mean(embeddings, axis=0)

            centroids[cluster_name] = {
                "centroid": centroid,
                "class": group[DataFrameKeys.cluster_class].iloc[0],
                "tc_uuids": group["tc_uuid"].tolist(),
            }

        return centroids

    def _check_existing_faiss_for_type(self, type):
        type_based_path = os.path.join(FaissConfigurations.base_path, f"{type}_custom")
        if os.path.exists(type_based_path):
            if os.path.isfile(os.path.join(type_based_path, "centroids.npy")):
                return True

        print(f"Existing FAISS index not found for type: {type}. Creating a new one.")
        return False

    def _get_run_ids_metadata_dict(self, data, df):
        from src.helpers import get_bu_name

        columns = ["runtime", "jira_id", "soc_name", "log", "model_name"]
        tc_ids_dict = {}
        for tc_uuid in data["tc_uuids"]:
            tc_uuid_df = df[df["tc_uuid"].isin([tc_uuid])]
            tc_uuid_df = tc_uuid_df[[col for col in columns if col in tc_uuid_df.columns]]
            records = tc_uuid_df.to_dict(orient="records")

            logger.info(f"Adding records for tc_id {tc_uuid}: {records}")
            # Handle the case where records might be empty
            if records:
                record = records[0]  # Take the first record
                # Add BU directly to the record
                if "soc_name" in record:
                    record["BU"] = get_bu_name(record["soc_name"])
                tc_ids_dict[tc_uuid] = record
            else:
                # Handle empty records case
                tc_ids_dict[tc_uuid] = []

        return tc_ids_dict

    def save_threaded(self, dataframe: pd.DataFrame, type_: str = None, run_id: Optional[str] = None) -> None:
        """
        Save embeddings and metadata for the given type.
        Args:
            dataframe: DataFrame containing embeddings and metadata
            type_: Type of data (e.g., 'benchmark', 'test')
            run_id: Optional run ID to track
        """

        def _process_type(type_group):
            type_, typed_dataframe = type_group
            try:
                # Create directory for this type
                type_dir = os.path.join(self.base_path, f"{type_}_custom")
                os.makedirs(type_dir, exist_ok=True)

                # Use lock for database operations
                with lock:
                    update_error_map_qgenie_table(typed_dataframe)

                if self._check_existing_faiss_for_type(type_):
                    self.update(typed_dataframe, type_, run_id=run_id)
                    return f"Updated existing clusters for type: {type_}"

                # Compute centroids
                centroids = self._compute_centroids(typed_dataframe)

                # Create metadata in the format you're currently using
                metadata = {}
                for cluster_name, data in centroids.items():
                    metadata[cluster_name] = {"class": data["class"], "run_ids": {}}

                    if run_id:
                        metadata[cluster_name]["run_ids"][run_id] = self._get_run_ids_metadata_dict(
                            data, typed_dataframe
                        )

                assert len(centroids) == len(metadata), (
                    f"CRITICAL ERROR: Number of centroids ({len(centroids)}) "
                    f"does not match number of metadata entries ({len(metadata)})"
                )

                # Save centroids
                with open(os.path.join(type_dir, "centroids.npy"), "wb") as f:
                    np.save(
                        f,
                        np.array([centroids[name]["centroid"] for name in centroids.keys()]),
                    )

                # Save metadata
                with open(os.path.join(type_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=3)

                return f"Saved {len(centroids)} clusters for type: {type_}"
            except Exception as e:
                return f"Error processing type {type_}: {str(e)}\n{traceback.format_exc()}"

        def _update_metadata(already_grouped):
            type_, grouped_df = already_grouped
            print(f"Updating metadata for type: {type_}")
            type_dir = os.path.join(self.base_path, f"{type_}_custom")

            try:
                with open(os.path.join(type_dir, "metadata.json"), "r") as f:
                    existing_metadata = json.load(f)

                for cluster_name, group in grouped_df.groupby(DataFrameKeys.cluster_name):
                    if cluster_name not in existing_metadata:
                        print(
                            f"Cluster name: {cluster_name} didn't exists in metadata, but marked as already grouped for run_id {run_id} !!!!"
                        )
                        continue

                    print(f"updating metadata for {type_} in cluster name: {cluster_name}")
                    new_cluster_meta = self._get_run_ids_metadata_dict({"tc_uuids": group["tc_uuid"].tolist()}, group)

                    if run_id not in existing_metadata[cluster_name]["run_ids"]:
                        existing_metadata[cluster_name]["run_ids"].update({run_id: new_cluster_meta})
                    else:
                        print(f"Run id: {run_id} already processed exists in metadata, avoiding updating metadata")

                with open(os.path.join(type_dir, "metadata.json"), "w") as f:
                    json.dump(existing_metadata, f, indent=3)

            except Exception as e:
                return f"Failure while updating metadata for id: {run_id} : {traceback.format_exc()}"

            return f"Sucessfully updated metadata for run id: {run_id}'s type: {type_}"

        from src.helpers import update_error_map_qgenie_table

        if run_id is None:
            if "testplan_id" in dataframe.columns:
                run_id = dataframe.testplan_id.tolist()[0]

        if run_id:
            self._update_processed_runids(self.base_path, run_id)

        # Filter out rows without embeddings and non-grouped entries
        filtered_df = dataframe[
            (dataframe[DataFrameKeys.embeddings_key].notna())
            & (dataframe[DataFrameKeys.cluster_name] != ClusterSpecificKeys.non_grouped_key)
            & (pd.isna(dataframe[DataFrameKeys.grouped_from_faiss]))
        ].copy()

        already_existing_issues = dataframe[
            ~(pd.isna(dataframe[DataFrameKeys.grouped_from_faiss])) & (dataframe[DataFrameKeys.embeddings_key].isna())
        ].copy()

        lock = Lock()
        type_groups = list(filtered_df.groupby("type"))
        print(f"Processing {len(type_groups)} types in parallel")

        if not filtered_df.empty:
            max_workers = min(10, len(type_groups))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_type = {
                    executor.submit(_process_type, type_group): type_group[0] for type_group in type_groups
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_type):
                    type_ = future_to_type[future]
                    try:
                        result = future.result()
                        print(result)
                    except Exception as e:
                        print(f"Exception processing type {type_}: {str(e)}")

        if not already_existing_issues.empty:
            print(f"found DataFrame with already existing issues: {run_id}: Updating metadata")
            already_grouped_dfs = list(already_existing_issues.groupby("type"))
            max_workers = min(10, len(already_grouped_dfs))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_type = {
                    executor.submit(_update_metadata, already_grouped_df): already_grouped_df[0]
                    for already_grouped_df in already_grouped_dfs
                }

                for future in concurrent.futures.as_completed(future_to_type):
                    type_ = future_to_type[future]
                    try:
                        result = future.result()
                        print(result)
                    except Exception as e:
                        print(f"Exception saving metadat for {type_}: {str(e)}")

    def save(self, dataframe: pd.DataFrame, type_: str = None, run_id: Optional[str] = None) -> None:
        """
        Save embeddings and metadata for the given type.

        Args:
            dataframe: DataFrame containing embeddings and metadata
            type_: Type of data (e.g., 'benchmark', 'test')
            run_id: Optional run ID to track
        """

        # Filter out rows without embeddings and non-grouped entries
        filtered_df = dataframe[
            (dataframe[DataFrameKeys.embeddings_key].notna())
            & (dataframe[DataFrameKeys.cluster_name] != ClusterSpecificKeys.non_grouped_key)
        ].copy()

        if filtered_df.empty:
            print(f"No valid clustered embeddings found DataFrame: {run_id}")

        already_existing_issues = dataframe[~(pd.isna(dataframe[DataFrameKeys.grouped_from_faiss]))].copy()

        if run_id:
            self._update_processed_runids(self.base_path, run_id)

        for type_, typed_dataframe in filtered_df.groupby("type"):
            # Create directory for this type
            type_dir = os.path.join(self.base_path, f"{type_}_custom")
            os.makedirs(type_dir, exist_ok=True)

            from src.helpers import update_error_map_qgenie_table

            update_error_map_qgenie_table(typed_dataframe)

            if self._check_existing_faiss_for_type(type_):
                self.update(typed_dataframe, type_, run_id=run_id)
                continue

            # Compute centroids
            centroids = self._compute_centroids(typed_dataframe)

            # Create metadata in the format you're currently using
            metadata = {}
            for cluster_name, data in centroids.items():
                metadata[cluster_name] = {"class": data["class"], "run_ids": {}}

                if run_id:
                    metadata[cluster_name]["run_ids"][run_id] = self._get_run_ids_metadata_dict(data, typed_dataframe)

            assert len(centroids) == len(metadata), (
                f"CRITICAL ERROR: Number of centroids ({len(centroids)}) "
                f"does not match number of metadata entries ({len(metadata)})"
            )
            # Save centroids
            with open(os.path.join(type_dir, "centroids.npy"), "wb") as f:
                np.save(
                    f,
                    np.array([centroids[name]["centroid"] for name in centroids.keys()]),
                )

            # Save metadata
            with open(os.path.join(type_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=3)

            print(f"Saved {len(centroids)} clusters for type: {type_}")

    def update(
        self,
        filtered_df: pd.DataFrame,
        type_: str,
        similarity_threshold: float = 0.95,
        run_id: Optional[str] = None,
    ) -> None:
        """
        Update existing embeddings with new data.

        Args:
            dataframe: DataFrame containing new embeddings and metadata
            type_: Type of data (e.g., 'benchmark', 'test')
            similarity_threshold: Threshold for merging clusters
            run_id: Optional run ID to track
        """
        type_dir = os.path.join(self.base_path, f"{type_}_custom")

        # Load existing data
        with open(os.path.join(type_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        with open(os.path.join(type_dir, "centroids.npy"), "rb") as f:
            existing_centroids = np.load(f)

        # Compute centroids for new data
        new_centroids_dict = self._compute_centroids(filtered_df)

        # For each new centroid, check if it's similar to an existing one
        updated_metadata = metadata.copy()
        updated_centroids = existing_centroids.copy()
        existing_cluster_names = list(metadata.keys())

        for cluster_name, data in new_centroids_dict.items():
            new_centroid = data["centroid"]
            new_class = data["class"]

            # Normalize the new centroid
            new_centroid = self._normalize_vectors(np.array([new_centroid]))[0]

            # Compute similarities
            similarities = cosine_similarity([new_centroid], existing_centroids)[0]

            if len(similarities) > 0:
                max_sim_idx = np.argmax(similarities)
                max_sim = similarities[max_sim_idx]

                if (max_sim >= similarity_threshold) or cluster_name in updated_metadata:

                    # Merge with existing cluster
                    if cluster_name not in updated_metadata:
                        existing_cluster_name = existing_cluster_names[max_sim_idx]
                        print(
                            f"Merging new cluster {cluster_name} with existing {existing_cluster_name} (sim={max_sim:.3f})"
                        )
                    else:
                        existing_cluster_name = cluster_name
                        print(f"Cluster Name: {cluster_name} already exists in metadata updating with same")
                        for idx, _cluster_name in enumerate(existing_cluster_names):
                            if _cluster_name == cluster_name:
                                max_sim_idx = idx
                                break

                    # Update centroid (weighted average)
                    updated_centroids[max_sim_idx] = 0.7 * updated_centroids[max_sim_idx] + 0.3 * new_centroid

                    # Update run_ids in metadata
                    if run_id and run_id not in updated_metadata[existing_cluster_name]["run_ids"]:
                        updated_metadata[existing_cluster_name]["run_ids"][run_id] = self._get_run_ids_metadata_dict(
                            data, filtered_df
                        )
                else:
                    # Add as new cluster
                    updated_centroids = np.vstack([updated_centroids, [new_centroid]])
                    updated_metadata[cluster_name] = {"class": new_class, "run_ids": {}}
                    if run_id:
                        updated_metadata[cluster_name]["run_ids"][run_id] = self._get_run_ids_metadata_dict(
                            data, filtered_df
                        )
                    existing_cluster_names.append(cluster_name)
            else:
                # Add as new cluster (first cluster)
                updated_centroids = np.array([new_centroid])
                updated_metadata[cluster_name] = {"class": new_class, "run_ids": {}}
                if run_id:
                    updated_metadata[cluster_name]["run_ids"][run_id] = self._get_run_ids_metadata_dict(
                        data, filtered_df
                    )
                existing_cluster_names.append(cluster_name)

        assert len(updated_centroids) == len(updated_metadata), (
            f"CRITICAL ERROR: Number of centroids ({len(updated_centroids)}) "
            f"does not match number of metadata entries ({len(updated_metadata)})"
        )

        # Save updated data
        with open(os.path.join(type_dir, "centroids.npy"), "wb") as f:
            np.save(f, updated_centroids)

        with open(os.path.join(type_dir, "metadata.json"), "w") as f:
            json.dump(updated_metadata, f, indent=3)

        print(f"Updated to {len(updated_metadata)} clusters for type: {type_}")

    def search(self, type_: str, query: str, similarity_threshold: float = 0.95) -> Tuple[str, str]:
        """
        Search for similar clusters for a single query.

        Args:
            type_: Type of data to search in
            query: Query text
            similarity_threshold: Minimum similarity threshold

        Returns:
            Tuple of (cluster_name, class_name)
        """
        # Get embedding for query
        query_embedding = FallbackEmbeddings().embed_query(query)
        query_embedding = self._normalize_vectors(np.array([query_embedding]))[0]

        return self._search_with_embedding(type_, query_embedding, query, similarity_threshold)

    def _search_with_embedding(
        self,
        type_: str,
        query_embedding: np.ndarray,
        original_query: str = "",
        similarity_threshold: float = 0.95,
    ) -> Tuple[str, str]:
        """
        Search using a precomputed embedding.
        """
        try:
            type_dir = os.path.join(self.base_path, f"{type_}_custom")

            if not os.path.exists(type_dir) or not os.path.exists(os.path.join(type_dir, "metadata.json")):
                print(f"No data found for type: {type_}")
                return ClusterSpecificKeys.non_grouped_key, np.nan

            # Load data
            data_loading_retry = 3
            metadata, centroids = None, None
            while data_loading_retry:
                try:
                    with open(os.path.join(type_dir, "metadata.json"), "r") as f:
                        metadata = json.load(f)
                    with open(os.path.join(type_dir, "centroids.npy"), "rb") as f:
                        centroids = np.load(f)
                except Exception as e:
                    print(f"Exception occured while loading data for batch search: {e}")
                    print(f"retrying in 5 seconds")
                    time.sleep(5)

                if metadata is not None and isinstance(metadata, dict):
                    break
                data_loading_retry -= 1

            if metadata is None or not isinstance(metadata, dict):
                return ClusterSpecificKeys.non_grouped_key, np.nan

            cluster_names = list(metadata.keys())
            # Compute similarities
            similarities = cosine_similarity([query_embedding], centroids)[0]
            cluster_name = ""

            # Get best match
            if len(similarities) > 0:
                max_sim_idx = np.argmax(similarities)
                max_sim = similarities[max_sim_idx]

                if max_sim >= similarity_threshold:
                    cluster_name = cluster_names[max_sim_idx]
                    class_name = metadata[cluster_name]["class"]

                    print(f"For query: '{original_query}', found match: {cluster_name} with similarity {max_sim:.3f}")
                    return cluster_name, class_name

            print(
                f"No match found for query: '{original_query}' (best similarity: {max(similarities) if len(similarities) > 0 else 0:.3f}) for: {cluster_name}"
            )
        except Exception as e:
            raise Exception(traceback.format_exc())

        return ClusterSpecificKeys.non_grouped_key, np.nan

    def _batch_search_with_embeddings(
        self,
        type_: str,
        query_embeddings: np.ndarray,
        original_queries: List[str],
        similarity_threshold: float = 0.95,
    ) -> Tuple[List[str], List[str]]:
        """
        Search using precomputed embeddings for multiple queries at once.
        """
        try:
            type_dir = os.path.join(self.base_path, f"{type_}_custom")
            if not os.path.exists(type_dir) or not os.path.exists(os.path.join(type_dir, "metadata.json")):
                print(f"No data found for type: {type_}")
                return [ClusterSpecificKeys.non_grouped_key] * len(original_queries), [np.nan] * len(original_queries)

            # Load data
            data_loading_retry = 3
            metadata, centroids = None, None
            while data_loading_retry:
                try:
                    with open(os.path.join(type_dir, "metadata.json"), "r") as f:
                        metadata = json.load(f)
                    with open(os.path.join(type_dir, "centroids.npy"), "rb") as f:
                        centroids = np.load(f)
                except Exception as e:
                    print(f"Exception occured while loading data for batch search: {e}")
                    print(f"retrying in 5 seconds")
                    time.sleep(5)

                if metadata is not None and isinstance(metadata, dict):
                    break
                data_loading_retry -= 1

            result_cluster_names = []
            result_class_names = []

            if metadata is None or not isinstance(metadata, dict):
                return [ClusterSpecificKeys.non_grouped_key] * len(original_queries), [np.nan] * len(original_queries)

            cluster_names = list(metadata.keys())
            # Compute similarities for all queries at once
            similarities = cosine_similarity(query_embeddings, centroids)

            # Process each query's results
            for _, (query, sim_row) in enumerate(zip(original_queries, similarities)):
                cluster_name = ""
                if len(sim_row) > 0:
                    max_sim_idx = np.argmax(sim_row)
                    max_sim = sim_row[max_sim_idx]
                    if max_sim >= similarity_threshold:
                        cluster_name = cluster_names[max_sim_idx]
                        class_name = metadata[cluster_name]["class"]
                        print(f"For query: '{query}', found match: {cluster_name} with similarity {max_sim:.3f}")
                        result_cluster_names.append(cluster_name)
                        result_class_names.append(class_name)
                        continue
                    print(
                        f"No match found for query: '{query}' (best similarity: {max(sim_row):.3f}): for: {cluster_name}"
                    )
                result_cluster_names.append(ClusterSpecificKeys.non_grouped_key)
                result_class_names.append(np.nan)

            return result_cluster_names, result_class_names
        except Exception as e:
            raise Exception(traceback.format_exc())

    async def batch_search(
        self,
        type_: str,
        queries: Union[str, List[str]],
        similarity_threshold: float = 0.95,
    ) -> Tuple[List[str], List[str]]:
        """
        Search for similar clusters for multiple queries.

        Args:
            type_: Type of data to search in
            queries: List of query texts or single query
            similarity_threshold: Minimum similarity threshold

        Returns:
            Tuple of (cluster_names, class_names)
        """
        if isinstance(queries, str):
            queries = [queries]

        # Get embeddings for all queries at once
        embeddings = await FallbackEmbeddings().aembed(queries)
        embeddings = self._normalize_vectors(np.array(embeddings))

        cluster_names = []
        class_names = []

        cluster_name, class_name = self._batch_search_with_embeddings(type_, embeddings, queries, similarity_threshold)
        cluster_names.extend(cluster_name)
        class_names.extend(class_name)

        return cluster_names, class_names

    def add_jira_id(self, type_: str, cluster_name: str, jira_id: str) -> bool:
        """
        Add a JIRA ID to a cluster's metadata.

        Args:
            type_: Type of data
            cluster_name: Name of the cluster
            jira_id: JIRA ID to add

        Returns:
            True if successful, False otherwise
        """
        type_dir = os.path.join(self.base_path, f"{type_}_custom")

        if not os.path.exists(type_dir) or not os.path.exists(os.path.join(type_dir, "metadata.json")):
            print(f"No data found for type: {type_}")
            return False

        # Load metadata
        with open(os.path.join(type_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        if cluster_name not in metadata:
            print(f"Cluster {cluster_name} not found in type {type_}")
            return False

        # Add or update JIRA ID
        metadata[cluster_name].setdefault("jira_ids", [])
        if jira_id not in metadata[cluster_name]["jira_ids"]:
            metadata[cluster_name]["jira_ids"].append(jira_id)

        # Save updated metadata
        with open(os.path.join(type_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=3)

        return True

    def get_all_clusters(self, type_: str) -> Dict:
        """
        Get all clusters and their metadata for a given type.

        Args:
            type_: Type of data

        Returns:
            Dictionary of cluster metadata
        """
        type_dir = os.path.join(self.base_path, f"{type_}_custom")

        if not os.path.exists(type_dir) or not os.path.exists(os.path.join(type_dir, "metadata.json")):
            print(f"No data found for type: {type_}")
            return {}

        # Load metadata
        with open(os.path.join(type_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        return metadata

    def _update_processed_runids(self, type_dir: str, run_id: str) -> None:
        """Update the list of processed run IDs."""
        processed_runids_file = os.path.join(type_dir, "processed_runids.json")
        os.makedirs(type_dir, exist_ok=True)

        if os.path.exists(processed_runids_file):
            with open(processed_runids_file, "r") as f:
                processed_runids = json.load(f)
        else:
            processed_runids = []

        if run_id not in processed_runids:
            processed_runids.append(run_id)

        with open(processed_runids_file, "w") as f:
            json.dump(processed_runids, f, indent=3)
