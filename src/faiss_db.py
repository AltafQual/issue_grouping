import asyncio
import json
import math
import os
from functools import lru_cache
from typing import Union

import faiss
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.constants import ClusterSpecificKeys, DataFrameKeys, FaissConfigurations
from src.embeddings import QGenieBGEM3Embedding
from src.qgenie import classify_cluster_based_of_type


class EmbeddingsDB(object):

    def save(self):
        raise NotImplementedError

    def search(self):
        raise NotImplementedError


def calculate_nlist(num_embeddings: int) -> int:
    return max(1, int(math.sqrt(num_embeddings)))


class FaissIVFFlatIndex(EmbeddingsDB):

    def _check_existing_faiss_for_type(self, type):
        type_based_path = os.path.join(FaissConfigurations.base_path, f"{type}_faiss")
        if os.path.exists(type_based_path):
            if os.path.isfile(os.path.join(type_based_path, "index.faiss")):
                return True

        print(f"Existing FAISS index not found for type: {type}. Creating a new one.")
        return False

    def update(
        self,
        type_: Union[str, bytes],
        dataframe: pd.DataFrame,
        new_embeddings_grouped: pd.Series,
        faiss_dir_path: str = FaissConfigurations.base_path,
        similarity_threshold: float = 0.95,
        run_id=None,
    ):
        d = 1024
        new_cluster_names = new_embeddings_grouped.index.tolist()
        new_embeddings = np.array(new_embeddings_grouped.tolist())

        base_path = os.path.join(faiss_dir_path, f"{type_}_faiss")
        os.makedirs(base_path, exist_ok=True)

        # If the FAISS index does not exist yet, `faiss_db` will be None.
        # In that case we start with empty embeddings and no existing cluster names.
        existing_embeddings = []
        existing_cluster_names = []

        faiss_db, metadata = self.load(type_)
        if not metadata:
            metadata = {}

        if faiss_db is None:
            # No existing index – start from scratch
            existing_cluster_names = []
            existing_embeddings = np.empty((0, d))
        else:
            existing_cluster_names = list(metadata.keys())
            # Reconstruct existing vectors; ensure we get a NumPy array
            existing_embeddings = faiss_db.reconstruct_n(0, faiss_db.ntotal)
            existing_embeddings = np.array(existing_embeddings)

        print(f"Existing cluster names: {existing_cluster_names}, new cluster names: {new_cluster_names}")
        merged_embeddings = list(existing_embeddings)
        merged_cluster_names = list(existing_cluster_names)

        for i, new_emb in enumerate(new_embeddings):
            is_new = True
            if len(existing_embeddings) > 0:
                # Convert the single new embedding to a 2‑D array for cosine_similarity
                sims = cosine_similarity(np.array([new_emb]), existing_embeddings)[0]
                max_sim_idx = np.argmax(sims)
                if sims[max_sim_idx] >= similarity_threshold:
                    # Update existing centroid by averaging
                    merged_embeddings[max_sim_idx] = np.mean([merged_embeddings[max_sim_idx], new_emb], axis=0)
                    is_new = False

            if is_new:
                # Add new centroid
                merged_embeddings.append(new_emb)
                merged_cluster_names.append(new_cluster_names[i])

        final_embeddings = np.array(merged_embeddings)
        final_cluster_names = merged_cluster_names

        # Rebuild FAISS index
        nlist = calculate_nlist(len(final_embeddings))
        quantizer = faiss.IndexFlatIP(d)
        faiss_db = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        faiss_db.train(final_embeddings)
        faiss_db.add(final_embeddings)
        faiss_db.nprobe = 100

        faiss.write_index(faiss_db, os.path.join(base_path, "index.faiss").lower())

        # This ensures the order of keys in the metadata dict aligns exactly with the
        # order of vectors in the rebuilt FAISS index, preventing out‑of‑range errors.
        ordered_metadata = {}
        for cluster_name in final_cluster_names:
            if cluster_name in metadata:
                # Preserve existing entry (class, run_ids, etc.)
                entry = metadata[cluster_name]
            else:
                # New cluster – compute its class
                from src.helpers import get_cluster_class, sql_connection

                cluster_class = classify_cluster_based_of_type(
                    dataframe[dataframe[DataFrameKeys.cluster_name] == cluster_name].reason.tolist()
                )
                cluster_class = get_cluster_class(cluster_class)
                entry = {"class": cluster_class}
                # Update DB for the new cluster
                sql_connection.update_error_group_class(
                    dataframe[dataframe[DataFrameKeys.cluster_name] == cluster_name],
                    cluster_class,
                )
            # Handle run_id accumulation
            if run_id:
                entry.setdefault("run_ids", [])
                # Ensure run_ids is a list before appending
                if not isinstance(entry["run_ids"], list):
                    entry["run_ids"] = list(entry["run_ids"])
                entry["run_ids"].append(run_id)
                # Optional trimming to keep list size reasonable
                if len(entry["run_ids"]) > 50:
                    entry["run_ids"] = entry["run_ids"][-45:]
            ordered_metadata[cluster_name] = entry

        # Replace metadata with the ordered version
        metadata = ordered_metadata

        with open(os.path.join(base_path, "metadata.json"), "w") as f:
            f.write(json.dumps(metadata, indent=3))

    def write_and_update_processed_runids(self, faiss_dir_path, run_id):
        if not run_id:
            return

        # save the run id to global processed run ids list
        os.makedirs(faiss_dir_path, exist_ok=True)
        if "processed_runids.json" not in os.listdir(faiss_dir_path):
            processed_runids = []
        else:
            processed_runids = json.loads(open(os.path.join(faiss_dir_path, "processed_runids.json")).read())

        processed_runids.append(run_id)

        # Keep only the latest 1000 entries (drop the oldest 10 if exceeded)
        if len(processed_runids) > 1000:
            processed_runids = processed_runids[10:]

        with open(os.path.join(faiss_dir_path, "processed_runids.json"), "w") as f:
            f.write(json.dumps(processed_runids, indent=3))

    def save(self, dataframe: pd.DataFrame, faiss_dir_path: str = FaissConfigurations.base_path, run_id=None):
        from src.helpers import update_error_map_qgenie_table

        self.write_and_update_processed_runids(faiss_dir_path, run_id)
        d = 1024
        for t, df in dataframe.groupby("type"):
            metadata = {}
            print(f"Saving FAISS for type: {t}")
            filtered_df = df[df[DataFrameKeys.embeddings_key].notna()]

            if not filtered_df.empty:
                update_error_map_qgenie_table(filtered_df)

                # Compute the mean embedding for each cluster using a named helper
                def _mean_embeddings(x):
                    return np.mean(np.vstack(x), axis=0)

                embeddings_grouped = filtered_df.groupby(DataFrameKeys.cluster_name)[
                    DataFrameKeys.embeddings_key
                ].apply(_mean_embeddings)

                if not self._check_existing_faiss_for_type(t):

                    if embeddings_grouped.empty:
                        print(f"No embeddings found for type: {t} skipping...")
                        continue
                    embeddings = np.array(embeddings_grouped.tolist())
                    nlist = calculate_nlist(len(embeddings))
                    quantizer = faiss.IndexFlatIP(d)
                    faiss_db = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
                    faiss_db.train(embeddings)
                    faiss_db.add(embeddings)
                    faiss_db.nprobe = 100
                    base_path = os.path.join(faiss_dir_path, f"{t}_faiss")
                    os.makedirs(base_path, exist_ok=True)

                    faiss.write_index(faiss_db, os.path.join(base_path, "index.faiss").lower())

                    for cluster_name in embeddings_grouped.index.tolist():
                        from src.helpers import get_cluster_class, sql_connection

                        cluster_class = classify_cluster_based_of_type(
                            filtered_df[filtered_df[DataFrameKeys.cluster_name] == cluster_name].reason.tolist()
                        )
                        cluster_class = get_cluster_class(cluster_class)
                        metadata[cluster_name] = {"class": cluster_class}
                        metadata[cluster_name].setdefault("run_ids", [])
                        if run_id:
                            metadata[cluster_name]["run_ids"].append(run_id)

                        sql_connection.update_error_group_class(
                            filtered_df[filtered_df[DataFrameKeys.cluster_name] == cluster_name], cluster_class
                        )

                    with open(os.path.join(base_path, "metadata.json"), "w") as f:
                        f.write(json.dumps(metadata, indent=3))
                else:
                    if not embeddings_grouped.empty:
                        print(f"Existing FAISS index found for type: {t}. Updating data")
                        self.update(t, filtered_df, embeddings_grouped, faiss_dir_path, run_id=run_id)

    def load(self, type: str, only_metadata=False):
        db_path = os.path.join(FaissConfigurations.base_path, f"{type}_faiss")
        faiss_db = None
        if only_metadata is False:
            faiss_db = faiss.read_index(os.path.join(db_path, "index.faiss").lower())

        metadata = json.loads(open(os.path.join(db_path, "metadata.json")).read())
        return faiss_db, metadata

    def search(self, type: str, query: str, k: int = 2):
        # Load the index; if it does not exist, return an empty result
        faiss_db, _ = self.load(type)
        if faiss_db is None:
            return []
        similar_clusters = faiss_db.search(
            np.array(QGenieBGEM3Embedding().embed_query(query)).reshape(-1, 1),
            k=k,
        )
        return similar_clusters


class SearchInExistingFaiss(object):
    def __init__(self):
        self.base_path = FaissConfigurations.base_path

    def _load_faiss(self, type: str) -> tuple:
        db_path = os.path.join(self.base_path, f"{type}_faiss")
        if os.path.exists(db_path):
            faiss_db = faiss.read_index(os.path.join(db_path, "index.faiss").lower())
            metadata = json.loads(open(os.path.join(db_path, "metadata.json")).read())
            return faiss_db, metadata
        return None, None

    def search(self, type: str, query: str, k: int = 2) -> Union[str, None]:
        faiss_db, metadata = self._load_faiss(type)
        key = ClusterSpecificKeys.non_grouped_key

        if faiss_db is None:
            return key

        Distance, Index = faiss_db.search(np.array(QGenieBGEM3Embedding().embed_query(query)).reshape(1, -1), k=k)
        index = int(Index[0][0])
        score = float(Distance[0][0])

        if score >= 0.95:
            # Use ordered keys from metadata to map index to cluster
            key = str(list(metadata.keys())[index])

        print(f"For Query: {query}, score: {score}, cluster: {key}")
        return key

    async def batch_search(self, type: str, query: Union[str, list[str]], k: int = 2) -> list[str]:
        faiss_db, metadata = self._load_faiss(type)
        key = ClusterSpecificKeys.non_grouped_key
        queries = [query] if isinstance(query, str) else query

        if faiss_db is None:
            return [key] * len(queries)

        embeddings = await QGenieBGEM3Embedding().aembed_documents(queries)

        # Search in FAISS
        Distance, Index = faiss_db.search(np.array(embeddings), k=k)

        cluster_names = []
        for i, query in enumerate(queries):
            index = int(Index[i][0])
            score = float(Distance[i][0])
            cluster_name = str(list(metadata.keys())[index])

            print(f"For Query: {query}, score: {score}, cluster: {cluster_name}")
            if score >= 0.95:
                cluster_names.append(cluster_name)
            else:
                cluster_names.append(ClusterSpecificKeys.non_grouped_key)

        return cluster_names
