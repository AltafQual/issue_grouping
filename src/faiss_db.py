import json
import math
import os

import faiss
import numpy as np
import pandas as pd

from src.constants import DataFrameKeys, FaissConfigurations
from src.embeddings import QGenieBGEM3Embedding


class EmbeddingsDB(object):

    def save(self):
        raise NotImplementedError

    def search(self):
        raise NotImplementedError


def calculate_nlist(num_embeddings: int) -> int:
    return max(1, int(math.sqrt(num_embeddings)))


class FaissIVFFlatIndex(EmbeddingsDB):

    def _check_existing_faiss_for_type(self, type):
        type_based_path = os.path.join(FaissConfigurations.base_path, type)
        if os.path.exists(type_based_path):
            if os.path.isfile(os.path.join(type_based_path, "index.faiss")):
                return True

        print(f"Existing FAISS index not found for type: {type}. Creating a new one.")
        return False

    def save(self, dataframe: pd.DataFrame, faiss_dir_path: str = FaissConfigurations.base_path):
        d = 1024
        for t, df in dataframe.groupby("type"):
            metadata = {}
            print(f"Saving FAISS for type: {t}")

            # if self._check_existing_faiss_for_type(t):
            #     print(f"Existing FAISS index found for type: {t}. Skipping creation.")
            #     continue

            filtered_df = df[df[DataFrameKeys.embeddings_key].notna()]
            embeddings_grouped = filtered_df.groupby(DataFrameKeys.cluster_name)[DataFrameKeys.embeddings_key].apply(
                lambda x: np.mean(np.vstack(x), axis=0)
            )
            metadata["cluster_names"] = embeddings_grouped.index.tolist()
            embeddings = np.array(embeddings_grouped.tolist())
            nlist = calculate_nlist(len(embeddings))
            quantizer = faiss.IndexFlatIP(d)
            faiss_db = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            faiss_db.train(embeddings)
            faiss_db.add(embeddings)
            faiss_db.nprobe = 100
            os.makedirs(faiss_dir_path, exist_ok=True)

            base_path = os.path.join(faiss_dir_path, f"{t}_faiss")
            faiss.write_index(faiss_db, os.path.join(base_path, "index.faiss").lower())

            with open(os.path.join(base_path, "metadata.json"), "w") as f:
                f.write(json.dumps(metadata, indent=3))

    def load(self, type: str):
        db_path = os.path.join(FaissConfigurations.base_path, f"{type}_faiss")
        faiss_db = faiss.read_index(os.path.join(db_path, "index.faiss").lower())
        metadata = json.loads(open(os.path.join(db_path, "metadata.json")).read())
        return faiss_db, metadata

    def search(self, type: str, query: str, k: int = 2):
        faiss_db, metadata = self.load(type)
        similar_clusters = faiss_db.search(np.array(QGenieBGEM3Embedding().embed_query(query)).reshape(-1, 1), k=k)
        return similar_clusters
