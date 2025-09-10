import numpy as np
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from src.constants import DataFrameKeys, FaissConfigurations


class EmbeddingsDB(object):
    def __init__(self):
        raise NotImplementedError

    def save(self):
        pass

    def load(self):
        pass


class FaissDB(EmbeddingsDB):
    def __init__(self, file_name: str = None, path: str = FaissConfigurations.base_path):
        self.file_name = file_name
        self.path = path
        self.faiss_db = self.load_from_faiss(file_name, path)

    def save_to_faiss(self, df: pd.DataFrame, file_name: str, path: str = FaissConfigurations.base_path):
        documents = [
            Document(
                page_content=row[DataFrameKeys.preprocessed_text_key],
                metadata={"issue_id": idx, "cluster_name": row[DataFrameKeys.cluster_name]},
            )
            for idx, row in df.iterrows()
        ]

        embeddings = np.array(df[DataFrameKeys.embeddings_key].tolist())
        faiss_db = FAISS.from_embeddings(embeddings=embeddings, documents=documents)
        db_name = file_name if file_name else FaissConfigurations.default_db_name
        faiss_db.save_local(f"{path}/{db_name}")
        return faiss_db

    def save(self) -> None:
        self.save_to_faiss(self.faiss_db, self.file_name, self.path)

    def load_from_faiss(file_name: str, path: str = FaissConfigurations.base_path):
        return FAISS.load_local(f"{path}/", embeddings=None)

    def load(self):
        db_name = self.file_name if self.file_name else FaissConfigurations.default_db_name
        return FAISS.load(db_name)
