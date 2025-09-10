import numpy as np
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from src.constants import DataFrameKeys, FaissConfigurations


def save_to_faiss(df: pd.DataFrame, file_name: str, path: str = FaissConfigurations.base_path):
    import pdb

    pdb.set_trace()
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


def load_from_faiss(file_name: str, path: str = FaissConfigurations.base_path):
    return FAISS.load_local(f"{path}/", embeddings=None)
