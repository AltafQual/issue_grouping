import os
import json

import faiss
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import numpy as np

from src.constants import FaissConfigurations
from src.embeddings import QGenieBGEM3Embedding


class ErrorLog(BaseModel):
    type: str = Field(description="Type to which error belongs ex: SaveContext, Converter, Quantizer etc..")
    error: str = Field(description="Error message")


app = FastAPI(
    title="IssueGrouping",
    contact={"name": "Mohammed Altaf", "email": "altaf@qti.qualcomm.com"},
    docs_url="/api",
    redoc_url=None,
)


@app.get("/docs", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/api")


@app.post("/api/get_error_cluster_name/")
async def get_error_cluster_name(error_object: ErrorLog):
    """
    This API provides the cluster name to which the error belongs to.
    """
    base_path = os.path.join(FaissConfigurations.base_path, f"{error_object.type}_faiss")
    faiss_db_path = os.path.join(base_path, "index.faiss")
    faiss_db = faiss.read_index(faiss_db_path)
    
    with open(os.path.join(base_path, "metadata.json"), "rb") as f:
        metadata = json.loads(f)

    I, D = faiss_db.search(np.array(QGenieBGEM3Embedding().embed_query(error_object.error)).reshape(1, -1), 1)
    index = int(I[0][0])
    score = float(D[0][0])
        
    result =  {"cluster_name": metadata['cluster_names'][index], "cluster_score": score}
    print(result)
    return result
