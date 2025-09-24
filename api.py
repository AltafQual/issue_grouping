import json
import os

import faiss
import numpy as np
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src import helpers
from src.constants import FaissConfigurations
from src.embeddings import QGenieBGEM3Embedding
from src.failure_analyzer import FailureAnalyzer

analyzer = FailureAnalyzer()


class ErrorLog(BaseModel):
    type: str = Field(description="Type to which error belongs ex: SaveContext, Converter, Quantizer etc..")
    error: str = Field(description="Error message")
    runtime: str = Field(description="Runtime at which the error occurred")


class InitiateIssueGrouping(BaseModel):
    tc_id: str = Field(description="TC UUID of the test you want to run issue grouping on")


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
    # process query
    error = error_object.error
    error = helpers.preprocess_error_log(error)
    error = helpers.mask_numbers(error)
    error = helpers.trim(error)

    base_path = os.path.join(FaissConfigurations.base_path, f"{error_object.type}_faiss")
    faiss_db = faiss.read_index(os.path.join(base_path, "index.faiss").lower())
    metadata = json.loads(open(os.path.join(base_path, "metadata.json")).read())

    # distance, indices
    D, I = faiss_db.search(np.array(QGenieBGEM3Embedding().embed_query(error)).reshape(1, -1), 1)
    index = int(I[0][0])
    score = float(D[0][0])
    cluster_name = metadata["cluster_names"][index]
    error_group_id = helpers.get_error_group_id(error_object.type, error_object.runtime, cluster_name)
    return {
        "id": error_group_id,
        "metadata": {
            "cluster_name": cluster_name,
            "cluster_score": round(score, 2),
            "runtime": error_object.runtime,
        },
    }


@app.post("/api/initiate_issue_grouping/")
async def get_error_cluster_name(tc_id_object: InitiateIssueGrouping, background_tasks: BackgroundTasks):
    tc_id = tc_id_object.tc_id
    data = helpers.sql_connection.fetch_result_based_on_runid(tc_id)
    if data.empty:
        return {"status": f"Error: No data found for the TC UUID: {tc_id}"}

    background_tasks.add_task(helpers.concurrent_process_by_type, data, update_faiss_and_sql=True)
    return {"status": f"Successfully Started processing: {tc_id}"}
