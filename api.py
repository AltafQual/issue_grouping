import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import faiss
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src import helpers
from src.constants import DataFrameKeys, FaissConfigurations
from src.embeddings import QGenieBGEM3Embedding
from src.failure_analyzer import FailureAnalyzer

analyzer = FailureAnalyzer()


class ErrorLog(BaseModel):
    type: str = Field(description="Type to which error belongs ex: SaveContext, Converter, Quantizer etc..")
    error: str = Field(description="Error message")
    runtime: str = Field(description="Runtime at which the error occurred")


class InitiateIssueGrouping(BaseModel):
    run_id: str = Field(description="Run ID of the test you want to run issue grouping on")


class Regression(BaseModel):
    run_id_a: str = Field(description="first valid test case Run ID")
    run_id_b: str = Field(description="second valid test case Run ID")


class ClusterInfo(BaseModel):
    run_id_a: str = Field(description="first valid test case Run ID")
    run_id_b: str = Field(description="second valid test case Run ID")


class RegressionResponse(BaseModel):
    status: int = 200
    data: Dict[str, Any] = {}

    def add(self, key: str, value: Any) -> None:
        self.data[key] = value

    def to_dict(self) -> Dict:
        return self.dict()


class ClusterInfoResponse(BaseModel):
    status: int = 200
    type: Dict[str, Any] = {}
    model: Dict[str, Any] = {}

    def add(self, key: str, value: Any, data_type: str) -> None:
        if data_type == "type":
            self.type[key] = value
        elif data_type == "model":
            self.model[key] = value
        print(f"Data: {key}: {value} not added valid `data_type` not provided")

    def to_dict(self) -> Dict:
        return self.dict()

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(helpers.faiss_update_worker())
    yield


app = FastAPI(
    title="IssueGrouping",
    contact={"name": "Mohammed Altaf", "email": "altaf@qti.qualcomm.com"},
    docs_url="/api",
    redoc_url=None,
    lifespan=lifespan,
)


@app.get("/docs", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/api")


@app.post("/api/get_error_cluster_name/")
async def get_error_cluster_name(error_object: ErrorLog) -> Dict:
    """
    This API provides the cluster name to which the error belongs to.
    """
    # process query
    error = error_object.error
    error = helpers.preprocess_error_log(error)
    error = helpers.mask_numbers(error)
    error = helpers.trim(error)

    response_metadata = {
        "runtime": error_object.runtime,
    }

    base_path = os.path.join(FaissConfigurations.base_path, f"{error_object.type}_faiss")
    faiss_db = faiss.read_index(os.path.join(base_path, "index.faiss").lower())
    metadata = json.loads(open(os.path.join(base_path, "metadata.json")).read())

    # distance, indices
    D, I = faiss_db.search(np.array(QGenieBGEM3Embedding().embed_query(error)).reshape(1, -1), 1)
    index = int(I[0][0])
    score = round(float(D[0][0]), 2)
    if score >= 80:
        cluster_name = metadata["cluster_names"][index]
        response_metadata["cluster_name"] = cluster_name
        response_metadata["cluster_score"] = score
        error_group_id = helpers.get_error_group_id(error_object.type, error_object.runtime, cluster_name)
        return {"id": error_group_id, "metadata": response_metadata}

    dataframe = pd.DataFrame(
        {
            "tc_uuid": [""],
            "reason": [error_object.error],
            "type": [error_object.type],
            "runtime": [error_object.runtime],
            "soc_name": [""],
        }
    )
    new_cluster = await helpers.async_sequential_process_by_type(dataframe, update_faiss_and_sql=True)
    clustered_df = new_cluster[error_object.type]
    cluster_name = clustered_df.iloc[0][DataFrameKeys.cluster_name]
    _id = helpers.get_error_group_id(error_object.type, error_object.runtime, cluster_name)
    response_metadata["cluster_name"] = cluster_name
    response_metadata["cluster_score"] = 1
    return {"id": _id, "metadata": response_metadata}


@app.post("/api/initiate_issue_grouping/")
async def get_error_cluster_name(tc_id_object: InitiateIssueGrouping, background_tasks: BackgroundTasks) -> Dict:
    run_id = tc_id_object.run_id
    data = helpers.sql_connection.fetch_result_based_on_runid(run_id)
    if data.empty:
        return {"status": f"Error: No data found for the Run ID: {run_id}"}

    background_tasks.add_task(
        helpers.async_sequential_process_by_type, data, update_faiss_and_sql=True, run_id=run_id.strip()
    )
    return {"status": f"Successfully Started processing: {run_id}"}


@app.post("/api/regression_between_two_tests/", response_model=RegressionResponse)
async def get_regression_between_two_tests(regression_object: Regression) -> Dict:
    response = RegressionResponse()
    try:
        results = helpers.find_regressions_between_two_tests(regression_object.run_id_a, regression_object.run_id_b)

        if not results.empty:
            new_cluster = await helpers.async_sequential_process_by_type(results)
            clustered_df = pd.concat(
                [df.assign(cluster_type=cluster_name) for cluster_name, df in new_cluster.items()],
                ignore_index=True,
            )

            clustered_df = clustered_df.drop(
                columns=[
                    col
                    for col in [
                        DataFrameKeys.embeddings_key,
                        DataFrameKeys.bins,
                        DataFrameKeys.error_logs_length,
                        DataFrameKeys.cluster_type_int,
                        DataFrameKeys.preprocessed_text_key,
                        DataFrameKeys.grouped_from_faiss,
                    ]
                    if col in clustered_df.columns
                ]
            )
            grouped = clustered_df.groupby(DataFrameKeys.cluster_name)
            for cluster_name, group in grouped:
                response.data[cluster_name] = group.to_dict(orient="records")

        else:
            response.status = 204

    except Exception as e:
        print(f"Exception occured while finding regression: {e}")
        response.status = 500

    return response.to_dict()


@app.post("/api/get_two_run_ids_cluster_info/", response_model=ClusterInfoResponse)
async def get_two_run_ids_cluster_info(cluster_info_object: ClusterInfo) -> Dict:
    print(f"Received run ids: {cluster_info_object}")
    response = ClusterInfoResponse()
    try:
        results = helpers.find_regressions_between_two_tests(cluster_info_object.run_id_a, cluster_info_object.run_id_b)

        if not results.empty:
            new_cluster = await helpers.concurrent_process_by_type(results, update_faiss_and_sql=True)
            for test_type, df in new_cluster.items():
                df = df.drop(
                    columns=[
                        col
                        for col in [
                            DataFrameKeys.embeddings_key,
                            DataFrameKeys.bins,
                            DataFrameKeys.error_logs_length,
                            DataFrameKeys.cluster_type_int,
                            DataFrameKeys.preprocessed_text_key,
                            DataFrameKeys.grouped_from_faiss,
                        ]
                        if col in df.columns
                    ]
                )
                for runtime, runtime_df in df.groupby("runtime"):
                    for cluster_name, cluster_df in runtime_df.groupby(DataFrameKeys.cluster_name):
                        new_entry = cluster_df.to_dict(orient="records")

                        response.type.setdefault(test_type, {})
                        response.type[test_type].setdefault(runtime, {})
                        response.type[test_type][runtime].setdefault(cluster_name, [])
                        response.type[test_type][runtime][cluster_name].extend(new_entry)

                for model_name, model_df in df.groupby("name"):
                    response.model.setdefault(model_name, [])
                    model_cluster_details = model_df.to_dict(orient="records")
                    response.model[model_name].extend(model_cluster_details)
        else:
            response.status = 404

    except Exception as e:
        print(f"Exception occured while finding regression: {e}")
        response.status = 500

    return response.to_dict()
