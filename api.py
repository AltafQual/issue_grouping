import asyncio
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict

import pandas as pd
from cachetools import TTLCache
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src import helpers
from src.constants import DataFrameKeys
from src.custom_clustering import CustomEmbeddingCluster
from src.failure_analyzer import FailureAnalyzer

analyzer = FailureAnalyzer()

# 24hrs(in seconds) * 15 days
TTL_CACHE = TTLCache(ttl=(86400 * 15), maxsize=100)


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
    error_message: str = ""
    time_taken: float = 0
    run_id_a: str = ""
    run_id_b: str = ""
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
    asyncio.create_task(asyncio.to_thread(helpers.tc_id_scheduler))
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


@app.get("/api/get_error_cluster_name/")
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
    cluster_name, cluster_class = CustomEmbeddingCluster().search(error_object.type, error)

    if cluster_class != -1:
        response_metadata["cluster_name"] = cluster_name
        response_metadata["cluster_class"] = cluster_class
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
    new_cluster = await helpers.async_sequential_process_by_type(dataframe)
    clustered_df = new_cluster[error_object.type]
    cluster_name = clustered_df.iloc[0][DataFrameKeys.cluster_name]
    class_name = clustered_df.iloc[0][DataFrameKeys.cluster_class]
    _id = helpers.get_error_group_id(error_object.type, error_object.runtime, cluster_name)
    response_metadata["cluster_name"] = cluster_name
    response_metadata["cluster_class"] = class_name
    return {"id": _id, "metadata": response_metadata}


@app.post("/api/initiate_issue_grouping/")
async def inititate_issue_grouping(tc_id_object: InitiateIssueGrouping, background_tasks: BackgroundTasks) -> Dict:
    run_id = tc_id_object.run_id
    data = helpers.sql_connection.fetch_result_based_on_runid(run_id)
    if data.empty:
        return {"status": f"Error: No data found for the Run ID: {run_id}"}

    background_tasks.add_task(
        helpers.async_sequential_process_by_type, data, update_faiss_and_sql=True, run_id=run_id.strip()
    )
    return {"status": f"Successfully Started processing: {run_id}"}


@app.post("/api/get_two_run_ids_cluster_info/", response_model=ClusterInfoResponse)
async def get_two_run_ids_cluster_info(cluster_info_object: ClusterInfo) -> Dict:
    print(f"Received run ids: {cluster_info_object}")
    start_time = time.time()
    response = ClusterInfoResponse()
    response.run_id_a = cluster_info_object.run_id_a
    response.run_id_b = cluster_info_object.run_id_b
    if (cluster_info_object.run_id_a, cluster_info_object.run_id_b) in TTL_CACHE:
        result = TTL_CACHE[(cluster_info_object.run_id_a, cluster_info_object.run_id_b)]
        result['time_taken'] = round(time.time() - start_time)
        return result 
    try:
        results = helpers.find_regressions_between_two_tests(cluster_info_object.run_id_a, cluster_info_object.run_id_b)
        if not results.empty:
            new_cluster = await helpers.async_sequential_process_by_type(results)
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

                response.type[test_type] = {}
                for runtime, runtime_df in df.groupby("runtime"):
                    response.type[test_type][runtime] = {}
                    for cluster_name, cluster_df in runtime_df.groupby(DataFrameKeys.cluster_name):
                        cluster_entries = cluster_df.to_dict(orient="records")
                        response.type[test_type][runtime][cluster_name] = cluster_entries

                for model_name, model_df in df.groupby("name"):
                    model_cluster_details = model_df.to_dict(orient="records")
                    if model_name not in response.model:
                        response.model[model_name] = []
                    response.model[model_name].extend(model_cluster_details)
        else:
            response.status = 404
            response.error_message = (
                f"No data Found for regression between: {cluster_info_object.run_id_a} - {cluster_info_object.run_id_b}"
            )
    except Exception as e:
        print(f"Exception occured while finding regression: {e}")
        print(traceback.format_exc())
        response.status = 500
    response.time_taken = round(time.time() - start_time, 2)
    result = response.to_dict()
    if response.status == 200:
        TTL_CACHE[(cluster_info_object.run_id_a, cluster_info_object.run_id_b)] = result
    return result
