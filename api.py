import json
import os
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
    tc_id: str = Field(description="TC UUID of the test you want to run issue grouping on")


class Regression(BaseModel):
    run_id_a: str = Field(description="first valid test case TcUUID")
    run_id_b: str = Field(description="second valid test case TcUUID")


class ClusterInfo(BaseModel):
    run_id_a: str = Field(description="first valid test case TcUUID")
    run_id_b: str = Field(description="second valid test case TcUUID")


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
    tc_id = tc_id_object.tc_id
    data = helpers.sql_connection.fetch_result_based_on_runid(tc_id)
    if data.empty:
        return {"status": f"Error: No data found for the TC UUID: {tc_id}"}

    background_tasks.add_task(helpers.async_sequential_process_by_type, data, update_faiss_and_sql=True)
    return {"status": f"Successfully Started processing: {tc_id}"}


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

            grouped = clustered_df.groupby(DataFrameKeys.cluster_name)
            for cluster_name, group in grouped:
                response.data[cluster_name] = {
                    "tc_uuids": group["tc_uuid"].tolist(),
                    "runtimes": group["runtime"].tolist(),
                    "soc_names": group["soc_name"].tolist(),
                }

        else:
            response.status = 204
    except Exception as e:
        print(f"Exception occured while finding regression: {e}")
        response.status = 500

    return response.to_dict()


@app.post("/api/get_two_run_ids_cluster_info/", response_model=ClusterInfoResponse)
async def get_two_run_ids_cluster_info(cluster_info_object: ClusterInfo) -> Dict:
    response = ClusterInfoResponse()
    try:
        results = helpers.find_regressions_between_two_tests(cluster_info_object.run_id_a, cluster_info_object.run_id_b)

        if not results.empty:
            new_cluster = await helpers.async_sequential_process_by_type(results)
            for test_type, df in new_cluster.items():
                for runtime, runtime_df in df.groupby("runtime"):
                    for cluster_name, cluster_df in runtime_df.groupby(DataFrameKeys.cluster_name):
                        new_entry = [
                            {
                                "tc_uuid": row["tc_uuid"],
                                "soc_name": row["soc_name"],
                                "runtime": row["runtime"],
                                "cluster_name": row[DataFrameKeys.cluster_name],
                            }
                            for _, row in cluster_df.iterrows()
                        ]

                        response.type.setdefault(test_type, {})
                        if test_type.lower() not in {"converter", "quantizer", "savecontext"}:
                            response.type[test_type].setdefault(runtime, {})
                            response.type[test_type][runtime].setdefault(cluster_name, [])
                            response.type[test_type][runtime][cluster_name].extend(new_entry)
                        else:
                            response.type[test_type].setdefault(cluster_name, [])
                            response.type[test_type][cluster_name].extend(new_entry)

                for model_name, model_df in df.groupby("name"):
                    response.model.setdefault(model_name, [])
                    model_cluster_details = [
                        {
                            "tc_uuid": row["tc_uuid"],
                            "soc_name": row["soc_name"],
                            "runtime": row["runtime"],
                            "cluster_name": row[DataFrameKeys.cluster_name],
                        }
                        for _, row in model_df.iterrows()
                    ]
                    response.model[model_name].extend(model_cluster_details)
        else:
            response.status = 204

    except Exception as e:
        print(f"Exception occured while finding regression: {e}")
        response.status = 500

    return response.to_dict()
