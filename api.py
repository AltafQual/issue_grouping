import asyncio
import json
import logging
import threading
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict

import pandas as pd
import psutil
from cachetools import TTLCache
from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi.responses import ORJSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from src import helpers
from src.consolidated_reports_analysis import CombinedRegressionAnalysis, ConsolidatedReportAnalysis
from src.constants import CONSOLIDATED_REPORTS, DataFrameKeys
from src.custom_clustering import CustomEmbeddingCluster
from src.failure_analyzer import FailureAnalyzer
from src.gerrit_data_fetching_helpers import get_gerrit_info_between_2_runids, get_regression_gerrits_based_of_type

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("memlog")
proc = psutil.Process()
analyzer = FailureAnalyzer()
TTL_CACHE = TTLCache(maxsize=1000, ttl= (604800 * 604800))
LOCK = threading.Lock()


class InitiateIssueGrouping(BaseModel):
    run_id: str = Field(description="Run ID of the test you want to run issue grouping on")


class InitiateConsolidatedReportGeneration(BaseModel):
    run_ids: list = Field(
        description="QAISW id example `qaisw-v2.44.0.260112072337_193906_nightly` to initiate report generation"
    )
    
class ModelOps(BaseModel):
    model_names: list = Field(
        description="list of all the model Names"
    )


class Regression(BaseModel):
    run_id_a: str = Field(description="first valid test case Run ID")
    run_id_b: str = Field(description="second valid test case Run ID")


class ClusterInfo(BaseModel):
    run_id_a: str = Field(description="first valid test case Run ID")
    run_id_b: str = Field(description="second valid test case Run ID")
    force: bool = Field(description="Skip cache and regenrate the regression", default=False)


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
    gerrit_info: Dict[str, Any] = {}

    def add(self, key: str, value: Any, data_type: str) -> None:
        if data_type == "type":
            self.type[key] = value
        elif data_type == "model":
            self.model[key] = value
        logger.info(f"Data: {key}: {value} not added valid `data_type` not provided")

    def to_dict(self) -> Dict:
        return self.dict()


def consolidated_report_worker():
    logger.info("Starting consolidated report analysis job")

    while True:
        try:
            with LOCK:
                try:
                    with open(CONSOLIDATED_REPORTS.PROCESSING_JSON, "r") as f:
                        data = json.load(f)
                except Exception:
                    data = []

            if not data:
                time.sleep(60)
                continue

            run_id = data[0]
            logger.info(f"[Worker] Processing run_id: {run_id}")

            try:
                analysis = CombinedRegressionAnalysis(ConsolidatedReportAnalysis())
                analysis.generate_final_summary_report(run_id)
            except Exception as e:
                logger.info(f"[Worker] Error processing {run_id}: {e}")

            with LOCK:

                with open(CONSOLIDATED_REPORTS.PROCESSING_JSON, "r") as f:
                    data = json.load(f)

                if run_id in data:
                    data.remove(run_id)

                with open(CONSOLIDATED_REPORTS.PROCESSING_JSON, "w") as f:
                    json.dump(data, f, indent=2)

        except Exception as e:
            logger.info(f"[Worker] Unexpected error: {e}")

        time.sleep(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(asyncio.to_thread(helpers.tc_id_scheduler))
    asyncio.create_task(asyncio.to_thread(consolidated_report_worker))
    yield


app = FastAPI(
    title="IssueGrouping",
    contact={"name": "Mohammed Altaf", "email": "altaf@qti.qualcomm.com"},
    docs_url="/api",
    redoc_url=None,
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)


@app.middleware("http")
async def mem_log(request: Request, call_next):
    before = proc.memory_info().rss
    t0 = time.perf_counter()
    resp = await call_next(request)
    after = proc.memory_info().rss
    logger.info(
        "path=%s \nmethod=%s \nstatus=%s \nelapsed_ms=%.2f \nrss_before_mb=%.2f \nrss_after_mb=%.2f \ndelta_mb=%.2f",
        request.url.path,
        request.method,
        resp.status_code,
        (time.perf_counter() - t0) * 1000,
        before / (1024 * 1024),
        after / (1024 * 1024),
        (after - before) / (1024 * 1024),
    )
    return resp


@app.get("/docs", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/api")


@app.get("/api/get_error_cluster_name/")
async def get_error_cluster_name(
    _type: str = Query(..., description="Type ex: SaveContext, Converter, Quantizer etc.."),
    error: str = Query(..., description="Error message"),
    runtime: str = Query(..., description="Runtime at which the error occurred"),
) -> Dict:
    """
    This API provides the cluster name to which the error belongs to.
    """
    # process query
    error = helpers.preprocess_error_log(error)
    error = helpers.mask_numbers(error)
    error = helpers.trim(error)

    response_metadata = {
        "runtime": runtime,
    }
    cluster_name, cluster_class = CustomEmbeddingCluster().search(_type, error)

    if cluster_name != -1:
        response_metadata["cluster_name"] = cluster_name
        response_metadata["cluster_class"] = cluster_class
        error_group_id = helpers.get_error_group_id(_type, runtime, cluster_name)
        return {"id": error_group_id, "metadata": response_metadata}

    dataframe = pd.DataFrame(
        {
            "tc_uuid": [""],
            "reason": [error],
            "type": [_type],
            "runtime": [runtime],
            "soc_name": [""],
        }
    )
    new_cluster = await helpers.async_sequential_process_by_type(dataframe)
    clustered_df = new_cluster[_type]
    cluster_name = clustered_df.iloc[0][DataFrameKeys.cluster_name]
    class_name = clustered_df.iloc[0][DataFrameKeys.cluster_class]
    _id = helpers.get_error_group_id(_type, runtime, cluster_name)
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
        helpers.async_sequential_process_by_type,
        data,
        update_faiss_and_sql=True,
        run_id=run_id.strip(),
    )
    return {"status": f"Successfully Started processing: {run_id}"}


@app.post(
    "/api/get_two_run_ids_cluster_info/",
    response_model=ClusterInfoResponse,
    status_code=200,
)
async def get_two_run_ids_cluster_info(cluster_info_object: ClusterInfo) -> Dict:
    logger.info(f"Received run ids: {cluster_info_object}")
    start_time = time.time()
    response = ClusterInfoResponse()
    response.run_id_a = cluster_info_object.run_id_a
    response.run_id_b = cluster_info_object.run_id_b
    if (
        cluster_info_object.run_id_a,
        cluster_info_object.run_id_b,
    ) in TTL_CACHE and cluster_info_object.force != True:
        result = TTL_CACHE[(cluster_info_object.run_id_a, cluster_info_object.run_id_b)]
        result["time_taken"] = round(time.time() - start_time)
        return result
    try:
        backend_type_mapping = {}
        results = helpers.find_regressions_between_two_tests(cluster_info_object.run_id_a, cluster_info_object.run_id_b)
        if not results.empty:
            new_cluster = await helpers.async_sequential_process_by_type(results)
            for test_type, df in new_cluster.items():
                backend_type_mapping[test_type] = pd.unique(df["runtime"])
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

            gerrit_info = await get_regression_gerrits_based_of_type(
                cluster_info_object.run_id_a,
                cluster_info_object.run_id_b,
                backend_type_mapping,
            )
            if gerrit_info:
                response.gerrit_info = dict(gerrit_info)
        else:
            response.status = 404
            response.error_message = (
                f"No data Found for regression between: {cluster_info_object.run_id_a} - {cluster_info_object.run_id_b}"
            )
    except Exception as e:
        logger.exception(f"Exception occured while finding regression: {e}")
        logger.error(traceback.format_exc())
        response.status = 500
    response.time_taken = round(time.time() - start_time, 2)
    result = response.to_dict()
    if response.status == 200:
        TTL_CACHE[(cluster_info_object.run_id_a, cluster_info_object.run_id_b)] = result

    return result


@app.post("/api/get_gerrits_merged_between_two_run_ids/", status_code=200)
async def get_gerrits_merged_between_two_run_ids(
    cluster_info_object: ClusterInfo,
) -> Dict:
    logger.info(f"Received run ids: {cluster_info_object}")
    return {
        "gerrit_data": await get_gerrit_info_between_2_runids(
            cluster_info_object.run_id_a, cluster_info_object.run_id_b
        )
    }


@app.post("/api/consolidated_report_regression_analysis/", status_code=202)
async def initiate_consolidated_report_regression_analysis(
    initiate_report_generation: InitiateConsolidatedReportGeneration,
):
    with LOCK:
        try:
            with open(CONSOLIDATED_REPORTS.PROCESSING_JSON, "r") as f:
                data = json.load(f)
        except Exception:
            data = []

        non_processing_ids = []
        for rid in initiate_report_generation.run_ids:
            if rid not in data:
                data.append(rid)
            else:
                non_processing_ids.append(rid)

        with open(CONSOLIDATED_REPORTS.PROCESSING_JSON, "w") as f:
            json.dump(data, f, indent=2)

    result = "Successfully queued ids for processing."
    if non_processing_ids:
        result += f" Except {','.join(non_processing_ids)}: These are already in processing"
    return {"response": result}

@app.post("/api/model_ops/", status_code=200)
async def fetch_model_ops(
    model_ops_object: ModelOps,
):
    return helpers.fetch_model_ops(model_ops_object.model_names)