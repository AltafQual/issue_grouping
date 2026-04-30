"""Regression report — two test-run comparison.

Provides :class:`RegressionAnalysisReport` (per-run-ID HTML regression report)
and :class:`RegressionReport` (thin orchestrator) plus the HTTP API helpers
``get_two_run_ids_cluster_info`` and ``get_two_run_ids_cluster_info_async``
(previously in :mod:`src.regression_api_call`).

Layering
--------
Imports from ``src.constants``, ``src.logger``, ``src.llm.client``,
``src.reports.html_renderer``, ``src.reports.kpi_calculator``, and
``src.utils.timer``.  No imports from ``src.consolidated_reports_analysis``
or ``src.regression_api_call``.
"""

from __future__ import annotations

import os

import requests

from src.constants import CONSOLIDATED_REPORTS
from src.llm.client import error_summary_generation
from src.logger import AppLogger
from src.reports.html_renderer import REPORT_CSS, HTMLRenderer
from src.reports.kpi_calculator import generate_executive_summary
from src.utils.timer import execution_timer

try:
    import aiohttp as aiohttp_lib
except ImportError:
    aiohttp_lib = None

logger = AppLogger().get_logger(__name__)

__all__ = [
    "RegressionAnalysisReport",
    "RegressionReport",
    "get_two_run_ids_cluster_info",
    "get_two_run_ids_cluster_info_async",
]

_ISSUE_GROUPING_API_URL = os.getenv("ISSUE_GROUPING_API_URL", "http://hyd-lablnx904:8010")


def get_two_run_ids_cluster_info(run_id_a: str, run_id_b: str, timeout: int = 600, force: bool = False) -> dict:
    """Call the API to get regression comparison for two run IDs.

    Args:
        run_id_a: Latest (candidate) run ID.
        run_id_b: Baseline run ID to compare against.
        timeout: Request timeout in seconds.
        force: Force re-processing even if cached.

    Returns:
        JSON response dict, or empty dict on error.
    """
    url = f"{_ISSUE_GROUPING_API_URL}/api/get_two_run_ids_cluster_info/"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    payload = {"run_id_a": run_id_a, "run_id_b": run_id_b, "force": force}
    try:
        logger.info(f"POST: url={url}, json={payload}")
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching regression between two run IDs: {e}")
        return {}


async def get_two_run_ids_cluster_info_async(
    run_id_a: str, run_id_b: str, timeout: int = 600, force: bool = False
) -> dict:
    """Async version of :func:`get_two_run_ids_cluster_info`.

    Args:
        run_id_a: Latest (candidate) run ID.
        run_id_b: Baseline run ID to compare against.
        timeout: Request timeout in seconds.
        force: Force re-processing even if cached.

    Returns:
        JSON response dict, or empty dict on error.
    """
    if aiohttp_lib is None:
        logger.warning("aiohttp is not installed — cannot make async request.")
        return {}

    url = f"{_ISSUE_GROUPING_API_URL}/api/get_two_run_ids_cluster_info/"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    payload = {"run_id_a": run_id_a, "run_id_b": run_id_b, "force": force}
    try:
        logger.info(f"POST (async): url={url}, json={payload}")
        async with aiohttp_lib.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=timeout) as response:
                response.raise_for_status()
                return await response.json()
    except Exception as e:
        logger.error(f"Error fetching regression between two run IDs (async): {e}")
        return {}


class RegressionAnalysisReport:
    """Builds per-run-ID HTML regression report pages.

    Args:
        qairt_id: QAIRT release identifier used to locate the output folder.
    """

    def __init__(self, qairt_id: str):
        self.model_regressed_errors_list, self.error_summary_list = [], []
        self.types_to_filter_for_regression_analysis = ["bm_regression", "bm_verifier"]
        self.__destination_folder = os.path.join(CONSOLIDATED_REPORTS.path, qairt_id)
        self.__server_prefix = "https://aisw-hyd.qualcomm.com/fs"
        self.regression_data = None
        self.gerrits_information = None
        self.runtime_type_regression_error_data = None
        self.__auto_soc_column_header = [
            "htp",
            "htp_fp16",
            "mcp",
            "gpu",
            "gpu_fp16",
            "cpu",
            "mcp_x86",
            "htp_x86",
            "lpai",
        ]
        self._qairt_id = qairt_id
        self._has_rc_in_runid = False

    def __get_error_qgenie_summary(self, error=None, raw_data_list=None, error_log_key=None, filter_key=None):
        """
        Qgenie Helper to provide summary of error log

        if `error` key is passed to the function
        - it will directly be used to generate the summary

        other wise `raw_data_list` which is the whole regression/custom list of data dictionary can be provided along with `error_log_key`
        - error_log_key will be used to extract the error logs from the raw data
        - along with this `filter_key` can also be provide which will be check in raw data dict while extracting error logs using error_log_key

        """
        final_error_list, error_summary = None, ""

        if error:
            if isinstance(error, str):
                error = [error]
            final_error_list = error

        elif raw_data_list and error_log_key:
            error = []
            for data in raw_data_list:
                error_str = None
                if filter_key and filter_key in data:
                    error_str = data.get(error_log_key)
                elif error_log_key and error_log_key in data:
                    error_str = data.get(error_log_key)
                else:
                    logger.info(f"Skipping {data} no {error_log_key}/{filter_key} exists")
                    continue

                if error_str:
                    error.append(error_str)
            final_error_list = error
        else:
            logger.info(f"Either `error` or (`raw_data_dict` and `error_log_key`) has to be provided")
            return error_summary

        error_summary = error_summary_generation(final_error_list)
        for error in final_error_list:
            if error not in self.error_summary_list:
                self.error_summary_list.append(error)
        return error_summary

    def __filter_reason_and_get_qgenie_summary(self, failure_data):
        inline_css = """
            <style>
            table { table-layout: fixed; width: 100%; border-collapse: collapse; }

            /* Keep td as a table cell to avoid layout breakage */
            td.qgenie-summary { vertical-align: middle; padding: 8px; text-align: left; }

            /* Inner wrapper safely handles vertical centering */
            td.qgenie-summary > .cell-content {
                display: flex;
                align-items: center;
                width: 100%;
                white-space: normal;
                word-break: break-word;
            }

            td.qgenie-summary > .cell-content > ul {
                margin: 0;
                padding-left: 20px;
                list-style: disc;
                text-align: left;
            }
            </style>
            """

        unique_cluster_name, errors_list = set(), []
        for data in failure_data:
            if data["clusters"] not in unique_cluster_name:
                errors_list.append(data["reason"])
                unique_cluster_name.add(data["clusters"])
        error_summary = self.__get_error_qgenie_summary(error=errors_list)
        if error_summary:
            return (
                inline_css + '<td class="qgenie-summary">'
                '<div class="cell-content">'
                "<ul>"
                f"{error_summary}"
                "</ul>"
                "</div>"
                "</td>"
            )
        else:
            return inline_css + '<td class="qgenie-summary"><div class="cell-content">-</div></td>'

    def __type_runtime_based_error_data(self, data):
        runtime_type_regression_error_data = {}

        for _type, runtimes in (data or {}).items():
            if not isinstance(runtimes, dict):
                continue
            for runtime, clusters in runtimes.items():
                if not isinstance(clusters, dict):
                    continue

                type_bucket = runtime_type_regression_error_data.setdefault(_type, {})
                reasons_list = type_bucket.setdefault(runtime, [])

                for _, entries in clusters.items():
                    if not entries:
                        continue
                    first = entries[0]
                    if (
                        isinstance(first, dict)
                        and "reason" in first
                        and first["reason"] is not None
                        and first["cluster_class"] == "sdk_issue"
                    ):
                        reasons_list.append(first["reason"])

        self.runtime_type_regression_error_data = runtime_type_regression_error_data

    def filter_regression_data(self, data, _processing_type=None):
        if _processing_type is None or not data:
            return data

        if _processing_type == "type":
            updated_data = {}
            for _type, data_in_type in data.items():
                if _type.lower() not in self.types_to_filter_for_regression_analysis:
                    updated_data[_type] = data_in_type
                    self.__type_runtime_based_error_data(updated_data)
            return updated_data

        elif _processing_type == "model":
            updated_data = {}
            for model_name, model_data_list in data.items():
                updated_model_data_list = [
                    item
                    for item in model_data_list
                    if item.get("type", "").lower() not in self.types_to_filter_for_regression_analysis
                ]
                updated_data[model_name] = updated_model_data_list
            return updated_data

        return data

    def __create_detailed_type_regression_page(self, test_type, runtime, clustered_data):
        output_dir = os.path.join(
            self.__destination_folder, "regression_htmls", f"{self.__current_run_id}_{self.__prev_run_id}"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_name = f"{test_type}_{runtime}_clustered_regression_report.html"
        file_path = os.path.join(output_dir, file_name)
        total_failure_count = 0

        html_content = f"<html><head><title>Cluster Level Details</title>{REPORT_CSS}</head><body><div class='container'><h1>Cluster Level Details</h1>"

        for cluster_name, cluster_data in clustered_data.items():
            inner_html_content = ""
            inner_html_content += "<table border='1'><tr><th>TC UUID</th><th>Name</th><th>SoC</th><th>Runtime</th><th>Reason</th><th>Log</th></tr>"
            total_failure_count += len(cluster_data)
            for issue_grouped_cluster_dict in cluster_data:
                log_path = issue_grouped_cluster_dict.get("log_path", "N/A")
                log_link = (
                    f"<a href='https://aisw-hyd.qualcomm.com/fs/{log_path}'>Log</a>" if log_path != "N/A" else "N/A"
                )
                inner_html_content += f"<tr><td>{issue_grouped_cluster_dict.get('tc_uuid', 'N/A')}</td><td>{issue_grouped_cluster_dict.get('name', 'N/A')}</td><td>{issue_grouped_cluster_dict.get('soc_name', 'N/A')}</td><td>{issue_grouped_cluster_dict.get('runtime', 'N/A')}</td><td>{issue_grouped_cluster_dict.get('reason', 'N/A')}</td><td>{log_link}</td></tr>"

            html_content += f"<h2>{cluster_name} Details -- Total Occurences {len(cluster_data)}</h2>"
            html_content += inner_html_content
            html_content += "</table></body></html>"

        with open(file_path, "w") as f:
            f.write(html_content)

        return self.__server_prefix + file_path, total_failure_count

    def __build_type_based_regression_data(self, test_type, regression_data):
        data_dict = {}

        for runtime, runtime_dict in regression_data.items():
            runtime_html_path, failure_count = self.__create_detailed_type_regression_page(
                test_type, runtime, runtime_dict
            )
            data_dict[runtime] = {"html": runtime_html_path, "failure_count": failure_count}

        return data_dict

    def __create_detailed_soc_regression_page(self, soc_name, soc_regression_data):
        output_dir = os.path.join(
            self.__destination_folder, "regression_htmls", f"{self.__current_run_id}_{self.__prev_run_id}"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_name = f"{soc_name}_clustered_regression_report.html"
        file_path = os.path.join(output_dir, file_name)

        html_content = f"<html><head><title>SOC Level Details</title>{REPORT_CSS}</head><body><div class='container'><h1>SOC Level Details</h1>"

        html_content += "<table border='1'><tr><th>TC UUID</th><th>Name</th><th>Cluster Name</th><th>Runtime</th><th>Reason</th><th>Log</th></tr>"
        for soc_data in soc_regression_data:
            log_path = soc_data.get("log_path", "N/A")
            log_link = f"<a href='https://aisw-hyd.qualcomm.com/fs/{log_path}'>Log</a>" if log_path != "N/A" else "N/A"
            html_content += f"<tr><td>{soc_data.get('tc_uuid', 'N/A')}</td><td>{soc_data.get('name', 'N/A')}</td><td>{soc_data.get('clusters', 'N/A')}</td><td>{soc_data.get('runtime', 'N/A')}</td><td>{soc_data.get('reason', 'N/A')}</td><td>{log_link}</td></tr>"

        html_content += "</table></body></html>"
        with open(file_path, "w") as f:
            f.write(html_content)

        return self.__server_prefix + file_path

    def __create_detailed_runtime_regression_page(self, runtime, runtime_regression_data):
        output_dir = os.path.join(
            self.__destination_folder, "regression_htmls", f"{self.__current_run_id}_{self.__prev_run_id}"
        )
        file_name = f"{runtime}_clustered_regression_report.html"
        file_path = os.path.join(output_dir, file_name)

        html_content = f"<html><head><title>Runtime Level Details</title>{REPORT_CSS}</head><body><div class='container'><h1>Runtime Level Details</h1>"

        html_content += "<table border='1'><tr><th>TC UUID</th><th>Name</th><th>Cluster Name</th><th>Type</th><th>SOC Name</th><th>Reason</th><th>Log</th></tr>"
        for runtime_data in runtime_regression_data:
            log_path = runtime_data.get("log_path", "N/A")
            log_link = f"<a href='https://aisw-hyd.qualcomm.com/fs/{log_path}'>Log</a>" if log_path != "N/A" else "N/A"
            html_content += f"<tr><td>{runtime_data.get('tc_uuid', 'N/A')}</td><td>{runtime_data.get('name', 'N/A')}</td><td>{runtime_data.get('clusters', 'N/A')}</td><td>{runtime_data.get('type', 'N/A')}</td><td>{runtime_data.get('soc_name', 'N/A')}</td><td>{runtime_data.get('reason', 'N/A')}</td><td>{log_link}</td></tr>"

        html_content += "</table></body></html>"
        with open(file_path, "w") as f:
            f.write(html_content)

        return self.__server_prefix + file_path

    def __create_detailed_model_failure_regression_page(self, model_regression_data):
        output_dir = os.path.join(
            self.__destination_folder, "regression_htmls", f"{self.__current_run_id}_{self.__prev_run_id}"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_name = "model_failures_regression_report.html"
        file_path = os.path.join(output_dir, file_name)
        cluster_names_seen = set()
        html_content = f"<html><head><title>Model Level Failure Details</title>{REPORT_CSS}</head><body><div class='container'><h1>Model Level Failure Details</h1>"

        for model_name, model_data in model_regression_data.items():
            html_content += f"<h2>{model_name}</h2>"

            html_content += "<table border='1'><tr><th>TC UUID</th><th>Name</th><th>Type</th><th>Cluster Name</th><th>Type</th><th>SOC Name</th><th>Run Time</th><th>Error</th><th>Log</th></tr>"
            for data in model_data:
                cluster_name = data.get("clusters", "")
                log_path = data.get("log_path", "N/A")
                log_link = (
                    f"<a href='https://aisw-hyd.qualcomm.com/fs/{log_path}'>Log</a>" if log_path != "N/A" else "N/A"
                )
                html_content += f"<tr><td>{data.get('tc_uuid', 'N/A')}</td><td>{data.get('name', 'N/A')}</td><td>{data.get('type', 'N/A')}</td><td>{data.get('clusters', 'N/A')}</td><td>{data.get('type', 'N/A')}</td><td>{data.get('soc_name', 'N/A')}</td><td>{data.get('runtime', 'N/A')}</td><td>{data.get('reason', 'N/A')}</td><td>{log_link}</td></tr>"

                # store the errors for summary generation
                if cluster_name not in cluster_names_seen:
                    reason = (data.get("reason") or "").strip()
                    if reason:
                        self.model_regressed_errors_list.append(reason)

                cluster_names_seen.add(cluster_name)
            html_content += "</table></body></html>"

        with open(file_path, "w") as f:
            f.write(html_content)

        return self.__server_prefix + file_path

    @execution_timer
    def generate_regression_analysis_report(self, run_id_a, run_id_b, regression_data):
        self.__current_run_id = run_id_a
        self.__prev_run_id = run_id_b
        self.regression_data = regression_data
        self.gerrits_information = regression_data.get("gerrit_info", {})
        logger.info(f"Processing: {run_id_a}: {run_id_b}")

        if any("rc" in run_id.lower() for run_id in [run_id_a, run_id_b]):
            self._has_rc_in_runid = True

        type_based_regression_data = self.filter_regression_data(
            regression_data.get("type", {}), _processing_type="type"
        )
        model_based_regression_data = self.filter_regression_data(
            regression_data.get("model", {}), _processing_type="model"
        )
        if not regression_data or (isinstance(regression_data, dict) and regression_data.get("status", 500) != 200):
            logger.info(
                f"Empty regression data found between: {self.__current_run_id} and {self.__prev_run_id} \n Regression Data received: {regression_data}"
            )
            return ""
        head = f"<html><head><title>Regression Analysis</title>{REPORT_CSS}</head><body><div class='container'>"
        regression_html = ""

        logger.info("Building html for Type based Failures")
        regression_html += f"<h3>Type based Failures</h3>"
        type_based_data_dict = {}
        for test_type, types_dict in type_based_regression_data.items():
            type_based_data_dict[test_type] = self.__build_type_based_regression_data(test_type, types_dict)

        regression_html += f"<table><tr><th>Type/Runtime</th>"
        for headers in self.__auto_soc_column_header:
            regression_html += f"<th>{headers.capitalize()}</th>"
        regression_html += "</tr>"

        for _type, type_data in type_based_data_dict.items():
            regression_html += f"<tr><th>{_type.capitalize()}</th>"
            for runtime in self.__auto_soc_column_header:
                html_path, failure_count = type_data.get(runtime, {}).get("html"), type_data.get(runtime, {}).get(
                    "failure_count", 0
                )

                if html_path:
                    regression_html += f"<td><a href='{html_path}'>{failure_count}</a></td>"
                else:
                    regression_html += f"<td>-</td>"

            regression_html += "</tr>"
        regression_html += "</table>"

        logger.info("Building model failure report")
        regression_html += f"<h3>Failed Model Report</h3>"
        detailed_page_link = self.__create_detailed_model_failure_regression_page(model_based_regression_data)
        regression_html += f"<tr><td><a href='{detailed_page_link}'>Model Failure report</a></td><td> -- Total Models Failed {len(model_based_regression_data.keys())}</td></tr>"

        soc_regression_data = {}
        runtimes_regression_data = {}
        for _, types_dict in type_based_regression_data.items():
            for runtime, runtime_data in types_dict.items():
                for data_list in runtime_data.values():
                    for data in data_list:
                        soc_regression_data[data["soc_name"]] = soc_regression_data.get(data["soc_name"], []) + [data]
                        runtimes_regression_data[runtime] = runtimes_regression_data.get(runtime, []) + [data]

        logger.info("Building runtime based failure report")
        regression_html += f"<h3>Runtime based Failures</h3>"
        regression_html += "<table><tr><th>Runtime</th><th>Failure Count</th><th>Qgenie Summary</th>"
        if not regression_html.endswith("</tr>"):
            regression_html += "</tr>"
        for runtime, runtime_failure_data in runtimes_regression_data.items():
            detailed_page_link = self.__create_detailed_runtime_regression_page(runtime, runtime_failure_data)
            regression_html += (
                f"<tr><td>{runtime}</td><td><a href='{detailed_page_link}'>{len(runtime_failure_data)}</a></td>"
            )
            regression_html += self.__filter_reason_and_get_qgenie_summary(runtime_failure_data)
            regression_html += "</tr>"
        regression_html += "</table>"

        regression_html += f"<h3>SOC based Failures</h3>"
        regression_html += "<table><tr><th>Soc Name</th><th>Failure Count</th><th>Qgenie Summary</th></tr>"

        for soc_name, soc_failure_data in soc_regression_data.items():
            detailed_page_link = self.__create_detailed_soc_regression_page(soc_name, soc_failure_data)
            regression_html += (
                f"<tr><td>{soc_name}</td><td><a href='{detailed_page_link}'>{len(soc_failure_data)}</a></td>"
            )
            regression_html += self.__filter_reason_and_get_qgenie_summary(soc_failure_data)
            regression_html += "</tr>"
        regression_html += "</table>"

        executive_summary_html = f"<h2>Regression Analysis between {run_id_a} : {run_id_b}</h2>"
        executive_summary_html += "<h2>Executive Summary of Failures</h2>"
        executive_summary_html += generate_executive_summary(
            self.error_summary_list, self.model_regressed_errors_list, filter=False
        )
        final_html = head + executive_summary_html + regression_html
        regression_html_path = os.path.join(
            self.__destination_folder,
            "regression_htmls",
            f"{self.__current_run_id}_{self.__prev_run_id}",
            f"{self.__current_run_id}_{self.__prev_run_id}.html",
        )
        with open(regression_html_path, "w") as f:
            f.write(final_html)

        self.regression_html_path = regression_html_path
        return regression_html_path


class RegressionReport:
    """Orchestrates a regression comparison between two test runs.

    Args:
        tc_id_a: Previous (baseline) test-plan ID.
        tc_id_b: Current (candidate) test-plan ID.

    Example::

        report = RegressionReport(tc_id_a="QNN-001", tc_id_b="QNN-002")
        result = report.run()
    """

    def __init__(self, tc_id_a: str, tc_id_b: str) -> None:
        self.tc_id_a = tc_id_a
        self.tc_id_b = tc_id_b

    def run(self) -> dict:
        """Run regression analysis and return the cluster comparison dict.

        Returns:
            Dict with regression comparison data from the API.
        """
        return get_two_run_ids_cluster_info(self.tc_id_a, self.tc_id_b)
