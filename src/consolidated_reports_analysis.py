import logging
import os
import re
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from html import escape

import pandas as pd
from joblib import dump, load

from src.constants import CONSOLIDATED_REPORTS
from src.execution_timer_log import execution_timer
from src.get_prev_testplan_id import iterate_db_get_testplan
from src.regression_api_call import get_two_run_ids_cluster_info

logger = logging.getLogger(__name__)

REPORT_CSS = """
<style>
    :root {
        --primary: #00629B; /* Qualcomm Blue approx or dark blue */
        --secondary: #3253DC;
        --bg-body: #f5f7fa;
        --bg-container: #ffffff;
        --text-main: #333333;
        --text-muted: #6c757d;
        --border-color: #e9ecef;
        --table-head-bg: #00629B;
        --table-head-text: #ffffff;
        --row-hover: #f1faff;
        --accent-danger: #dc3545;
        --accent-success: #28a745;
        --shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    body {
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        background-color: var(--bg-body);
        color: var(--text-main);
        margin: 0;
        padding: 20px;
        line-height: 1.6;
    }

    .container {
        max-width: 100%;
        margin: 0 auto;
        background-color: var(--bg-container);
        padding: 40px;
        border-radius: 8px;
        box-shadow: var(--shadow);
    }

    h1, h2, h3, h4 {
        color: var(--primary);
        font-weight: 600;
        margin-top: 1.5em;
        margin-bottom: 0.75em;
    }

    h1 { font-size: 2.2em; border-bottom: 3px solid var(--primary); padding-bottom: 10px; margin-top: 0; }
    h2 { font-size: 1.8em; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; }
    h3 { font-size: 1.4em; color: #444; }

    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 25px;
        background-color: white;
        border: 1px solid var(--border-color);
    }

    th, td {
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
        font-size: 0.95em;
        vertical-align: top;
        word-wrap: break-word;
        overflow-wrap: normal;
    }

    th {
        background-color: var(--table-head-bg);
        color: var(--table-head-text);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.85em;
    }

    tr:nth-child(even) { background-color: #f8f9fa; }
    tr:hover { background-color: var(--row-hover); }

    a { color: var(--secondary); text-decoration: none; font-weight: 500; }
    a:hover { text-decoration: underline; color: #003d73; }

    ul { margin: 0; padding-left: 20px; }
    li { margin-bottom: 5px; }

    /* Summary Box / Dashboard Styles */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }

    .card {
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: var(--shadow);
        border-top: 4px solid var(--primary);
        text-align: center;
        border: 1px solid var(--border-color);
    }

    .card h4 { margin: 0; font-size: 0.9em; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
    .card .count { font-size: 2.5em; font-weight: bold; color: var(--primary); margin: 10px 0; }

    .summary-section {
        background-color: #fff;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 20px;
        margin-bottom: 30px;
    }

    /* Helper Classes */
    .text-center { text-align: center; }

    /* Specific overrides */
    .qgenie-summary .cell-content, .gerrit-cell .cell-content {
        display: flex;
        flex-direction: column;
        width: 100%;
    }

    .exec-summary th { background-color: #2c3e50; }

    .footer {
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid var(--border-color);
        color: var(--text-muted);
        font-size: 0.9em;
        text-align: center;
    }
</style>
"""


def filter_error_logs(error_logs_list, custom_filter_list=None):
    filter_list = [
        r"\btimer\s+expired\b",
        r"\bmodel\s+not\s+found\b",
        r"\bdevice\s+creation\b",
        r"\binference\s+passed\s+but\s+output\s+not\s+generated\b",
        r"\bverifier\s+failed\b",
        r"\bdevice\s+not\s+found\b",
        r"\bmissing\s+shared\s+libraries\b",
        r"\bdevice(?:[_\s-])?unavailable\b",
        r"\bnot\s+found\b",
        r"\bno\s+space\s+left\s+on\s+device\b",
        r"\bno\s+such\s+file\s+or\s+directory",
        r"\bfilenotfounderror\b",
        r"\bqnn-net-run.exe\b",
    ]
    if custom_filter_list:
        filter_list.extend(custom_filter_list)
    __error_filters = tuple(re.compile(error_regex, re.IGNORECASE) for error_regex in filter_list)
    filtered_error_logs = []
    for error_log in error_logs_list:
        if error_log and not any(pattern.search(error_log) for pattern in __error_filters):
            filtered_error_logs.append(error_log)

    print(f"Error logs in total: {len(error_logs_list)}, filtered error logs: {len(filtered_error_logs)}")
    return filtered_error_logs


@execution_timer
def get_cummilative_sumary(errors, filter=True, custom_filter=None):
    from src.qgenie import cummilative_summary_generation

    if filter:
        print("Filter enabled, filtering logs")
        return cummilative_summary_generation(filter_error_logs(errors, custom_filter))
    else:
        return cummilative_summary_generation(errors)


@execution_timer
def generate_executive_summary(soc_errors_list=None, model_error_list=None, filter=True):
    def ensure_td(summary_html: str) -> str:
        if summary_html and "<td" in summary_html.lower():
            return summary_html
        return f"<td>{summary_html or '-'}</td>"

    inline_css = """
    <style>
    table.exec-summary {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
    }
    table.exec-summary th {
        text-align: left;
        padding: 8px;
        border-bottom: 1px solid #ccc;
        font-weight: normal;
    }
    table.exec-summary td {
        text-align: left;
        vertical-align: middle;
        padding: 12px;
    }
    table.exec-summary td li {
        margin-bottom: 2px;
        list-style: disc; 
        margin-left: 20px; 
    }
    </style>
    """

    if model_error_list:
        executive_summary = (
            inline_css + "<table class='exec-summary'>" "<tr><th>SOC/RunTime Summary</th><th>Model Summary</th></tr>"
        )
        soc_runtime_summary = get_cummilative_sumary(soc_errors_list, filter)
        model_summary = get_cummilative_sumary(model_error_list, filter)
        executive_summary += "<tr>" f"{ensure_td(soc_runtime_summary)}" f"{ensure_td(model_summary)}" "</tr>" "</table>"
    else:
        executive_summary = inline_css + "<table class='exec-summary'>" "<tr><th>SOC/RunTime Summary</th></tr>"
        soc_runtime_summary = get_cummilative_sumary(soc_errors_list, filter)
        executive_summary += "<tr>" f"{ensure_td(soc_runtime_summary)}" "</tr>" "</table>"

    return executive_summary


class OrderedDefaultDict(OrderedDict):
    def __init__(self, default_factory=None, *args, **kwargs):
        if default_factory is not None and not callable(default_factory):
            raise TypeError("default_factory must be callable or None")
        self.default_factory = default_factory
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value


class ConsolidatedReportAnalysis:
    def __init__(self):
        self.reports_folder_path = CONSOLIDATED_REPORTS.path

    def build_prev_run_id_prev_version_rc_id(
        self,
        run_id: str,
        prev_release_info: str = CONSOLIDATED_REPORTS.prev_release_info,
        prev_release_rc_number: str = CONSOLIDATED_REPORTS.prev_release_rc_number,
    ) -> str:
        # 1) Normalize any existing trailing RC (e.g., _RC3, _RC10) by stripping it.
        run_id = re.sub(r"(_RC\d+)$", "", run_id)

        # 2) Replace the version+commit segment:
        #    Pattern example: v2.44.0.260112072337_193906
        #    Regex: v<digits>.<digits>.<digits>.<digits>_<digits>
        version_commit_pattern = re.compile(r"v\d+\.\d+\.\d+\.\d+_\d+")
        if not version_commit_pattern.search(run_id):
            # If there is no recognizable version+commit pattern, we leave it unchanged and still append RC.
            replaced = run_id
        else:
            replaced = version_commit_pattern.sub(prev_release_info, run_id, count=1)

        # 3) Append the RC suffix (ensure single underscore separator).
        #    If the ID already ends with the same RC, avoid duplication.
        rc_suffix = f"_{prev_release_rc_number}"
        if not replaced.endswith(rc_suffix):
            replaced = f"{replaced}{rc_suffix}"

        return replaced

    def build_prev_run_id(self, run_id: str):
        _, _, previous_testplan_id, previous_release_testplan_id = iterate_db_get_testplan(run_id)
        print(
            f"previous testplan id: {previous_testplan_id} : Current testplan id: {run_id}: previous release id: {previous_release_testplan_id}"
        )
        return previous_testplan_id if previous_testplan_id else previous_release_testplan_id

    def get_unqiue_runids(self, qairt_id):
        try:
            print(f"Processing QAIRT Id: {qairt_id}")
            qairt_folder = os.path.join(self.reports_folder_path, qairt_id)
            functional_report_file = None
            for file in os.listdir(qairt_folder):
                if file.startswith("Functional"):
                    functional_report_file = file
                    break

            if functional_report_file is None:
                logger.warning(f"Not able to find any Functional report for: {qairt_id}")
                return []

            df = pd.read_excel(
                os.path.join(qairt_folder, functional_report_file),
                sheet_name=CONSOLIDATED_REPORTS.sheet_name,
                engine="openpyxl",
            )
            unique_test_ids = df["testplan_id"].unique().tolist()
        except Exception as e:
            logger.exception(f"Exception Occured while processing: {qairt_id} {e}")
            return []
        return unique_test_ids

    def get_regression_info_json(self, qairt_id):
        regression_information_dict = OrderedDefaultDict(dict)
        unique_test_ids = self.get_unqiue_runids(qairt_id)
        for test_id in unique_test_ids:
            old_release_id = self.build_prev_run_id(test_id)
            regression_information_dict[test_id] = get_two_run_ids_cluster_info(test_id, old_release_id)

        return regression_information_dict


class RegressionAnalysisReport:
    def __init__(self, qairt_id: str):
        self.model_regressed_errors_list, self.error_summary_list = [], []
        self.types_to_filter_for_regression_analysis = ["bm_regression"]
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
                    print(f"Skipping {data} no {error_log_key}/{filter_key} exists")
                    continue

                if error_str:
                    error.append(error_str)
            final_error_list = error
        else:
            print(f"Either `error` or (`raw_data_dict` and `error_log_key`) has to be provided")
            return error_summary

        from src.qgenie import error_summary_generation

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
                    if isinstance(first, dict) and "reason" in first and first["reason"] is not None:
                        reasons_list.append(first["reason"])

        self.runtime_type_regression_error_data = runtime_type_regression_error_data

    def __filter_regression_data(self, data, _processing_type=None):
        if _processing_type is None or not data:
            return data

        if _processing_type == "type":
            updated_data = {}
            for _type, data_in_type in data.items():
                if _type.lower() not in self.types_to_filter_for_regression_analysis:
                    updated_data[_type] = data_in_type
            self.__type_runtime_based_error_data(data)
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
        print(f"Processing: {run_id_a}: {run_id_b}")
        if any("rc" in run_id.lower() for run_id in [run_id_a, run_id_b]):
            self._has_rc_in_runid = True

        type_based_regression_data = self.__filter_regression_data(
            regression_data.get("type", {}), _processing_type="type"
        )
        model_based_regression_data = self.__filter_regression_data(
            regression_data.get("model", {}), _processing_type="model"
        )
        if not regression_data or (isinstance(regression_data, dict) and regression_data.get("status", 500) != 200):
            print(
                f"Empty regression data found between: {self.__current_run_id} and {self.__prev_run_id} \n Regression Data received: {regression_data}"
            )
            return ""
        head = f"<html><head><title>Regression Analysis</title>{REPORT_CSS}</head><body><div class='container'>"
        regression_html = f"<h2>Regression Analysis between {run_id_a} : {run_id_b}</h2>"

        print("Building html for Type based Failures")
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

        print("Building model failure report")
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

        print("Building runtime based failure report")
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

        executive_summary_html = "<h2>Executive Summary of Failures</h2>"
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


class CombinedRegressionAnalysis:
    def __init__(self, consolidated_report_analysis: ConsolidatedReportAnalysis):
        self.consolidated_report_analysis = consolidated_report_analysis
        self._regression_analysis_object = OrderedDefaultDict(dict)
        self._regression_html_paths = OrderedDefaultDict(str)
        self.__processed_run_id = False
        self.combined_soc_errors_list = []
        self.combined_model_errros_list = []
        self.combined_type_runtime_wise_errors_dict = {}
        self._qairt_id = None

    def _regex(self, pattern: str):
        rx = re.compile(pattern, re.IGNORECASE)
        return lambda s: rx.search(s) is not None

    def classify_run_id(self, run_id: str, rules=None):
        """
        Classify a run_id into one of the labels based on ordered rules.
        Returns the first matching label or None if nothing matches.

        Default precedence:
            1) pt     -> android/iot/xr
            2) win    -> windows/compute
            3) auto   -> auto
            4) llm    -> GenAI
        """
        if rules is None:
            rules = [
                ("Compute", self._regex(r"(?:^|[^a-z0-9])win(?:dows)?(?:[^a-z0-9]|$)")),
                ("auto", self._regex(r"(?:^|[^a-z0-9])auto(?:[^a-z0-9]|$)")),
                ("GenAI", self._regex(r"(?:^|[^a-z0-9])llm(?:[^a-z0-9]|$)")),
                ("Mobile/IOT/XR", self._regex(r"(?:^|[^a-z0-9])pt(?:[^a-z0-9]|$)")),
            ]

        for label, predicate in rules:
            if predicate(run_id):
                return label
        return "Unknown"

    def save_regression_analysis_objects(self, qairt_id):
        regression_artifacts_path = os.path.join(
            self.consolidated_report_analysis.reports_folder_path,
            qairt_id,
            "regression_artifacts",
            f"{qairt_id}_regression_analysis_object.joblib",
        )
        regression_html_path = os.path.join(
            self.consolidated_report_analysis.reports_folder_path,
            qairt_id,
            "regression_artifacts",
            f"{qairt_id}_regression_html_paths.joblib",
        )
        os.makedirs(os.path.dirname(regression_artifacts_path), exist_ok=True)
        os.makedirs(os.path.dirname(regression_html_path), exist_ok=True)

        try:
            dump(self._regression_analysis_object, regression_artifacts_path, compress=0, protocol=5)
            dump(self._regression_html_paths, regression_html_path, compress=0, protocol=5)
        except Exception as e:
            logger.exception(f"Error occured while saving objects: {e}")

    def load_regression_analysis_objects(self, qairt_id):
        regression_artifacts_path = os.path.join(
            self.consolidated_report_analysis.reports_folder_path,
            qairt_id,
            "regression_artifacts",
            f"{qairt_id}_regression_analysis_object.joblib",
        )
        regression_html_path = os.path.join(
            self.consolidated_report_analysis.reports_folder_path,
            qairt_id,
            "regression_artifacts",
            f"{qairt_id}_regression_html_paths.joblib",
        )
        if not os.path.isfile(regression_artifacts_path):
            return
        try:
            self._regression_analysis_object = load(regression_artifacts_path, mmap_mode="r")
            self._regression_html_paths = load(regression_html_path, mmap_mode="r")
        except Exception as e:
            logger.exception(f"Error occured while loading objects: {e}")

    def merge_two_jsons(self, dst, src, dedupe=False):
        for key, src_val in src.items():
            if key not in dst:
                dst[key] = deepcopy(src_val)
                continue

            dst_val = dst[key]

            if isinstance(dst_val, dict) and isinstance(src_val, dict):
                self.merge_two_jsons(dst_val, src_val, dedupe=dedupe)
            elif isinstance(dst_val, list) and isinstance(src_val, list):
                dst_val.extend(src_val)
                if dedupe:
                    seen = set()
                    unique = []
                    for item in dst_val:
                        marker = item if isinstance(item, (str, int, float, bool, type(None))) else repr(item)
                        if marker not in seen:
                            seen.add(marker)
                            unique.append(item)
                    dst[key] = unique
            else:
                dst[key] = deepcopy(src_val)

        return dst

    def generate_each_run_id_regression_report(self, qairt_id):
        unique_run_ids_for_qairt_id = self.consolidated_report_analysis.get_unqiue_runids(qairt_id)
        if not unique_run_ids_for_qairt_id:
            logger.error(f"No run ids found for {qairt_id}")
            return unique_run_ids_for_qairt_id

        print(f"Got all the run ids for qairt id: {qairt_id}: Run IDS: {unique_run_ids_for_qairt_id}")
        self.load_regression_analysis_objects(qairt_id)
        for _id in unique_run_ids_for_qairt_id:
            """
            Both the object and html should be available for the run_id to skip processing
            HTML creation might have failed because the data might have not been purged yet.
            So, in that case when rerun happens we only generated htmls for the one which have not been previously
            """
            if (
                self._regression_analysis_object
                and self._regression_html_paths
                and self._regression_analysis_object.get(_id)
                and self._regression_html_paths.get(_id)
            ):
                print(f"{_id} already processed skipping")
                continue

            prev_id = self.consolidated_report_analysis.build_prev_run_id(_id)
            print(f"Processing: {_id}: {prev_id}")
            regression_json = get_two_run_ids_cluster_info(_id, prev_id, force=True)
            regression_analysis = RegressionAnalysisReport(qairt_id)
            html_path = regression_analysis.generate_regression_analysis_report(_id, prev_id, regression_json)
            print(f"HTML for {_id}: {html_path}")

            self._regression_analysis_object[_id] = regression_analysis
            self._regression_html_paths[_id] = html_path
            if html_path:
                self.__processed_run_id = True

        # only in case of new processing save the regression objects
        if self.__processed_run_id:
            self.save_regression_analysis_objects(qairt_id)

    def __generated_regressed_gerrits_page(self, qairt_id, gerrits_information):
        output_dir = os.path.join(CONSOLIDATED_REPORTS.path, qairt_id, "regression_htmls")
        file_name = "gerrits_regression_report.html"
        file_path = os.path.join(output_dir, file_name)

        html_content = f"<html><head><title>Gerrits Merged</title>{REPORT_CSS}</head><body><div class='container'><h2>List of Gerrits merged in {qairt_id}</h2>"
        project_wise_gerrits = OrderedDefaultDict(list)
        unique_gerrits = set()
        for _, runtime_based_gerrit_data in gerrits_information.items():
            for _, backend_gerrit_data in runtime_based_gerrit_data.items():
                for gerrit_data in backend_gerrit_data:
                    if gerrit_data["commit_url"] not in unique_gerrits:
                        project_wise_gerrits[gerrit_data["repository_name"]].append(gerrit_data)
                        unique_gerrits.add(gerrit_data["commit_url"].lower())

        for repo_name, gerrits_data in project_wise_gerrits.items():
            html_content += f"<h3>Repository Name: {repo_name}</h3>"
            for data in gerrits_data:
                html_content += "<table border='1'><tr><th>Gerrit Raised By</th><th>Email</th><th>Commit Message</th><th>Gerrit Link</th></tr>"
                commit_url = f"<a href='{data['commit_url']}' target='_blank'>Gerrit</a>"
                html_content += f"<tr><td>{data['gerrit_raised_by'][0]['name']}</td><td>{data['gerrit_raised_by'][0]['email']}</td><td>{data['commit_message']}</td><td>{commit_url}</td></tr>"
            html_content += "</table>"

        html_content += "</body></html>"
        with open(file_path, "w") as f:
            f.write(html_content)

        return "https://aisw-hyd.qualcomm.com/fs/" + file_path

    def _get_gerrits_data(self, runtime_first=False):
        gerrits_data = None
        for run in self._regression_analysis_object.values():
            if gerrits_data:
                break

            if run and run.gerrits_information and not run._has_rc_in_runid:
                gerrits_data = run.gerrits_information

        if runtime_first:
            gerrits_data = self.pivot_type_to_runtime(gerrits_data, flatten=True)

        return gerrits_data

    def generate_gerrits_merged_report(self):
        return self.__generated_regressed_gerrits_page(self._qairt_id, self._get_gerrits_data())

    def list_to_html_ul(self, items):
        """Convert a Python list into an HTML <ul><li>...</li></ul> block."""
        li_html = "".join(f"<li>{item}</li>" for item in items)
        return f"<ul>{li_html}</ul>"

    def get_runtime_based_gerrit_row(self, runtime, gerrit_data, rows_span=0):
        print(f"All types of gerrits merged: {gerrit_data.keys()}, runtime: {runtime}")
        if runtime == "tools":
            runtime_gerrits = list(gerrit_data.get("quantizer") or []) + list(gerrit_data.get("converter") or [])
        else:
            runtime_gerrits = list(gerrit_data.get(runtime) or [])

        seen_jiras = OrderedDefaultDict(set)
        items_html = []
        repository_based_filteration = OrderedDefaultDict(list)

        # Group by repository
        for gerrit_info in runtime_gerrits:
            repo = (gerrit_info.get("repository_name") or "").strip()
            repository_based_filteration[repo].append(gerrit_info)

        for repo_name, repo_data in repository_based_filteration.items():
            repo_key = (repo_name or "").lower()
            repo_esc = escape(repo_name or "", quote=True)

            inner_li = []
            for data in repo_data:
                url = (data.get("commit_url") or "").strip()
                msg = (data.get("commit_message") or "").strip()

                # Deduplicate by message per repo (case-insensitive)
                key = msg.lower()
                if key and key not in seen_jiras[repo_key]:
                    msg_esc = escape(msg, quote=True)
                    url_esc = escape(url, quote=True) if url else ""
                    if url_esc:
                        inner_li.append(f'<li><a href="{url_esc}">{msg_esc}</a></li>')
                    else:
                        inner_li.append(f"<li>{msg_esc}</li>")
                    seen_jiras[repo_key].add(key)

            if inner_li:
                # One repo section wrapped in <li>
                items_html.append(f"<li><b>{repo_esc}</b><ul>{''.join(inner_li)}</ul></li>")

        base_html = f'<td rowspan="{rows_span}" class="gerrit-cell"><div class="cell-content">'

        if not items_html:
            # No gerrits
            return base_html + "-</div></td>"

        # Wrap all repo sections under a single UL
        return base_html + f"<ul>{''.join(items_html)}</ul>" + "</div>" + "</td>"

    def extract_converter_quantizer_logs(self, data):
        """
        Given a dict shaped like:
            {
            <type>: {
                <runtime>: [errors...]
            },
            ...
            }

        - Extracts the 'converter' and 'quantizer' type entries (case-insensitive).
        - Removes them from 'data' IN-PLACE.
        - Returns:
            ( extracted_tools_dict, tools_row_name )

        where:
            extracted_tools_dict is a dict that may contain keys:
            - 'converter': { <runtime>: [errors...] }
            - 'quantizer': { <runtime>: [errors...] }

        If neither exists, returns {}, data unchanged
        """

        targets_present = {"converter": None, "quantizer": None}
        for k in list(data.keys()):
            kl = k.lower()
            if kl in targets_present and targets_present[kl] is None:
                targets_present[kl] = k

        extracted = {}
        for target_name, actual_key in targets_present.items():
            if actual_key is not None:
                value = data.pop(actual_key, None)  # remove from 'data' in-place
                if isinstance(value, dict) and value:
                    # Store under target key names: 'converter' / 'quantizer'
                    extracted[target_name] = value

        return self.pivot_type_to_runtime(extracted)

    def pivot_type_to_runtime(self, data: dict, flatten=False) -> dict:
        """
        Convert a nested mapping of the form:
            { type: { runtime: [errors] } }
        into:
            { runtime: { type: [errors] } }

        - Preserves only the (type, runtime) pairs present in 'data'.
        - Assumes leaf values are lists (e.g., list of errors).
        - Ignores non-dict inner nodes gracefully.

        Example:
            {
            "benchmark": {"cpu": [...], "gpu": [...]},
            "inference": {"htp": [...], "gpu": [...]}
            }
            ->
            {
            "cpu": {"benchmark": [...]},
            "gpu": {"benchmark": [...], "inference": [...]},
            "htp": {"inference": [...]}
            }
        """
        if not isinstance(data, Mapping):
            return {}

        out: dict = {}
        for type_key, runtimes in data.items():
            if not isinstance(runtimes, Mapping):
                # Skip malformed nodes (expecting dict of runtime -> list)
                continue

            for runtime_key, errors in runtimes.items():
                if flatten:
                    out.setdefault(runtime_key, [])
                    out[runtime_key] = errors
                else:
                    out.setdefault(runtime_key, {})
                    out[runtime_key][type_key] = errors

        return out

    def generate_bu_regression_report(self, qairt_id):
        qairt_regression_report_path = os.path.join(
            CONSOLIDATED_REPORTS.path,
            qairt_id,
            "regression_htmls",
            f"BU_{qairt_id}.html",
        )

        qairt_regression_report = (
            f"<html><head><title>BU Wise Analysis</title>{REPORT_CSS}</head><body><div class='container'>"
        )
        qairt_regression_report += f"<h2>BU Wise Analysis Report ({qairt_id})</h2>"

        # BU Wise Executive Summary
        bu_wise_run_ids = OrderedDefaultDict(list)
        for run_id in self._regression_analysis_object:
            bu_wise_run_ids[self.classify_run_id(run_id)].append(run_id)

        for bu, run_ids in bu_wise_run_ids.items():
            print(f"Generating executing summary for bu: {bu}: {run_ids}")
            qairt_regression_report += f"<h3>{bu.upper()} Analysis Report</h3>"
            updated_run_ids = []
            for run_id in run_ids:
                if self._regression_html_paths.get(run_id):
                    updated_run_ids.append(
                        f"<a href='https://aisw-hyd.qualcomm.com/fs/{self._regression_html_paths[run_id]}'>{run_id}</a>"
                    )
                else:
                    updated_run_ids.append(run_id)
            qairt_regression_report += self.list_to_html_ul(updated_run_ids)

            soc_errors_list = []
            for run_id in run_ids:
                if self._regression_analysis_object[run_id]:
                    soc_errors_list.extend(self._regression_analysis_object[run_id].error_summary_list)

            self.combined_soc_errors_list.extend(soc_errors_list)

            print(f"Total errors: {len(soc_errors_list)}")
            qairt_regression_report += generate_executive_summary(soc_errors_list)

        qairt_regression_report += "</div></body></html>"
        with open(qairt_regression_report_path, "w") as f:
            f.write(qairt_regression_report)
        return qairt_regression_report_path

    def add_gerrit_details_to_report(self, qairt_html_report):
        gerrits_merged_html_path, gerrits_count = self.generate_gerrits_merged_report()
        if gerrits_count:
            qairt_html_report += "<h3> Lists of Gerrits Merged </h3>"
            qairt_html_report += self.list_to_html_ul(
                [f"<a href='{gerrits_merged_html_path}'>{gerrits_count} Gerrits Merged</a>"]
            )
        else:
            qairt_html_report += "<h3> No Gerrits Merged !! </h3>"

        return qairt_html_report

    def generate_qairt_regression_report(self, qairt_id):
        qairt_regression_report_path = os.path.join(
            CONSOLIDATED_REPORTS.path,
            qairt_id,
            f"{qairt_id}.html",
        )

        # if no run id is processed return the previous path with any processing
        if not self.__processed_run_id:
            return qairt_regression_report_path

        qairt_regression_report = (
            f"<html><head><title>Regression Report</title>{REPORT_CSS}</head><body><div class='container'>"
        )
        qairt_regression_report += f"<h2>QAIRT Analysis Report ({qairt_id})</h2>"

        prev_run_id = None
        combined_runtime_type_json_data = None
        for run_id in self._regression_analysis_object:
            if not self._regression_analysis_object[run_id].runtime_type_regression_error_data:
                continue
            if prev_run_id is None:
                prev_run_id = run_id
                continue

            if not combined_runtime_type_json_data:
                combined_runtime_type_json_data = self.merge_two_jsons(
                    self._regression_analysis_object[prev_run_id].runtime_type_regression_error_data,
                    self._regression_analysis_object[run_id].runtime_type_regression_error_data,
                )
            else:
                combined_runtime_type_json_data = self.merge_two_jsons(
                    combined_runtime_type_json_data,
                    self._regression_analysis_object[run_id].runtime_type_regression_error_data,
                )
        if combined_runtime_type_json_data is None:
            combined_runtime_type_json_data = self._regression_analysis_object[
                prev_run_id
            ].runtime_type_regression_error_data

        converter_quantizer_dict = self.extract_converter_quantizer_logs(combined_runtime_type_json_data)
        combined_runtime_type_json_data = self.pivot_type_to_runtime(combined_runtime_type_json_data)
        list_of_summay_to_avoid = ["no logs to provide", "no logs"]

        # build qairt report
        qairt_regression_report += (
            "<table border='1'><tr><th>Runtime/Tools</th><th>Type</th><th>Summary</th><th>Gerrits Merged</th></tr>"
        )
        gerrits_data = self._get_gerrits_data(runtime_first=True)

        # add tools row
        if converter_quantizer_dict:
            first_row = True
            for _, runtimes_dict in converter_quantizer_dict.items():
                runtimes = runtimes_dict.keys()
                summaries_list = [get_cummilative_sumary(runtimes_dict[runtime]) for runtime in runtimes]

                summary_idx_to_avoid = []
                for idx, summary in enumerate(summaries_list):
                    if any(summay_to_avoid in summary.lower() for summay_to_avoid in list_of_summay_to_avoid):
                        summary_idx_to_avoid.append(idx)

                updated_runtimes = []
                for idx, runtime in enumerate(runtimes):
                    if idx not in summary_idx_to_avoid:
                        updated_runtimes.append((runtime, idx))

                runtimes = updated_runtimes
                rowspan = len(runtimes)

                if rowspan:
                    for runtime, idx in runtimes:
                        summary_html = f"<ul>{summaries_list[idx]}</ul>"
                        if first_row:
                            gerrit_cell = self.get_runtime_based_gerrit_row("tools", gerrits_data, rowspan)
                            qairt_regression_report += (
                                f"<tr>"
                                f"<td rowspan='{rowspan}'>Tools</td>"
                                f"<td>{runtime.capitalize()}</td>"
                                f"<td><ul>{summary_html}</ul></td>"
                                f"{gerrit_cell}"
                                f"</tr>"
                            )
                            first_row = False
                        else:
                            qairt_regression_report += (
                                f"<tr>" f"<td>{runtime.capitalize()}</td>" f"<td><ul>{summary_html}</ul></td>" f"</tr>"
                            )
        core_data = []
        # process CPU seprately
        if "cpu" in combined_runtime_type_json_data:
            current_runtime = "cpu"
            types_dict = combined_runtime_type_json_data[current_runtime]

            # adding graph prepare section if exists
            if "savecontext" in types_dict or "graph_prepare" in types_dict:
                graph_prepare_row_data = []
                savecontext_data = types_dict.get("savecontext") or ""
                gprepare_data = types_dict.get("graph_prepare") or ""

                if savecontext_data:
                    savecontext_summary = get_cummilative_sumary(savecontext_data)
                    graph_prepare_row_data.append(("savecontext", savecontext_summary))
                    del types_dict["savecontext"]

                if gprepare_data:
                    gprepare_summary = get_cummilative_sumary(gprepare_data)
                    graph_prepare_row_data.append(("graph prepare", gprepare_summary))
                    del types_dict["graph_prepare"]

                for idx, data in enumerate(graph_prepare_row_data):
                    if any(summary_to_avoid in data[1].lower() for summary_to_avoid in list_of_summay_to_avoid):
                        graph_prepare_row_data.pop(idx)

                rowspan = len(graph_prepare_row_data)
                if rowspan:
                    first_runtime = graph_prepare_row_data[0][0]
                    first_summary = graph_prepare_row_data[0][1]
                    gerrit_cell = self.get_runtime_based_gerrit_row("htp", gerrits_data, rowspan)
                    qairt_regression_report += (
                        f"<tr>"
                        f"<td rowspan='{rowspan}'>Graph Prepare</td>"
                        f"<td>{first_runtime}</td>"
                        f"<td><ul>{first_summary}</ul></td>"
                        f"{gerrit_cell}"
                        f"</tr>"
                    )
                    for runtime, summary in runtimes[1:]:
                        qairt_regression_report += f"<tr>" f"<td>{runtime}</td>" f"<td><ul>{summary}</ul></td>" f"</tr>"

            types_to_process_cpu = ["inference", "verifier", "bm_regression"]
            all_cpu_types = list(types_dict.keys())
            all_cpu_summaries = [get_cummilative_sumary(types_dict[_type]) for _type in all_cpu_types]

            summary_idx_to_avoid = []
            for idx, summary in enumerate(all_cpu_summaries):
                if not any(summay_to_avoid in summary.lower() for summay_to_avoid in list_of_summay_to_avoid):
                    if all_cpu_types[idx] not in types_to_process_cpu and all_cpu_types[idx] != "benchmark":
                        core_data.append((all_cpu_types[idx], summary))
                        summary_idx_to_avoid.append(idx)
                else:
                    summary_idx_to_avoid.append(idx)

            updated_cpu_types = []
            for idx, runtime in enumerate(all_cpu_types):
                if idx not in summary_idx_to_avoid:
                    updated_cpu_types.append((runtime, idx))

            runtimes = updated_cpu_types
            rowspan = len(runtimes)
            if rowspan:
                first_runtime = runtimes[0][0]
                first_summary = all_cpu_summaries[runtimes[0][1]]
                gerrit_cell = self.get_runtime_based_gerrit_row(current_runtime, gerrits_data, rowspan)
                qairt_regression_report += (
                    f"<tr>"
                    f"<td rowspan='{rowspan}'>cpu</td>"
                    f"<td>{first_runtime}</td>"
                    f"<td><ul>{first_summary}</ul></td>"
                    f"{gerrit_cell}"
                    f"</tr>"
                )
                for runtime, idx in runtimes[1:]:
                    summary = all_cpu_summaries[idx]
                    qairt_regression_report += f"<tr>" f"<td>{runtime}</td>" f"<td><ul>{summary}</ul></td>" f"</tr>"
            del combined_runtime_type_json_data["cpu"]

        # add all rows apart from tools
        for current_runtime, types_dict in combined_runtime_type_json_data.items():
            runtimes = list(types_dict.keys())
            cpu_summaries_list = [get_cummilative_sumary(types_dict[runtime]) for runtime in runtimes]
            summary_idx_to_avoid = []
            for idx, summary in enumerate(cpu_summaries_list):
                if any(summay_to_avoid in summary.lower() for summay_to_avoid in list_of_summay_to_avoid):
                    summary_idx_to_avoid.append(idx)
            updated_runtimes = []
            for idx, runtime in enumerate(runtimes):
                if idx not in summary_idx_to_avoid and runtime != "benchmark":
                    updated_runtimes.append((runtime, idx))

            runtimes = updated_runtimes
            rowspan = len(runtimes)
            if rowspan:
                first_runtime = runtimes[0][0]
                first_summary = cpu_summaries_list[runtimes[0][1]]
                gerrit_cell = self.get_runtime_based_gerrit_row(current_runtime, gerrits_data, rowspan)
                qairt_regression_report += (
                    f"<tr>"
                    f"<td rowspan='{rowspan}'>{current_runtime}</td>"
                    f"<td>{first_runtime}</td>"
                    f"<td><ul>{first_summary}</ul></td>"
                    f"{gerrit_cell}"
                    f"</tr>"
                )
                for runtime, idx in runtimes[1:]:
                    summary = cpu_summaries_list[idx]
                    qairt_regression_report += f"<tr>" f"<td>{runtime}</td>" f"<td><ul>{summary}</ul></td>" f"</tr>"

        gerrit_cell = self.generate_gerrits_merged_report()
        gerrit_cell = f"<a href='{gerrit_cell}'>All Merged Gerrits</a>"
        if core_data:
            rowspan = len(core_data)
            if rowspan:
                first_runtime = core_data[0][0]
                first_summary = core_data[0][1]
                qairt_regression_report += (
                    f"<tr>"
                    f"<td rowspan='{rowspan}'>Core</td>"
                    f"<td>{first_runtime}</td>"
                    f"<td><ul>{first_summary}</ul></td>"
                    f"<td>{gerrit_cell}</td>"
                    f"</tr>"
                )
                for runtime, summary in core_data[1:]:
                    qairt_regression_report += f"<tr>" f"<td>{runtime}</td>" f"<td><ul>{summary}</ul></td>" f"</tr>"
        else:
            qairt_regression_report += (
                f"<tr>" f"<td>Core</td>" f"<td>-</td>" f"<td>-</td>" f"<td>{gerrit_cell}</td>" f"</tr>"
            )
        qairt_regression_report += "</table>"

        bu_summary_path = self.generate_bu_regression_report(qairt_id)
        qairt_regression_report += "<h3> BU Summary Page </h3>"
        qairt_regression_report += self.list_to_html_ul(
            [f"<a href='https://aisw-hyd.qualcomm.com/fs/{bu_summary_path}' target='_blank'>BU Summary Page</a>"]
        )

        qairt_regression_report += "</div></body></html>"
        with open(qairt_regression_report_path, "w") as f:
            f.write(qairt_regression_report)
        return qairt_regression_report_path

    def generate_final_summary_report(self, qairt_id=None):
        self._qairt_id = qairt_id
        self.generate_each_run_id_regression_report(qairt_id)
        return self.generate_qairt_regression_report(qairt_id)


def run_report_generation_for_all_qairt_ids():
    for qairt_id in sorted(os.listdir(CONSOLIDATED_REPORTS.path), reverse=True):
        if qairt_id.startswith("qaisw"):
            try:
                report_analysis = CombinedRegressionAnalysis(ConsolidatedReportAnalysis())
                report_analysis.generate_final_summary_report(qairt_id)
                print(f"{qairt_id} Successfully processed !!")
            except Exception as e:
                continue
        else:
            print(f"Non Qaisw folder found: {qairt_id}... Skipping !!!")


if __name__ == "__main__":
    report_analysis = CombinedRegressionAnalysis(ConsolidatedReportAnalysis())
    qairt_id = "qaisw-v2.44.0.260112072337_193906_nightly"
    report_analysis.generate_final_summary_report(qairt_id)
