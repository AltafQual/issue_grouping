import os
import re
from collections import defaultdict

import pandas as pd
from joblib import dump, load

from src.constants import CONSOLIDATED_REPORTS
from src.execution_timer_log import execution_timer
from src.get_prev_testplan_id import iterate_db_get_testplan
from src.logger import AppLogger
from src.regression_api_call import get_two_run_ids_cluster_info

logger = AppLogger().get_logger(__name__)


def filter_error_logs(error_logs_list):
    __error_filters = tuple(
        re.compile(error_regex, re.IGNORECASE)
        for error_regex in [
            r"\btimer\s+expired\b",  # matches "timer expired"
            r"\bmodel\s+not\s+found\b",  # matches "model not found"
            r"\bdevice\s+creation\b",  # matches "device creation",
            r"\binference\s+passed\s+but\s+output\s+not\s+generated\b",
            r"\bverifier\s+failed\b",
            r"\bdevice\s+not\s+found\b",
            r"\bmissing\s+shared\s+libraries\b",
        ]
    )
    filtered_error_logs = []
    for error_log in error_logs_list:
        if error_log and not any(pattern.search(error_log) for pattern in __error_filters):
            filtered_error_logs.append(error_log)

    print(f"Error logs in total: {len(error_logs_list)}, filtered error logs: {len(filtered_error_logs)}")
    return filtered_error_logs


@execution_timer
def generate_executive_summary(soc_errors_list, model_error_list):
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

    executive_summary = (
        inline_css + "<table class='exec-summary'>" "<tr><th>SOC/RunTime Summary</th><th>Model Summary</th></tr>"
    )
    from src.qgenie import cummilative_summary_generation

    soc_runtime_summary = cummilative_summary_generation(filter_error_logs(soc_errors_list))
    model_summary = cummilative_summary_generation(filter_error_logs(model_error_list))

    executive_summary += "<tr>" f"{ensure_td(soc_runtime_summary)}" f"{ensure_td(model_summary)}" "</tr>" "</table>"
    return executive_summary


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
        p_n_df, p_r_df, previous_testplan_id, previous_release_testplan_id = iterate_db_get_testplan(run_id)
        logger.info(f"previous testplan id: {previous_testplan_id} : Current testplan id: {run_id}")
        return previous_testplan_id

    def get_unqiue_runids(self, qairt_id):
        logger.info(f"Processing QAIRT Id: {qairt_id}")
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
            os.path.join(qairt_folder, functional_report_file), sheet_name=CONSOLIDATED_REPORTS.sheet_name
        )
        unique_test_ids = df["testplan_id"].unique().tolist()
        return unique_test_ids

    def get_regression_info_json(self, qairt_id):
        regression_information_dict = defaultdict(dict)
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
        self.__regression_data = None
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

    def __filter_regression_data(self, data, _processing_type=None):
        if _processing_type is None or not data:
            return data

        if _processing_type == "type":
            updated_data = {}
            for _type, data_in_type in data.items():
                if _type.lower() not in self.types_to_filter_for_regression_analysis:
                    updated_data[_type] = data_in_type
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

        html_content = f"<html><head><h1>Cluster Level Details</h1></head><body>"

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

        html_content = f"<html><head><h1>SOC Level Details</title></h1><body>"

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

        html_content = f"<html><head><h1>Runtime Level Details</title></h1><body>"

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
        html_content = f"<html><head><title>Model Level Failure Details</title></head><body>"

        for model_name, model_data in model_regression_data.items():
            html_content += f"<h2>{model_name}</h2>"

            html_content += "<table border='1'><tr><th>TC UUID</th><th>Name</th><th>Type</th><th>Cluster Name</th><th>Type</th><th>SOC Name</th><th>Run Time</th><th>Log</th></tr>"
            for data in model_data:
                cluster_name = data.get("clusters", "")
                log_path = data.get("log_path", "N/A")
                log_link = (
                    f"<a href='https://aisw-hyd.qualcomm.com/fs/{log_path}'>Log</a>" if log_path != "N/A" else "N/A"
                )
                html_content += f"<tr><td>{data.get('tc_uuid', 'N/A')}</td><td>{data.get('name', 'N/A')}</td><td>{data.get('type', 'N/A')}</td><td>{data.get('clusters', 'N/A')}</td><td>{data.get('type', 'N/A')}</td><td>{data.get('soc_name', 'N/A')}</td><td>{data.get('runtime', 'N/A')}</td><td>{log_link}</td></tr>"

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
        self.__regression_data = regression_data
        self.__gerrits_information = regression_data.get("gerrit_info", {})
        logger.info(f"Processing: {run_id_a}: {run_id_b}")

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
        head = "<html><head><style> body { font-family: Arial, sans-serif; color: #000000;} table {border-collapse: collapse;}th,td {border: 2px solid black;text-align: center;padding: 7px;}</style></head>"
        regression_html = f"<h2>Regression Analysis between {run_id_a} : {run_id_b}</h2>"

        logger.info("Building html for Type based Failures")
        regression_html += f"<h3>Type based Failures</h3>"
        type_based_data_dict = {}
        for test_type, runtimes_dict in type_based_regression_data.items():
            type_based_data_dict[test_type] = self.__build_type_based_regression_data(test_type, runtimes_dict)

        regression_html += f"<table><tr><th>Type/Runtime</th>"
        for headers in self.__auto_soc_column_header:
            regression_html += f"<th>{headers.upper()}</th>"
        regression_html += "</tr>"

        for _type, type_data in type_based_data_dict.items():
            regression_html += f"<tr><th>{_type.upper()}</th>"
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
        for _, runtimes_dict in type_based_regression_data.items():
            for runtime, runtime_data in runtimes_dict.items():
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

        regression_html += "</br></br>Regards,</br>AISW AUTO</body></html>"

        executive_summary_html = "<h2>Executive Summary of Failures</h2>"
        executive_summary_html += generate_executive_summary(self.error_summary_list, self.model_regressed_errors_list)
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
        self._regression_analysis_object = defaultdict(dict)
        self._regression_html_paths = defaultdict(str)

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

    def generate_each_run_id_regression_report(self, qairt_id):
        html_path_dict = {}
        unique_run_ids_for_qairt_id = self.consolidated_report_analysis.get_unqiue_runids(qairt_id)
        logger.info(f"Got all the run ids for qairt id: {qairt_id}: Run IDS: {unique_run_ids_for_qairt_id}")

        # load existing data if any
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
                and _id in self._regression_analysis_object
                and self._regression_html_paths.get(_id, "")
            ):
                logger.info(f"{_id} already processed skipping")
                continue

            prev_id = self.consolidated_report_analysis.build_prev_run_id(_id)
            logger.info(f"Processing: {_id}: {prev_id}")
            regression_json = get_two_run_ids_cluster_info(_id, prev_id)
            regression_analysis = RegressionAnalysisReport(qairt_id)
            html_path = regression_analysis.generate_regression_analysis_report(_id, prev_id, regression_json)
            logger.info(f"HTML for {_id}: {html_path}")

            self._regression_analysis_object[_id] = regression_analysis
            html_path_dict[_id] = html_path

        self._regression_html_paths = html_path_dict
        self.save_regression_analysis_objects(qairt_id)

    def __generated_regressed_gerrits_page(self, qairt_id, gerrits_information):
        output_dir = os.path.join(CONSOLIDATED_REPORTS.path, qairt_id, "regression_htmls")
        file_name = "gerrits_regression_report.html"
        file_path = os.path.join(output_dir, file_name)

        gerrits_merged_count = 0
        html_content = f"<html><head><h2>List of Gerrits merged in {qairt_id}</h2></head><body>"

        for repo_name, gerrit_data in gerrits_information.items():
            html_content += f"<h3>Repository Name: {repo_name}</h3>"
            html_content += "<table border='1'><tr><th>Gerrit Raised By</th><th>Email</th><th>Gerrit Link</th></tr>"
            unique_gerrits = set()
            for data in gerrit_data:
                if data["commit_url"] not in unique_gerrits:
                    commit_url = f"<a href='{data['commit_url']}' target='_blank'>Gerrit</a>"
                    html_content += f"<tr><td>{data['gerrit_raised_by'][0]['name']}</td><td>{data['gerrit_raised_by'][0]['email']}</td><td>{commit_url}</td></tr>"
                    unique_gerrits.add(data["commit_url"])
                    gerrits_merged_count += 1
            html_content += "</table>"

        html_content += "</body></html>"
        with open(file_path, "w") as f:
            f.write(html_content)

        return "https://aisw-hyd.qualcomm.com/fs/" + file_path, gerrits_merged_count

    def generate_gerrits_merged_report(self, qairt_id):
        def _iter_projects(sub_gerrit_data):
            for type_bucket in sub_gerrit_data.values():
                for project_list in type_bucket.values():
                    yield from project_list

        all_gerrits_data = defaultdict(list)
        for run in self._regression_analysis_object.values():
            sub_gerrit_data = run._RegressionAnalysisReport__gerrits_information
            for project in _iter_projects(sub_gerrit_data):
                repo = project.get("repository_name")
                if repo:
                    all_gerrits_data[repo].append(project)

        return self.__generated_regressed_gerrits_page(qairt_id, all_gerrits_data)

    def list_to_html_ul(self, items):
        """Convert a Python list into an HTML <ul><li>...</li></ul> block."""
        li_html = "".join(f"<li>{item}</li>" for item in items)
        return f"<ul>{li_html}</ul>"

    def generate_qairt_regression_report(self, qairt_id):
        qairt_regression_report = "<html><head><style> body { font-family: Arial, sans-serif; color: #000000;} table {border-collapse: collapse;}th,td {border: 2px solid black;text-align: center;padding: 7px;}</style></head>"
        qairt_regression_report += f"<h2>Regression Analysis Report ({qairt_id})</h2>"

        # BU Wise Executive Summary
        bu_wise_run_ids = defaultdict(list)
        for run_id in self._regression_analysis_object:
            bu_wise_run_ids[self.classify_run_id(run_id)].append(run_id)

        for bu, run_ids in bu_wise_run_ids.items():
            logger.info(f"Generating executing summary for bu: {bu}: {run_ids}")
            qairt_regression_report += f"<h3>{bu.upper()} Analysis Report</h3>"
            qairt_regression_report += self.list_to_html_ul(run_ids)

            soc_errors_list, model_error_list = [], []
            for run_id in run_ids:
                soc_errors_list.extend(self._regression_analysis_object[run_id].error_summary_list)
                model_error_list.extend(self._regression_analysis_object[run_id].model_regressed_errors_list)

            logger.info(f"Total errors: {len(soc_errors_list)}, Total model errors: {len(model_error_list)}")
            qairt_regression_report += generate_executive_summary(soc_errors_list, model_error_list)

        qairt_regression_report += "<h3>Run ID Wise Reports</h3>"
        non_regressed_run_ids = []
        regressed_run_ids = []
        for run_id, path in self._regression_html_paths.items():
            if not path:
                non_regressed_run_ids.append(run_id)
            else:
                regressed_run_ids.append(f"<a href='https://aisw-hyd.qualcomm.com/fs/{path}'>{run_id}</a>")

        qairt_regression_report += self.list_to_html_ul(regressed_run_ids)
        if non_regressed_run_ids:
            qairt_regression_report += "<h3> Non Regressed Run IDS </h3>"
            qairt_regression_report += self.list_to_html_ul(non_regressed_run_ids)

        qairt_regression_report += "<h3> Lists of Gerrits Merged </h3>"
        gerrits_merged_html_path, gerrits_count = self.generate_gerrits_merged_report(qairt_id)
        qairt_regression_report += self.list_to_html_ul(
            [f"<a href='{gerrits_merged_html_path}'>{gerrits_count} Gerrits Merged</a>"]
        )
        qairt_regression_report += "</br>Regards,</br>AISW AUTO</body></html>"
        qairt_regression_report_path = os.path.join(
            CONSOLIDATED_REPORTS.path,
            qairt_id,
            f"{qairt_id}.html",
        )
        with open(qairt_regression_report_path, "w") as f:
            f.write(qairt_regression_report)

        return qairt_regression_report_path

    def generate_final_summary_report(self, qairt_id=None):
        self.generate_each_run_id_regression_report(qairt_id)
        final_qairt_report_path = self.generate_qairt_regression_report(qairt_id)
        return final_qairt_report_path


if __name__ == "__main__":
    report_analysis = CombinedRegressionAnalysis(ConsolidatedReportAnalysis())
    qairt_id = "qaisw-v2.44.0.260112072337_193906_nightly"
    report_analysis.generate_final_summary_report(qairt_id)
