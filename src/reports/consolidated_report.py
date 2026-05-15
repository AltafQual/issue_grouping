"""Consolidated regression report orchestrator.

Provides :class:`ConsolidatedReportAnalysis` (QAIRT run-ID data loader) and
:class:`CombinedRegressionAnalysis` (multi-run-ID orchestrator), plus the
module-level helpers :func:`should_process_id` and
:func:`run_report_generation_for_all_qairt_ids`.

Layering
--------
Imports from ``src.constants``, ``src.logger``, ``src.utils.run_id_utils``,
``src.reports.html_renderer``, ``src.reports.kpi_calculator``, and
``src.reports.regression_report``.  No imports from
``src.consolidated_reports_analysis`` or ``src.regression_api_call``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from copy import deepcopy
from datetime import date
from typing import Any, Dict

import pandas as pd
from dateutil.relativedelta import relativedelta
from joblib import dump, load

from src.constants import CONSOLIDATED_REPORTS
from src.logger import AppLogger
from src.reports.html_renderer import REPORT_CSS, HTMLRenderer
from src.reports.kpi_calculator import (
    KPICalculator,
    OrderedDefaultDict,
    generate_executive_summary,
    get_cummilative_sumary
)
from src.reports.regression_report import RegressionAnalysisReport, get_two_run_ids_cluster_info
from src.utils.run_id_utils import iterate_db_get_testplan

logger = AppLogger().get_logger()

NUM_FAILURES_TO_SHOW = 10

__all__ = [
    "ConsolidatedReportAnalysis",
    "CombinedRegressionAnalysis",
    "should_process_id",
    "run_report_generation_for_all_qairt_ids",
]


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
        logger.info(
            f"previous testplan id: {previous_testplan_id} : Current testplan id: {run_id}: previous release id: {previous_release_testplan_id}"
        )
        return previous_testplan_id if previous_testplan_id else previous_release_testplan_id

    def get_unqiue_runids(self, qairt_id):
        try:
            qairt_folder = os.path.join(self.reports_folder_path, qairt_id)
            functional_report_file = None
            for file in os.listdir(qairt_folder):
                if file.startswith("Functional_"):
                    functional_report_file = file
                    break

            if functional_report_file is None:
                logger.warning(f"Not able to find any Functional report for: {qairt_id}")
                return []
            logger.info(f"Attempting to read file: {functional_report_file}")
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
        self.list_of_summay_to_avoid = ["no logs to provide", "no logs"]
        self.unique_gerrits_count = 0

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
                ("Auto", self._regex(r"(?:^|[^a-z0-9])auto(?:[^a-z0-9]|$)")),
                ("GenAI", self._regex(r"(?:^|[^a-z0-9])llm(?:[^a-z0-9]|$)")),
                ("Mobile/IOT/XR", self._regex(r"(?:^|[^a-z0-9])pt(?:[^a-z0-9]|$)")),
            ]

        for label, predicate in rules:
            if predicate(run_id):
                return label
        return "Unknown"

    def _make_kpi_calculator(self) -> KPICalculator:
        """Create a :class:`KPICalculator` snapshot bound to the current state.

        Returns:
            A :class:`KPICalculator` instance holding a reference to the current
            regression analysis objects and helper callables.
        """
        return KPICalculator(
            regression_analysis_object=self._regression_analysis_object,
            classify_run_id_fn=self.classify_run_id,
            unique_gerrits_count=self.unique_gerrits_count,
            summaries_to_avoid=self.list_of_summay_to_avoid,
        )

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

    def merge_two_jsons(self, dst: Dict[str, Any], src: Dict[str, Any], dedupe: bool = False) -> Dict[str, Any]:
        """
        ### Qgenie generated code, don't modify unless you know what you are doing !!!
        Deeply merge JSON-like `src` into `dst` (in place), and return `dst`.

        Merge rules:
        1) If a key is missing in `dst`, copy it from `src`.
        2) If both values are dicts, merge them recursively.
        3) If both values are lists, extend `dst` with `src`'s list items.
            - If `dedupe=True`, remove duplicates while preserving order:
                * primitives (str/int/float/bool/None) compared by value
                * non-primitives compared by their repr(item)
        4) If value types differ or are scalars, overwrite `dst[key]` with a deep copy of `src[key]`.

        Args:
            dst: Destination dict that will be updated in-place.
            src: Source dict whose contents will be merged into `dst`.
            dedupe: If True and both values are lists, deduplicate after extending.

        Returns:
            The updated `dst` dict.

        Notes:
            - This function mutates `dst`.
            - Deduplication for non-primitive list items uses repr(item), which is a heuristic.
            If you need stable deduplication for complex objects, consider a custom key function.
        """
        for key, src_value in src.items():
            # If key doesn't exist in dst, just deep-copy from src
            if key not in dst:
                dst[key] = deepcopy(src_value)
                continue

            dst_value = dst[key]

            # Case 1: Both are dicts -> recursive deep merge
            if isinstance(dst_value, dict) and isinstance(src_value, dict):
                self.merge_two_jsons(dst_value, src_value, dedupe=dedupe)

            # Case 2: Both are lists -> extend, optionally dedupe
            elif isinstance(dst_value, list) and isinstance(src_value, list):
                dst_value.extend(src_value)

                if dedupe:
                    seen_markers = set()
                    unique_items = []

                    for item in dst_value:
                        # Use direct value as marker for primitives; repr for complex types
                        if isinstance(item, (str, int, float, bool, type(None))):
                            marker = item
                        else:
                            marker = repr(item)

                        if marker not in seen_markers:
                            seen_markers.add(marker)
                            unique_items.append(item)

                    dst[key] = unique_items

            # Case 3: Type mismatch or scalar values -> overwrite with src's deep copy
            else:
                dst[key] = deepcopy(src_value)

        return dst

    def generate_each_run_id_regression_report(self, qairt_id):
        unique_run_ids_for_qairt_id = self.consolidated_report_analysis.get_unqiue_runids(qairt_id)
        if not unique_run_ids_for_qairt_id:
            logger.error(f"No run ids found for {qairt_id}")
            return unique_run_ids_for_qairt_id

        logger.info(f"Got all the run ids for qairt id: {qairt_id}: Run IDS: {unique_run_ids_for_qairt_id}")
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
                logger.info(f"{_id} already processed skipping")
                continue

            prev_id = self.consolidated_report_analysis.build_prev_run_id(_id)
            logger.info(f"Processing: {_id}: {prev_id}")
            regression_json = get_two_run_ids_cluster_info(_id, prev_id, force=True)
            regression_analysis = RegressionAnalysisReport(qairt_id)
            html_path = regression_analysis.generate_regression_analysis_report(_id, prev_id, regression_json)
            logger.info(f"HTML for {_id}: {html_path}")

            self._regression_analysis_object[_id] = regression_analysis
            self._regression_html_paths[_id] = html_path
            if html_path:
                self.__processed_run_id = True

        # only in case of new processing save the regression objects
        if self.__processed_run_id:
            self.save_regression_analysis_objects(qairt_id)

    def __generated_regressed_gerrits_page(self, qairt_id, gerrits_information):
        if gerrits_information is None:
            logger.info(f"No gerrit info found for: {qairt_id} skipping building gerrit page")
            return None

        output_dir = os.path.join(CONSOLIDATED_REPORTS.path, qairt_id, "regression_htmls")
        file_name = "gerrits_regression_report.html"
        file_path = os.path.join(output_dir, file_name)

        html_content = f"<html><head><title>Gerrits Merged</title>{REPORT_CSS}</head><body><div class='container'><h2>List of Gerrits merged in {qairt_id}</h2>"
        project_wise_gerrits = OrderedDefaultDict(list)
        unique_gerrits = set()
        for _, runtime_based_gerrit_data in (gerrits_information or {}).items():
            if not isinstance(runtime_based_gerrit_data, dict):
                continue
            for _, backend_gerrit_data in runtime_based_gerrit_data.items():
                for gerrit_data in backend_gerrit_data or []:
                    if not isinstance(gerrit_data, dict):
                        continue
                    commit_url = (gerrit_data.get("commit_url") or "").strip()
                    repo_name = gerrit_data.get("repository_name") or "-"
                    if commit_url and commit_url.lower() not in unique_gerrits:
                        project_wise_gerrits[repo_name].append(gerrit_data)
                        unique_gerrits.add(commit_url.lower())

        for repo_name, gerrits_data in project_wise_gerrits.items():
            html_content += f"<h3>Repository Name: {repo_name}</h3>"
            html_content += "<table border='1'><tr><th>Gerrit Raised By</th><th>Email</th><th>Commit Message</th><th>Gerrit Link</th></tr>"
            for data in gerrits_data:
                commit_url_val = data.get("commit_url") or ""
                commit_url = f"<a href='{commit_url_val}' target='_blank'>Gerrit</a>" if commit_url_val else "-"
                raised_by = data.get("gerrit_raised_by") or []
                raised_by_name = raised_by[0].get("name", "-") if raised_by else "-"
                raised_by_email = raised_by[0].get("email", "-") if raised_by else "-"
                commit_message = data.get("commit_message") or "-"
                html_content += f"<tr><td>{raised_by_name}</td><td>{raised_by_email}</td><td>{commit_message}</td><td>{commit_url}</td></tr>"
            html_content += "</table>"

        html_content += "</body></html>"
        with open(file_path, "w") as f:
            f.write(html_content)

        self.unique_gerrits_count = len(unique_gerrits)
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

    def generate_gerrits_merged_report(self, gerrits_data=None):
        return self.__generated_regressed_gerrits_page(self._qairt_id, self._get_gerrits_data())

    def list_to_html_ul(self, items):
        """Convert a Python list into an HTML <ul><li>...</li></ul> block."""
        li_html = "".join(f"<li>{item}</li>" for item in items)
        return f"<ul>{li_html}</ul>"

    def get_runtime_based_gerrit_row(self, runtime, gerrit_data, rows_span=0):
        """Thin shim — delegates to :meth:`HTMLRenderer.runtime_gerrit_row`."""
        return HTMLRenderer.runtime_gerrit_row(runtime, gerrit_data, rows_span)

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
            logger.info(f"Generating executing summary for bu: {bu}: {run_ids}")
            qairt_regression_report += f"<h3>{bu.upper()} Analysis Report</h3>"

            qairt_regression_report += "<h4>Run IDS Processed</h4>"
            updated_run_ids = []
            for run_id in run_ids:
                if self._regression_html_paths.get(run_id):
                    updated_run_ids.append(
                        f"<a href='https://aisw-hyd.qualcomm.com/fs/{self._regression_html_paths[run_id]}'>{run_id}</a>"
                    )
                else:
                    updated_run_ids.append(run_id)
            qairt_regression_report += HTMLRenderer.list_to_html_ul(updated_run_ids)

            failure_data = self.fetch_filtered_regression_data_from_all_ids(filter=True, run_ids=run_ids)
            soc_errors_list = []
            for _, errors in failure_data.items():
                soc_errors_list.extend(errors)

            logger.info(f"Total errors: {len(soc_errors_list)}")
            qairt_regression_report += self.__get_bu_metrics_charts(run_ids)
            qairt_regression_report += generate_executive_summary(soc_errors_list)

        qairt_regression_report += "</div></body></html>"
        with open(qairt_regression_report_path, "w") as f:
            f.write(qairt_regression_report)
        return qairt_regression_report_path

    def add_gerrit_details_to_report(self, qairt_html_report):
        gerrits_merged_html_path = self.generate_gerrits_merged_report()
        if gerrits_merged_html_path:
            qairt_html_report += "<h3> Lists of Gerrits Merged </h3>"
            qairt_html_report += HTMLRenderer.list_to_html_ul(
                [f"<a href='{gerrits_merged_html_path}'>{self.unique_gerrits_count} Gerrits Merged</a>"]
            )
        else:
            qairt_html_report += "<h3> No Gerrits Merged !! </h3>"

        return qairt_html_report

    def fetch_filtered_regression_data_from_all_ids(self, key="soc_name", filter=False, run_ids=None) -> dict:
        """Thin shim — delegates to :meth:`KPICalculator.get_filtered_data`."""
        return self._make_kpi_calculator().get_filtered_data(key, apply_filter=filter, run_ids=run_ids)

    def __get_soc_failure_table(self, top_k=NUM_FAILURES_TO_SHOW):
        """Thin shim — delegates to :meth:`KPICalculator.build_soc_failure_table`."""
        return self._make_kpi_calculator().build_soc_failure_table(top_k)

    def __get_model_failure_table(self, top_k=NUM_FAILURES_TO_SHOW):
        """Thin shim — delegates to :meth:`KPICalculator.build_model_failure_table`."""
        return self._make_kpi_calculator().build_model_failure_table(top_k)

    def _bar_chart_html(self, data: list, color: str = "#00629B") -> str:
        """Thin shim — delegates to :meth:`HTMLRenderer.bar_chart_html`."""
        return HTMLRenderer.bar_chart_html(data, color)

    def _donut_html(self, data: list) -> str:
        """Thin shim — delegates to :meth:`HTMLRenderer.donut_html`."""
        return HTMLRenderer.donut_html(data)

    def __get_dsp_type_wise_failure_table(self, top_k=NUM_FAILURES_TO_SHOW):
        """Thin shim — delegates to :meth:`KPICalculator.build_dsp_failure_table`."""
        return self._make_kpi_calculator().build_dsp_failure_table(top_k)

    def __get_bu_metrics_charts(self, run_ids, top_k=NUM_FAILURES_TO_SHOW):
        """Thin shim — delegates to :meth:`KPICalculator.build_bu_metrics_charts`."""
        return self._make_kpi_calculator().build_bu_metrics_charts(run_ids, top_k)

    def __build_kpi_overview_html(self) -> str:
        """Thin shim — delegates to :meth:`KPICalculator.build_kpi_overview`."""
        return self._make_kpi_calculator().build_kpi_overview()

    def __build_bu_runtime_heatmap_html(self) -> str:
        """Thin shim — delegates to :meth:`KPICalculator.build_bu_runtime_heatmap`."""
        return self._make_kpi_calculator().build_bu_runtime_heatmap()

    def generate_qairt_regression_report(self, qairt_id):
        qairt_regression_report_path = os.path.join(
            CONSOLIDATED_REPORTS.path,
            qairt_id,
            f"{qairt_id}.html",
        )

        # In case of atleast on html genrated, then also generate the qairt report
        has_atleast_one_html = force_generate_qairt_report = False
        for run_id in self._regression_analysis_object:
            if self._regression_html_paths[run_id]:
                has_atleast_one_html = True
                break
        if has_atleast_one_html and not os.path.exists(qairt_regression_report_path):
            force_generate_qairt_report = True

        # if no run id is processed return the previous path with any processing
        if not force_generate_qairt_report and not self.__processed_run_id:
            return qairt_regression_report_path

        gerrits_data = self._get_gerrits_data(runtime_first=True)
        all_merged_gerrits_report = self.generate_gerrits_merged_report(gerrits_data)

        qairt_regression_report = (
            f"<html><head><title>Regression Report</title>{REPORT_CSS}</head><body><div class='container'>"
        )
        qairt_regression_report += f"<h2>QAIRT Analysis Report ({qairt_id})</h2>"
        qairt_regression_report += self.__build_kpi_overview_html()

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

        # build qairt report
        qairt_regression_report += (
            "<table border='1'><tr><th>Runtime/Tools</th><th>Type</th><th>Summary</th><th>Gerrits Merged</th></tr>"
        )

        # add tools row
        if converter_quantizer_dict:
            first_row = True
            for _, runtimes_dict in converter_quantizer_dict.items():
                runtimes = runtimes_dict.keys()
                summaries_list = [get_cummilative_sumary(runtimes_dict[runtime]) for runtime in runtimes]

                summary_idx_to_avoid = []
                for idx, summary in enumerate(summaries_list):
                    if any(summay_to_avoid in summary.lower() for summay_to_avoid in self.list_of_summay_to_avoid):
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
                            gerrit_cell = HTMLRenderer.runtime_gerrit_row("tools", gerrits_data, rowspan)
                            qairt_regression_report += (
                                f"<tr>"
                                f"<td rowspan='{rowspan}'>Tools</td>"
                                f"<td>{runtime.capitalize()}</td>"
                                f"<td>{summary_html}</td>"
                                f"{gerrit_cell}"
                                f"</tr>"
                            )
                            first_row = False
                        else:
                            qairt_regression_report += (
                                f"<tr>" f"<td>{runtime.capitalize()}</td>" f"<td>{summary_html}</td>" f"</tr>"
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
                    if not any(
                        summary_to_avoid in savecontext_summary.lower()
                        for summary_to_avoid in self.list_of_summay_to_avoid
                    ):
                        graph_prepare_row_data.append(("savecontext", savecontext_summary))
                    del combined_runtime_type_json_data[current_runtime]["savecontext"]

                if gprepare_data:
                    gprepare_summary = get_cummilative_sumary(gprepare_data)
                    if not any(
                        summary_to_avoid in gprepare_summary.lower()
                        for summary_to_avoid in self.list_of_summay_to_avoid
                    ):
                        graph_prepare_row_data.append(("graph prepare", gprepare_summary))
                    del combined_runtime_type_json_data[current_runtime]["graph_prepare"]

                rowspan = len(graph_prepare_row_data)
                if rowspan:
                    first_runtime = graph_prepare_row_data[0][0]
                    first_summary = graph_prepare_row_data[0][1]
                    gerrit_cell = HTMLRenderer.runtime_gerrit_row("htp", gerrits_data, rowspan)
                    qairt_regression_report += (
                        f"<tr>"
                        f"<td rowspan='{rowspan}'>Graph Prepare</td>"
                        f"<td>{first_runtime}</td>"
                        f"<td><ul>{first_summary}</ul></td>"
                        f"{gerrit_cell}"
                        f"</tr>"
                    )
                    for runtime, summary in graph_prepare_row_data[1:]:
                        qairt_regression_report += f"<tr>" f"<td>{runtime}</td>" f"<td><ul>{summary}</ul></td>" f"</tr>"

            types_to_process_cpu = ["inference", "verifier"]
            all_cpu_types = list(types_dict.keys())
            all_cpu_summaries = [get_cummilative_sumary(types_dict[_type]) for _type in all_cpu_types]

            summary_idx_to_avoid = []
            for idx, summary in enumerate(all_cpu_summaries):
                if not any(summay_to_avoid in summary.lower() for summay_to_avoid in self.list_of_summay_to_avoid):
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
                gerrit_cell = HTMLRenderer.runtime_gerrit_row(current_runtime, gerrits_data, rowspan)
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
                if any(summay_to_avoid in summary.lower() for summay_to_avoid in self.list_of_summay_to_avoid):
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
                gerrit_cell = HTMLRenderer.runtime_gerrit_row(current_runtime, gerrits_data, rowspan)
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

        if all_merged_gerrits_report:
            all_merged_gerrits_report_url = f"<a href='{all_merged_gerrits_report}'>All Merged Gerrits</a>"
        else:
            all_merged_gerrits_report_url = "-"
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
                    f"<td rowspan='{rowspan}'>{all_merged_gerrits_report_url}</td>"
                    f"</tr>"
                )
                for runtime, summary in core_data[1:]:
                    qairt_regression_report += f"<tr>" f"<td>{runtime}</td>" f"<td><ul>{summary}</ul></td>" f"</tr>"
        else:
            qairt_regression_report += (
                f"<tr>"
                f"<td>Core</td>"
                f"<td>-</td>"
                f"<td>-</td>"
                f"<td>{all_merged_gerrits_report_url}</td>"
                f"</tr>"
            )
        qairt_regression_report += "</table>"

        qairt_regression_report += self.__get_dsp_type_wise_failure_table()
        qairt_regression_report += self.__get_soc_failure_table()
        qairt_regression_report += self.__get_model_failure_table()

        qairt_regression_report += self.__build_bu_runtime_heatmap_html()

        # bu_summary_path = f"{CONSOLIDATED_REPORTS.path}/{qairt_id}/regression_htmls/BU_{qairt_id}.html"
        # if not os.path.exists(bu_summary_path):
        bu_summary_path = self.generate_bu_regression_report(qairt_id)
        qairt_regression_report += HTMLRenderer.list_to_html_ul(
            [
                f"<a href='https://aisw-hyd.qualcomm.com/fs/{bu_summary_path}' target='_blank' style='font-size: 20px; text-decoration: none;'>BU Summary Page</a>"
            ]
        )

        qairt_regression_report += "</div></body></html>"
        with open(qairt_regression_report_path, "w") as f:
            f.write(qairt_regression_report)
        return qairt_regression_report_path

    def generate_final_summary_report(self, qairt_id=None):
        self._qairt_id = qairt_id
        self.generate_each_run_id_regression_report(qairt_id)
        return self.generate_qairt_regression_report(qairt_id)


def _extract_id_date(qaisw_id: str) -> date | None:
    """
    Extracts a date from the 12-digit stamp in qaisw IDs:
    Example: qaisw-v2.44.0.260112072337_193906_nightly
             ---------------- 260112072337 -> YYMMDDHHMMSS
             Returns a datetime.date(YYYY, MM, DD)
    """
    m = re.search(r"-v\d+\.\d+\.\d+\.(\d{12})_", qaisw_id)
    if not m:
        return None
    stamp = m.group(1)  # YYMMDDhhmmss
    try:
        yy = int(stamp[0:2])
        mm = int(stamp[2:4])
        dd = int(stamp[4:6])
        # Map YY to 2000-2099 (adjust if your epoch differs)
        yyyy = 2000 + yy
        return date(yyyy, mm, dd)
    except (ValueError, OverflowError):
        return None


def _months_back(d: date, months: int) -> date:
    return d - relativedelta(months=months)


def should_process_id(qaisw_id: str, reference_date: date = date.today(), months_window: int = 1) -> bool:
    """
    Return True if this id's date is within [reference_date - months_window, reference_date] inclusive.
    Only compares dates (ignores time).
    """
    id_d = _extract_id_date(qaisw_id)
    if not id_d:
        return False
    start = _months_back(reference_date, months_window)
    return start <= id_d <= reference_date


def run_report_generation_for_all_qairt_ids(reference_date: date = date.today(), months_window: int = 1):
    qairt_ids = sorted(os.listdir(CONSOLIDATED_REPORTS.path), reverse=True)
    qairt_ids = [q for q in qairt_ids if q.startswith("qaisw")]
    qairt_ids = [q for q in qairt_ids if should_process_id(q, reference_date, months_window)]
    logger.info(f"Processing: {len(qairt_ids)} Qairt Ids")

    for qairt_id in qairt_ids:
        try:
            report_analysis = CombinedRegressionAnalysis(ConsolidatedReportAnalysis())
            report_analysis.generate_final_summary_report(qairt_id)
            logger.info(f"{qairt_id} Successfully processed !!")
        except Exception as e:
            logger.info(f"Exception while processing: {qairt_id}: {e}")
            continue


if __name__ == "__main__":
    report_analysis = CombinedRegressionAnalysis(ConsolidatedReportAnalysis())
    qairt_id = "qaisw-v2.44.0.260112072337_193906_nightly"
    report_analysis.generate_final_summary_report(qairt_id)
