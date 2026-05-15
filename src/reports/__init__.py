"""Reports package — regression and consolidated report generation.

Modules
-------
consolidated_report  — :class:`ConsolidatedReportAnalysis`, :class:`CombinedRegressionAnalysis`.
regression_report    — :class:`RegressionAnalysisReport`, :class:`RegressionReport`.
kpi_calculator       — :class:`KPICalculator` (KPI extraction from results).
html_renderer        — :class:`HTMLRenderer` shared rendering utilities.
enhanced_report      — :class:`EnhancedReport` (QAIRT-id driven reports).

Layering
--------
``reports`` imports from ``src.data``, ``src.pipeline``, ``src.llm``,
``src.utils``, ``src.constants``, and ``src.logger``.
"""

from src.reports.consolidated_report import (
    CombinedRegressionAnalysis,
    ConsolidatedReportAnalysis,
    run_report_generation_for_all_qairt_ids,
    should_process_id
)
from src.reports.enhanced_report import EnhancedReport
from src.reports.html_renderer import REPORT_CSS, HTMLRenderer
from src.reports.kpi_calculator import KPICalculator
from src.reports.regression_report import RegressionAnalysisReport, RegressionReport

__all__ = [
    "HTMLRenderer",
    "REPORT_CSS",
    "KPICalculator",
    "EnhancedReport",
    "ConsolidatedReportAnalysis",
    "CombinedRegressionAnalysis",
    "RegressionAnalysisReport",
    "RegressionReport",
    "run_report_generation_for_all_qairt_ids",
    "should_process_id",
]
