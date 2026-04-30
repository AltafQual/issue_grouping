"""Enhanced QAIRT consolidated report.

Provides :class:`EnhancedReport` and re-exports the public API from
:mod:`src.enhanced_consolidated_report`.

Layering
--------
Imports from ``src.enhanced_consolidated_report`` (legacy) and
``src.logger``.
"""

from __future__ import annotations

import sys

from src.enhanced_consolidated_report import main as _enhanced_main
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)

__all__ = ["EnhancedReport"]


class EnhancedReport:
    """Enhanced HTML report for a QAIRT release identifier.

    Reads the ``joblib`` artifact from the consolidated reports pipeline and
    generates a richer HTML report with KPI dashboard, BU cards, heatmaps,
    and cluster analysis.

    Args:
        qairt_id: QAIRT release identifier
            (e.g. ``"qaisw-v2.46.0.260319041023_nightly"``).
        no_llm: Skip LLM enrichment; generate metrics-only report.
        cache_llm: Cache LLM results to disk.

    Example::

        report = EnhancedReport(qairt_id="qaisw-v2.46.0...")
        report.generate()
    """

    def __init__(
        self,
        qairt_id: str,
        no_llm: bool = False,
        cache_llm: bool = False,
    ) -> None:
        self.qairt_id = qairt_id
        self.no_llm = no_llm
        self.cache_llm = cache_llm

    def generate(self) -> None:
        """Generate the enhanced HTML report (writes to disk).

        Delegates to the standalone ``main()`` entry point in
        :mod:`src.enhanced_consolidated_report`.
        """
        # Simulate CLI args expected by the legacy entry point
        argv_backup = sys.argv
        sys.argv = ["enhanced_report", "--qairt_id", self.qairt_id]
        if self.no_llm:
            sys.argv.append("--no_llm")
        if self.cache_llm:
            sys.argv.append("--cache_llm")
        try:
            _enhanced_main()
        finally:
            sys.argv = argv_backup
