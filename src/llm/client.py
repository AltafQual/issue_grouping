"""QGenie LLM client with retry logic and pre-configured model instances.

Provides :class:`CustomQGenieChat` (a ``QGenieChat`` subclass with
exponential-backoff retry) and :class:`QgenieModels` (a namespace of
pre-built model instances).

All Pydantic output schemas used by the LLM chains also live here to keep
them co-located with the client that drives them.

Layering
--------
Imports from ``src.constants``, ``src.core``, ``src.logger``, and
standard library / third-party packages.  No imports from ``src.helpers``
or higher-level packages.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any, Optional

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatResult
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError
from qgenie.integrations.langchain import QGenieChat

from src.constants import QGENEIE_API_KEY
from src.core.exceptions import LLMError
from src.logger import AppLogger
from src.utils.timer import execution_timer

logger = AppLogger().get_logger(__name__)

__all__ = [
    "CustomQGenieChat",
    "QgenieModels",
    "get_exponential_backoff_delay",
    # Pydantic output schemas
    "ClusteringResult",
    "NameClusteringResult",
    "MergeResult",
    "ReclusterResult",
    "ClassifyClusterGroup",
    "SubClusterVerifierFailed",
    "NearDuplicateResult",
    "NearDuplicateResultList",
    # Summary generation
    "error_summary_generation",
    "cummilative_summary_generation",
]

_MAX_RETRIES = 10


def get_exponential_backoff_delay(attempt: int, base_delay: int = 5, max_delay: int = 200) -> int:
    """Return exponential back-off delay in seconds for *attempt*.

    Args:
        attempt: Current retry attempt (1-indexed).
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap.

    Returns:
        Integer number of seconds to wait before the next retry.
    """
    return min(base_delay * (2 ** (attempt - 1)), max_delay)


# ---------------------------------------------------------------------------
# QGenie client
# ---------------------------------------------------------------------------


class CustomQGenieChat(QGenieChat):
    """``QGenieChat`` subclass with exponential-backoff retry on failures.

    Overrides both ``_generate`` (sync) and ``_agenerate`` (async) to retry
    up to 10 times on transient errors, skipping retries when the model
    returns a ``content_filter`` finish reason.

    Args:
        Accepts all keyword arguments of :class:`QGenieChat`.
    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation with retry.

        Args:
            messages: Conversation messages to send.
            stop: Optional stop sequences.
            run_manager: LangChain callback manager.
            **kwargs: Additional keyword arguments forwarded to the API.

        Returns:
            :class:`~langchain_core.outputs.ChatResult` from the model.
        """
        message_dicts, params = self._create_message_dicts(messages)
        params = {**params, **kwargs}
        params.pop("stream", None)

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self.client.chat(messages=message_dicts, **params)
                return self._create_chat_result(response)
            except ValidationError as exc:
                if "content_filter" in str(exc):
                    logger.warning("LLM: content_filter finish_reason; skipping retries.")
                    break
                logger.error(f"LLM ValidationError (attempt {attempt}): {traceback.format_exc()}")
            except Exception:
                logger.error(f"LLM error (attempt {attempt}): {traceback.format_exc()}")

            if attempt < _MAX_RETRIES:
                time.sleep(get_exponential_backoff_delay(attempt))

        return self._create_chat_result({})

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronous generation with retry.

        Args:
            messages: Conversation messages to send.
            stop: Optional stop sequences.
            run_manager: LangChain async callback manager.
            stream: Ignored (streaming not supported).
            **kwargs: Additional keyword arguments forwarded to the API.

        Returns:
            :class:`~langchain_core.outputs.ChatResult` from the model.
        """
        message_dicts, params = self._create_message_dicts(messages)
        params = {**params, **kwargs}
        params.pop("stream", None)

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await self.async_client.chat(messages=message_dicts, **params)
                return self._create_chat_result(response)
            except ValidationError as exc:
                if "content_filter" in str(exc):
                    logger.warning("LLM: content_filter finish_reason; skipping retries.")
                    break
                logger.error(f"LLM async ValidationError (attempt {attempt}): {traceback.format_exc()}")
            except Exception:
                logger.error(f"LLM async error (attempt {attempt}): {traceback.format_exc()}")

            if attempt < _MAX_RETRIES:
                await asyncio.sleep(get_exponential_backoff_delay(attempt))

        return self._create_chat_result({})


# ---------------------------------------------------------------------------
# Pre-configured model instances
# ---------------------------------------------------------------------------


class QgenieModels:
    """Namespace of pre-configured :class:`CustomQGenieChat` instances.

    Use these instances directly in LangChain chains to avoid constructing
    model objects in every function call.

    Example:
        >>> chain = prompt | QgenieModels.gemini_2_5_flash | output_parser
    """

    gemini_2_5_pro: CustomQGenieChat = CustomQGenieChat(
        model="vertexai::gemini-2.5-pro", api_key=QGENEIE_API_KEY, temperature=0.2, max_retries=5, timeout=5000
    )
    gemini_2_5_flash: CustomQGenieChat = CustomQGenieChat(
        model="vertexai::gemini-2.5-flash", api_key=QGENEIE_API_KEY, temperature=0.2, max_retries=5, timeout=5000
    )
    gemini_3_flash: CustomQGenieChat = CustomQGenieChat(
        model="vertexai::gemini-3-flash-preview", api_key=QGENEIE_API_KEY, temperature=0.2, max_retries=5, timeout=5000
    )
    azure_gpt_5_2: CustomQGenieChat = CustomQGenieChat(
        model="azure::gpt-5.2", api_key=QGENEIE_API_KEY, temperature=0.2, max_retries=5, timeout=5000
    )
    azure_o3_mini: CustomQGenieChat = CustomQGenieChat(
        model="azure::o3-mini", api_key=QGENEIE_API_KEY, temperature=0.2, max_retries=5, timeout=5000
    )
    azure_o3: CustomQGenieChat = CustomQGenieChat(
        model="azure::o3", api_key=QGENEIE_API_KEY, temperature=0.2, max_retries=5, timeout=5000
    )
    azure_gpt_5_4_mini: CustomQGenieChat = CustomQGenieChat(
        model="azure::gpt-5.4-mini", api_key=QGENEIE_API_KEY, temperature=0.2, max_retries=5, timeout=5000
    )
    azure_gpt_5_4: CustomQGenieChat = CustomQGenieChat(
        model="azure::gpt-5.4", api_key=QGENEIE_API_KEY, temperature=0.2, max_retries=5, timeout=5000
    )
    claude_4_5_haiku: CustomQGenieChat = CustomQGenieChat(
        model="anthropic::claude-4-5-haiku", api_key=QGENEIE_API_KEY, temperature=0.2, max_retries=5, timeout=5000
    )
    claude_4_5_sonnet: CustomQGenieChat = CustomQGenieChat(
        model="anthropic::claude-4-5-sonnet", api_key=QGENEIE_API_KEY, temperature=0.2, max_retries=5, timeout=5000
    )
    claude_4_5_opus: CustomQGenieChat = CustomQGenieChat(
        model="anthropic::claude-4-6-opus:1M", api_key=QGENEIE_API_KEY, temperature=0.2, max_retries=5, timeout=5000
    )


# ---------------------------------------------------------------------------
# Pydantic output schemas
# ---------------------------------------------------------------------------


class ClusteringResult(BaseModel):
    """LLM response schema for cluster analysis (name + misclassified IDs)."""

    cluster_name: str = Field(description="Name to the whole cluster")
    misclassified_ids: str = Field(description="Misclassified error ids in that cluster")


class NameClusteringResult(BaseModel):
    """LLM response schema for cluster naming."""

    cluster_name: str = Field(description="Name for the whole cluster")


class MergeResult(BaseModel):
    """LLM response schema for cluster merging."""

    merged_name: str = Field(description="Name for the merged cluster")
    outlier_indices: list[int] = Field(description="Indices of logs that do not belong in the merged cluster")


class ReclusterResult(BaseModel):
    """LLM response schema for outlier re-clustering."""

    clusters: list[dict] = Field(description="List of clusters with name and log indices")


class ClassifyClusterGroup(BaseModel):
    """LLM response schema for cluster type classification."""

    environment_issue: bool = Field(description="Whether this cluster is an environment issue")
    setup_issue: bool = Field(description="Whether this cluster is a setup failure")
    sdk_issue: bool = Field(description="Whether this is an SDK-related issue")


class SubClusterVerifierFailed(BaseModel):
    """LLM response schema for verifier-failed sub-clustering."""

    cluster_name: str = Field(description="Name of the sub-cluster")
    indices: list[int] = Field(description="Indices of logs belonging to this sub-cluster")
    previous_clusters: dict = Field(description="Existing verifier-failed clusters for re-grouping")


class NearDuplicateResult(BaseModel):
    """LLM response schema for a single near-duplicate pair assessment."""

    cluster_a: str = Field(description="Name of the first cluster")
    cluster_b: str = Field(description="Name of the second cluster")
    is_duplicate: bool = Field(description="Whether the two clusters are near-duplicates")
    reason: str = Field(description="One-sentence technical explanation")
    keep_name: Optional[str] = Field(description="Cluster name to keep if duplicate, else null")


class NearDuplicateResultList(BaseModel):
    """LLM response schema for a batch of near-duplicate pair assessments."""

    results: list = list[NearDuplicateResult]


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


@execution_timer
def error_summary_generation(errors_list: list[str]) -> str:
    """Generate a concise summary of a list of error logs via LLM.

    For lists of 10+ errors, uses Gemini 2.5 Pro for better synthesis;
    smaller lists use the faster O3 Mini model.

    Args:
        errors_list: List of error log strings to summarise.

    Returns:
        Summary string produced by the LLM.
    """
    from src.llm import prompts  # deferred to avoid circular import at class-definition time

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompts.ERROR_SUMMARIZATION_PROMPT), ("human", prompts.ERROR_LOGS_LIST)]
    )
    model_to_use = QgenieModels.azure_o3_mini
    if len(errors_list) >= 10:
        model_to_use = QgenieModels.gemini_2_5_pro
    chain = prompt_template | model_to_use | StrOutputParser()
    error_logs = "\n\n".join(f"Error Logs {i}:\n{error}" for i, error in enumerate(errors_list, start=1))
    return chain.invoke({"logs": error_logs})


@execution_timer
def cummilative_summary_generation(errors_list: list[str], short_final_summary: bool = False) -> str:
    """Generate a cumulative multi-pass summary of error logs.

    Processes logs in windows of 10 concurrently, then combines all
    per-window summaries into a single final summary.

    Args:
        errors_list: List of error log strings to summarise.
        short_final_summary: If ``True``, use the shorter final-summary prompt.

    Returns:
        Final summary string.
    """
    from src.llm import prompts  # deferred to avoid circular import at class-definition time

    def _chunk(iterable: list, size: int):
        for i in range(0, len(iterable), size):
            yield iterable[i : i + size]

    async def _process_windows_concurrently(windows: list) -> list[str]:
        semaphore = asyncio.Semaphore(5)

        async def _process_window(index: int, error_window: list) -> tuple[int, str]:
            async with semaphore:
                logger.info(f"Processing window {index} with length {len(error_window)}")
                pt = ChatPromptTemplate.from_messages(
                    [("system", prompts.SUMMARY_GENERATION_PROMPT), ("human", prompts.ERROR_LOGS_LIST)]
                )
                chain = pt | QgenieModels.azure_gpt_5_4 | StrOutputParser()
                logs_str = "\n\n".join(f"Error Logs {i}:\n{e}" for i, e in enumerate(error_window, start=1))
                summary = await chain.ainvoke({"logs": logs_str})
                return index, summary

        tasks = [_process_window(i, window) for i, window in enumerate(windows, start=1)]
        results = await asyncio.gather(*tasks)
        return [summary for _, summary in sorted(results, key=lambda x: x[0])]

    error_windows = list(_chunk(errors_list, 10))
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    summaries_list = loop.run_until_complete(_process_windows_concurrently(error_windows))

    logger.info(f"Total summaries generated: {len(summaries_list)}. Generating final summary.")
    final_sys = (
        prompts.SHORT_PARENT_SUMMARY_GENERATION_PROMPT
        if short_final_summary
        else prompts.PARENT_SUMMARY_GENERATION_PROMPT
    )
    pt = ChatPromptTemplate.from_messages([("system", final_sys), ("human", prompts.ERROR_LOGS_LIST)])
    chain = pt | QgenieModels.gemini_2_5_pro | StrOutputParser()
    all_summaries = "\n\n".join(f"Error Logs Summary {i}:\n{s}" for i, s in enumerate(summaries_list, start=1))
    return chain.invoke({"logs": all_summaries})
