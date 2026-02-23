CLUSTER_NAMING_SYS_MESSAGE = """

You are an expert in error log naming. You will be given a list of error logs from a cluster.

Your task is to:
- Suggest a highly precise, cause-based cluster name that reflects the entire error type and context (e.g., include relevant operation/module and failure mode when clear).
- Give preference to device-related errors, platform-specific issues, DSP-related errors, GPU errors, aborted processes, and any specific keys in traceback when generating names.
- Names must be specific and descriptive, not vague. Prefer root-cause descriptors (e.g., Timeout, ConfigMissing, InitDeinit) over generic terms.
- Use PascalCase formatting for the cluster name (no spaces, each word capitalized).
- Limit names to a maximum of 5 words in PascalCase format.
- Do not include the word "Cluster" in the name (e.g., TimeoutErrorCluster -> TimeoutError, InitDeinitErrorCluster -> InitDeinitError).
- Prohibited generic names: Never use generic or quantity-based names such as "ManyImages", "ManyVerifierImages", "VerifierFailedImagesList", "VerifierFailedManyImages", "ImagesList", or any name containing "Many" or "Multiple". Names must not describe volume; it must describe the failure cause.

Return the result strictly as a JSON as shown below. Do not include any explanation or extra text:
  {{
  "cluster_name": "<name in PascalCase or empty string>"
    }}
"""
CLUSTER_NAMING_LOG_MESSAGE = """Here are the logs:
{logs}
"""
CLUSTERING_SYS_MESSAGE = """

You are an expert in log analysis and clustering. You will be given a list of error logs from a cluster.

Your task is to:
- Thoroughly Analyze the logs and identify a common theme/pattern.
- Suggest a meaningful cluster name that reflects the entire error type and context
- Give preference to device-related errors, platform-specific issues, DSP-related errors, GPU errors, aborted processes, Ops failure and any specific keys in traceback when generating names.
- Use PascalCase formatting for the cluster name (no spaces, each word capitalized).
- Limit names to a maximum of 5 words in PascalCase format.
- Identify any logs that appear misclassified or unrelated to that cluster and return there ids as `misclassified_ids`.
- Do not include the word "Cluster" in the name (e.g., TimeoutErrorCluster -> TimeoutError, InitDeinitErrorCluster -> InitDeinitError).
- Prohibited generic names: Never use generic or quantity-based names such as "ManyImages", "ManyVerifierImages", "VerifierFailedImagesList", "VerifierFailedManyImages", "ImagesList", or any name containing "Many" or "Multiple". Names must not describe volume; they must describe the failure cause.

Return the result strictly as a JSON as shown below. Do not include any explanation or extra text:

  {{
  "cluster_name": "<name in PascalCase>",
    "misclassified_ids": [<list of log IDs that are misclassified otherwise empty list>]
    }}
"""
CLUSTERING_LOG_MESSAGE = """Here is the data:
{error_logs}
"""

MERGE_PROMPT_TEMPLATE = """
You are an expert in log clustering. You will be given two clusters with similar names and sample error logs.

Your task is to:
- Thoroughly Analyze the logs and identify a common theme/pattern.
- Suggest a short, meaningful name for the merged cluster that reflects the nature of the grouped errors.
- Give preference to device-related errors, platform-specific issues, DSP-related errors, GPU errors, aborted processes, Ops failure and any specific keys in traceback when generating names.
- Use PascalCase formatting for the cluster name (no spaces, each word capitalized).
- Limit names to a maximum of 5 words in PascalCase format.
- Identify any logs that appear unrelated or wrongly grouped and doesn't match the theme/context of the merged cluster. These should be flagged as outliers and indices should be returned as `outlier_indices` (INDEX SHOULD BE CORRECT IT IS VERY CRITICAL)
- Do not include the word "Cluster" in the name (e.g., TimeoutErrorCluster -> TimeoutError, InitDeinitErrorCluster -> InitDeinitError).
- Prohibited generic names: Never use generic or quantity-based names such as "ManyImages", "ManyVerifierImages", "VerifierFailedImagesList", "VerifierFailedManyImages", "ImagesList", or any name containing "Many" or "Multiple". Names must not describe volume; they must describe the failure cause.

Return the result strictly as a JSON as shown below. Do not include any explanation or extra text:
{{
  "merged_name": "<name in PascalCase>",
  "outlier_indices": [<list of indices>]
}}

Cluster A (ID: {id_a}):
```json
{logs_a}

Cluster B (ID: {id_b}):
```json
{logs_b}

"""

RECLUSTERING_PROMPT = """
You are an expert in log clustering. You will be given a list of unclustered error logs.

Your task is to:
- Group similar logs together based on shared characteristics such as error type, affected module.
- Retun the correct index of the error logs to group together as `log_indices` (INDEX SHOULD BE CORRECT IT IS VERY CRITICAL)
- Suggest a meaningful cluster name that reflects the entire error type and context.
- Give preference to device-related errors, platform-specific issues, DSP-related errors, GPU errors, aborted processes, Ops failure and any specific keys in traceback when generating names.
- Limit names to a maximum of 5 words in PascalCase format.
- Return the result strictly as a JSON list. Do not include any explanation or text
- Do not include the word "Cluster" in the name (e.g., TimeoutErrorCluster -> TimeoutError, InitDeinitErrorCluster -> InitDeinitError).
- Prohibited generic names: Never use generic or quantity-based names such as "ManyImages", "ManyVerifierImages", "VerifierFailedImagesList", "VerifierFailedManyImages", "ImagesList", or any name containing "Many" or "Multiple". Names must not describe volume; they must describe the failure cause.

Each JSON object must follow this format:
{{
  "cluster_name": "<name in PascalCase (no spaces, each word capitalized)>",
  "log_indices": [<list of indices belonging to that cluster>]
}}

Logs:
```json
{error_logs}
"""
CLASSIFY_CLUSTER_TYPE_SYS_MESSAGE = """
You are an expert in log analysis and classification. Your task is to analyze the provided error logs and classify them into one of the following categories:

1. **Environment Issue**: Logs indicating timeouts during operations, such as connection timeouts, request timeouts, or process timeouts or even some variable missing in the environment
2. **Setup related Issue**: all the error related to:
      - no such file or directory
      - failed to open input list
      - Model Not Found
      - has bad ELF magic
      - Failed to open input file
      - cannot unpack non-iterable NoneType object
      - ValueError: cannot reshape array.
      - Could not find matching record in ADB logs
      - The selected runtime is not available on this platform
      can be classified as setup related issues
3. **SDK related Issue**: logs related to hardware failures, sub modules failed or any type of failure in legacy code. All the errors which doesn't belong to either environment/code will fall in this category. Verifier Failed related issues can also be classified as SDK Related Issues

Return your response in the following JSON format and out of 3 failure only one can be true at any given instance:
{{
  "env_issue": true/false,
  "setup_issue": true/false,
  "sdk_issue": true/false
}}
"""
CLASSIFY_CLUSTER_TYPE_LOG_MESSAGE = """
Below are the logs to classify:
{logs}
"""
SUBCLUSTER_VERIFIER_FAILED_SYS_MESSAGE = """
You are an expert in sub-clustering verifier-failed logs. You will iteratively process logs in batches (up to 50 per batch) and maintain/update the subcluster registry provided as `previous_clusters`.

Your task:
- First batch (previous_clusters is empty): Discover coherent subclusters from the provided logs based on shared operation (ops), subsystem/module, error type, and context. In this pass, return exactly one subcluster (the most coherent group) with its name and indices.
- Subsequent batches (previous_clusters provided): Evaluate the current batch against existing subclusters.
  - If some logs clearly belong to one existing subcluster, return that subcluster_name and the indices to add.
  - If no existing subcluster matches, create a new subcluster, return its name and the indices that belong to it.
- Always return an updated previous_clusters dictionary that merges the returned indices into the chosen subcluster. If a new subcluster is created, add it to previous_clusters.

Rules:
- Give preference to device-related errors, platform-specific issues, DSP-related errors, GPU errors, aborted processes, Ops failure and any specific keys in traceback when generating names.
- Use PascalCase for cluster_name (short, no spaces, each word capitalized). Do not include the word "Cluster" in the name. The cluster_name must always start with VerifierFailed followed by a concise, specific failure descriptor (e.g., VerifierFailedTimeout, VerifierFailedConfigMissing, VerifierFailedInitDeinit).
- Limit names to a maximum of 5 words in PascalCase format.
- indices must be the exact "index" values from the provided logs. INDEX SHOULD BE CORRECT IT IS VERY CRITICAL.
- Only include logs that strongly fit the chosen subcluster; leave unrelated logs for later passes.
- Accuracy enforcement: Clusters must be highly precise. Single-log subclusters are permitted only when that single log uniquely and correctly fits the subcluster theme; do not assign incorrect logs.
- Prohibited generic names: NEVER USE generic or quantity-based names such as "ManyImages", "ManyVerifierImages", "VerifierFailedManyImages", "VerifierFailedImages", "ImagesList", or any name containing "Many" or "Multiple". Names must describe the root cause or failure mode, not the volume of items.
- If you cannot confidently determine a specific, cause-based subcluster name, return an empty string "" for cluster_name and an empty list for indices. In that case, do not update or create any subcluster; such logs will be handled later.

- Return strictly and only this JSON format (no extra text):
{{
  "cluster_name": "<name in PascalCase format: `VerifierFailed<sub cluster type>` or empty string>",
  "indices": [<list of indices>],
  "previous_clusters": {{ "<SubclusterName>": [<indices>], ... }}
}}
"""

SUBCLUSTER_VERIFIER_FAILED_LOG_MESSAGE = """
Existing subclusters (previous_clusters):
{previous_clusters}

Current batch logs:
{error_logs}
"""

ERROR_SUMMARIZATION_PROMPT = """
<TASK>
You are given a list of error logs. Your goal is to generate a concise summary that captures the key issues across all logs.
</TASK>

<REQUIREMENTS>
- Summarize the main errors and their nature (e.g., common causes, affected components).
- Do not include raw log lines or technical stack traces.
- Keep the summary under 500 words.
- Use clear, professional language suitable for an engineering report.
- Focus on key patterns or recurring problems rather than individual details.
- Do not include any introductory phrases or prefixes like “Here is the summary” or “Summary:”. Output should only contain the summary text.
- Do NOT expand or explain any abbreviations or short forms. Keep them exactly as they appear
- If There are No/Empty Logs, Just Return `No logs to provide summary` and don't include any suffix or prefix phrases along with this
</REQUIREMENTS>

<OUTPUT FORMAT>
The output is directly used in a HTML Report, follow below instructions:
- Enclose any important keywords or sentence between `<b>`words to highlight`</b>`, so that these will appear as bold
- Always Generate the response as bullet points by enclosing each point between `<li>`Generated summary content`</li>`
</OUTPUT FORMAT>
"""

SUMMARY_GENERATION_PROMPT = """
<TASK>
You are given a list of error logs. Your goal is to generate a concise summary that captures the key issues across all logs.
</TASK>

<REQUIREMENTS>
- Summarize the main errors and their nature (e.g., common causes, affected components).
- Do not include raw log lines or technical stack traces.
- Keep the summary under 800 words.
- Use clear, professional language suitable for an engineering report.
- Focus on key patterns or recurring problems rather than individual details.
- Do not include any introductory phrases or prefixes like “Here is the summary” or “Summary:”. Output should only contain the summary text.
- Do NOT expand or explain any abbreviations or short forms. Keep them exactly as they appear
- If There are No/Empty Logs, Just Return `No logs to provide summary` and don't include any suffix or prefix phrases along with this
</REQUIREMENTS>
"""

PARENT_SUMMARY_GENERATION_PROMPT = """
<TASK>
You are given a list of summaries of error logs. Your goal is to generate a concise summary combining all these summaries that captures the key issues.
</TASK>

<REQUIREMENTS>
- Summarize the main errors and their nature (e.g., common causes, affected components).
- Do not include raw log lines or technical stack traces.
- Keep the summary less than 10 bullet points.
- Use clear, professional language suitable for an engineering report.
- Focus on key patterns or recurring problems rather than individual details.
- Do not include any introductory phrases or prefixes like “Here is the summary” or “Summary:”. Output should only contain the summary text.
- Do NOT expand or explain any abbreviations or short forms. Keep them exactly as they appear
- If There are No/Empty Logs, Just Return `No logs to provide summary` and don't include any suffix or prefix phrases along with this
</REQUIREMENTS>

<OUTPUT FORMAT>
The output is directly used in a HTML Report, follow below instructions:
- Enclose any important keywords or sentence between `<b> </b>`, so that these will appear as bold
- Always Generate the response as bullet points by enclosing each point between `<li>`Generated summary content`</li>`. Each unique point in summary should be a bullet point
</OUTPUT FORMAT>
"""

ERROR_LOGS_LIST = """
<LOGS>
Below are the error logs to generate summary of:
{logs}
</LOGS>
"""
