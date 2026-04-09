CLUSTER_NAMING_SYS_MESSAGE = """
<TASK>
You are an expert in error log naming. You will be given a list of error logs from a cluster. Your task is to suggest a single, highly precise, cause-based cluster name that accurately reflects the error type and context of the provided logs.
</TASK>

<REQUIREMENTS>
- Suggest a name that reflects the entire error type and context (e.g., include relevant operation/module and failure mode when clear).
- Give preference to device-related errors, platform-specific issues, DSP-related errors, GPU errors, aborted processes, and any specific keys in traceback when generating names.
- Names must be specific and descriptive, not vague. Prefer root-cause descriptors (e.g., Timeout, ConfigMissing, InitDeinit) over generic terms.
- Use PascalCase formatting for the cluster name (no spaces, each word capitalized).
- Limit names to a maximum of 5 words in PascalCase format.
- Do not include the word "Cluster" in the name (e.g., TimeoutErrorCluster -> TimeoutError, InitDeinitErrorCluster -> InitDeinitError).
- Prohibited generic names: Never use generic or quantity-based names such as "ManyImages", "ManyVerifierImages", "VerifierFailedImagesList", "VerifierFailedManyImages", "ImagesList", or any name containing "Many" or "Multiple". Names must not describe volume; they must describe the failure cause.
</REQUIREMENTS>

<OUTPUT_FORMAT>
Return the result strictly as a JSON object as shown below. Do not include any explanation or extra text:
{{
  "cluster_name": "<name in PascalCase or empty string>"
}}
</OUTPUT_FORMAT>
"""
CLUSTER_NAMING_LOG_MESSAGE = """
<LOGS>
Here are the logs:
{logs}
</LOGS>
"""
CLUSTERING_SYS_MESSAGE = """
<TASK>
You are an expert in log analysis and clustering. You will be given a list of error logs from a cluster. Your task is to analyze the logs, suggest a meaningful cluster name, and identify any misclassified entries.
</TASK>

<STEPS>
1. Thoroughly analyze the logs and identify a common theme or pattern across them.
2. Suggest a meaningful cluster name that reflects the entire error type and context.
3. Identify any logs that appear misclassified or unrelated to the cluster and collect their IDs.
</STEPS>

<REQUIREMENTS>
- Give preference to device-related errors, platform-specific issues, DSP-related errors, GPU errors, aborted processes, Ops failures, and any specific keys in traceback when generating names.
- Use PascalCase formatting for the cluster name (no spaces, each word capitalized).
- Limit names to a maximum of 5 words in PascalCase format.
- Do not include the word "Cluster" in the name (e.g., TimeoutErrorCluster -> TimeoutError, InitDeinitErrorCluster -> InitDeinitError).
- Prohibited generic names: Never use generic or quantity-based names such as "ManyImages", "ManyVerifierImages", "VerifierFailedImagesList", "VerifierFailedManyImages", "ImagesList", or any name containing "Many" or "Multiple". Names must not describe volume; they must describe the failure cause.
</REQUIREMENTS>

<OUTPUT_FORMAT>
Return the result strictly as a JSON object as shown below. Do not include any explanation or extra text:
{{
  "cluster_name": "<name in PascalCase>",
  "misclassified_ids": [<list of log IDs that are misclassified, otherwise empty list>]
}}
</OUTPUT_FORMAT>
"""
CLUSTERING_LOG_MESSAGE = """
<DATA>
Here is the data:
{error_logs}
</DATA>
"""

MERGE_PROMPT_TEMPLATE = """
<TASK>
You are an expert in log clustering. You will be given two clusters with similar names and sample error logs. Your task is to analyze both clusters, propose a unified name for the merged cluster, and identify any outlier logs that do not belong.
</TASK>

<STEPS>
1. Thoroughly analyze the logs from both clusters and identify a common theme or pattern.
2. Suggest a short, meaningful name for the merged cluster that reflects the nature of the grouped errors.
3. Identify any logs that appear unrelated or wrongly grouped and do not match the theme/context of the merged cluster — flag these as outliers and return their indices.
</STEPS>

<REQUIREMENTS>
- Give preference to device-related errors, platform-specific issues, DSP-related errors, GPU errors, aborted processes, Ops failures, and any specific keys in traceback when generating names.
- Use PascalCase formatting for the cluster name (no spaces, each word capitalized).
- Limit names to a maximum of 5 words in PascalCase format.
- Do not include the word "Cluster" in the name (e.g., TimeoutErrorCluster -> TimeoutError, InitDeinitErrorCluster -> InitDeinitError).
- Prohibited generic names: Never use generic or quantity-based names such as "ManyImages", "ManyVerifierImages", "VerifierFailedImagesList", "VerifierFailedManyImages", "ImagesList", or any name containing "Many" or "Multiple". Names must not describe volume; they must describe the failure cause.
- The `outlier_indices` field is CRITICAL — indices must be exactly correct.
</REQUIREMENTS>

<OUTPUT_FORMAT>
Return the result strictly as a JSON object as shown below. Do not include any explanation or extra text:
{{
  "merged_name": "<name in PascalCase>",
  "outlier_indices": [<list of indices>]
}}
</OUTPUT_FORMAT>

<INPUT>
Cluster A (ID: {id_a}):
```json
{logs_a}

Cluster B (ID: {id_b}):
```json
{logs_b}
</INPUT>
"""

RECLUSTERING_PROMPT = """
<TASK>
You are an expert in log clustering. You will be given a list of unclustered error logs. Your task is to group similar logs together into coherent clusters and assign each cluster a meaningful name.
</TASK>

<STEPS>
1. Analyze all provided logs and identify groups of similar entries based on shared characteristics such as error type and affected module.
2. For each identified group, collect the exact indices of the logs that belong to it.
3. Assign each group a meaningful cluster name that reflects the error type and context.
</STEPS>

<REQUIREMENTS>
- Return the correct index of each error log to group together as `log_indices` — INDEX MUST BE EXACTLY CORRECT, THIS IS CRITICAL.
- Give preference to device-related errors, platform-specific issues, DSP-related errors, GPU errors, aborted processes, Ops failures, and any specific keys in traceback when generating names.
- Limit names to a maximum of 5 words in PascalCase format.
- Do not include the word "Cluster" in the name (e.g., TimeoutErrorCluster -> TimeoutError, InitDeinitErrorCluster -> InitDeinitError).
- Prohibited generic names: Never use generic or quantity-based names such as "ManyImages", "ManyVerifierImages", "VerifierFailedImagesList", "VerifierFailedManyImages", "ImagesList", or any name containing "Many" or "Multiple". Names must not describe volume; they must describe the failure cause.
</REQUIREMENTS>

<OUTPUT_FORMAT>
Return the result strictly as a JSON list. Do not include any explanation or extra text. Each object in the list must follow this exact format:
{{
  "cluster_name": "<name in PascalCase (no spaces, each word capitalized)>",
  "log_indices": [<list of indices belonging to that cluster>]
}}
</OUTPUT_FORMAT>

<LOGS>
```json
{error_logs}
</LOGS>
"""
CLASSIFY_CLUSTER_TYPE_SYS_MESSAGE = """
<TASK>
You are an expert in log analysis and classification. You will be given error logs along with a cluster name. Your task is to analyze the logs and classify them into exactly one of three issue categories.
</TASK>

<CATEGORIES>
1. **Environment Issue**: Logs indicating timeouts during operations (connection timeouts, request timeouts, process timeouts) or missing environment variables/configuration.
2. **Setup Related Issue**: Logs related to any of the following:
   - No such file or directory
   - Failed to open input list
   - Model Not Found
   - Has bad ELF magic
   - Failed to open input file
   - Cannot unpack non-iterable NoneType object
   - ValueError: cannot reshape array
   - Could not find matching record in ADB logs
   - The selected runtime is not available on this platform
3. **SDK Related Issue**: Logs related to hardware failures, sub-module failures, or any type of failure in legacy code. Verifier Failed related issues also fall under this category.
</CATEGORIES>

<REQUIREMENTS>
- Classify the logs into exactly one of the three categories above.
- Only one category flag can be `true` at any given instance.
</REQUIREMENTS>

<OUTPUT_FORMAT>
Return the result strictly as a JSON object as shown below. Do not include any explanation or extra text:
{{
  "env_issue": true/false,
  "setup_issue": true/false,
  "sdk_issue": true/false
}}
</OUTPUT_FORMAT>
"""
CLASSIFY_CLUSTER_TYPE_LOG_MESSAGE = """
<INPUT>
Cluster name: {cluster_name}

Below are the logs to classify:
{logs}
</INPUT>
"""
SUBCLUSTER_VERIFIER_FAILED_SYS_MESSAGE = """
<TASK>
You are an expert in sub-clustering verifier-failed logs. You will iteratively process logs in batches (up to 50 per batch) and maintain or update the subcluster registry provided as `previous_clusters`. Your task is to assign each batch of logs to an appropriate subcluster — either an existing one or a newly created one.
</TASK>

<STEPS>
1. **First batch** (`previous_clusters` is empty): Discover coherent subclusters from the provided logs based on shared operation (ops), subsystem/module, error type, and context. In this pass, return exactly one subcluster (the most coherent group) with its name and indices.
2. **Subsequent batches** (`previous_clusters` provided):
   - Evaluate the current batch against all existing subclusters.
   - If some logs clearly belong to one existing subcluster, return that `subcluster_name` and the indices to add.
   - If no existing subcluster matches, create a new subcluster, return its name and the indices that belong to it.
3. Always return an updated `previous_clusters` dictionary that merges the returned indices into the chosen subcluster. If a new subcluster is created, add it to `previous_clusters`.
</STEPS>

<REQUIREMENTS>
- Give preference to device-related errors, platform-specific issues, DSP-related errors, GPU errors, aborted processes, Ops failures, and any specific keys in traceback when generating names.
- Use PascalCase for `cluster_name` (short, no spaces, each word capitalized). Do not include the word "Cluster" in the name.
- The `cluster_name` must always start with `VerifierFailed` followed by a concise, specific failure descriptor (e.g., VerifierFailedTimeout, VerifierFailedConfigMissing, VerifierFailedInitDeinit).
- Limit names to a maximum of 5 words in PascalCase format.
- `indices` must be the exact "index" values from the provided logs — INDEX MUST BE EXACTLY CORRECT, THIS IS CRITICAL.
- Only include logs that strongly fit the chosen subcluster; leave unrelated logs for later passes.
- Accuracy enforcement: Clusters must be highly precise. Single-log subclusters are permitted only when that single log uniquely and correctly fits the subcluster theme; do not assign incorrect logs.
- Prohibited generic names: NEVER use generic or quantity-based names such as "ManyImages", "ManyVerifierImages", "VerifierFailedManyImages", "VerifierFailedImages", "ImagesList", or any name containing "Many" or "Multiple". Names must describe the root cause or failure mode, not the volume of items.
- If you cannot confidently determine a specific, cause-based subcluster name, return an empty string `""` for `cluster_name` and an empty list for `indices`. In that case, do not update or create any subcluster; such logs will be handled in a later pass.
</REQUIREMENTS>

<OUTPUT_FORMAT>
Return strictly and only this JSON object. Do not include any explanation or extra text:
{{
  "cluster_name": "<name in PascalCase format: `VerifierFailed<SubClusterType>` or empty string>",
  "indices": [<list of indices>],
  "previous_clusters": {{ "<SubclusterName>": [<indices>], ... }}
}}
</OUTPUT_FORMAT>
"""

SUBCLUSTER_VERIFIER_FAILED_LOG_MESSAGE = """
<INPUT>
Existing subclusters (previous_clusters):
{previous_clusters}

Current batch logs:
{error_logs}
</INPUT>
"""

ERROR_SUMMARIZATION_PROMPT = """
<TASK>
You are given one or more error logs. Your task is to generate a highly descriptive, technically accurate summary that clearly explains the nature of the errors and the underlying issues indicated by the logs.
</TASK>

<REQUIREMENTS>
- Provide a clear and detailed explanation of what each error fundamentally represents (e.g., data type mismatch, missing configuration, dependency failure).
- Describe the probable root cause of each error type, based strictly on patterns or signals present in the logs.
- Clearly identify the impacted components, modules, workflows, or system boundaries suggested by log context.
- Explain the operational impact or consequences of the errors (e.g., failed requests, interrupted background tasks, serialization failures).
- Highlight recurring patterns, repeated failures, or systemic issues indicated across multiple logs.
- Avoid vague, generic descriptions such as “critical error”, “application failed”, “something went wrong”.
- Do NOT include stack traces, raw log lines, or filenames in the final summary.
- Do NOT expand or explain abbreviations; keep all abbreviations exactly as they appear.
- Maintain strict factual accuracy; do not invent modules, components, or behavior that cannot be reasonably inferred from logs.
- If no logs are provided or input is empty, return only: `No logs to provide summary`
- Do NOT add any prefixes or suffixes around the output.
- Keep total output under 800 words.
- Use a polished, professional tone suitable for engineering leadership.
</REQUIREMENTS>

<OUTPUT FORMAT>
- Produce the summary ONLY as bullet points.
- Each bullet point MUST strictly follow this exact structure:
  <li>SUMMARY CONTENT GOES HERE</li>
- The summary content MUST be placed inside the opening <li> and closing </li> tags.
- Do NOT generate empty <li></li> tags.
- Do NOT place any text outside of <li>...</li> tags.
- Bold important keywords or important phrases using: <b> ... </b> within the <li> tags only.
- Do NOT output any wrapper elements such as <ul>, <ol>, or <div>.
- If any content cannot be wrapped inside <li> tags, it MUST NOT be generated.
</OUTPUT FORMAT>
"""

SUMMARY_GENERATION_PROMPT = """
<TASK>
You are given one or more error logs. Your task is to generate a highly descriptive, technically accurate summary that clearly explains the nature of the errors and the underlying issues indicated by the logs.
</TASK>

<REQUIREMENTS>
- Provide a clear and detailed explanation of what each error fundamentally represents (e.g., data type mismatch, missing configuration, dependency failure).
- Describe the probable root cause of each error type, based strictly on patterns or signals present in the logs.
- Clearly identify the impacted components, modules, workflows, or system boundaries suggested by log context.
- Explain the operational impact or consequences of the errors (e.g., failed requests, interrupted background tasks, serialization failures).
- Highlight recurring patterns, repeated failures, or systemic issues indicated across multiple logs.
- Avoid vague, generic descriptions such as “critical error”, “application failed”, “something went wrong”.
- Do NOT include stack traces, raw log lines, or filenames in the final summary.
- Do NOT expand or explain abbreviations; keep all abbreviations exactly as they appear.
- Maintain strict factual accuracy; do not invent modules, components, or behavior that cannot be reasonably inferred from logs.
- If no logs are provided or input is empty, return only: `No logs to provide summary`
- Do NOT add any prefixes or suffixes around the output.
- Keep total output under 800 words.
- Use a polished, professional tone suitable for engineering leadership.
</REQUIREMENTS>
"""

PARENT_SUMMARY_GENERATION_PROMPT = """
<TASK>
You are given a list of summaries of error logs. Your goal is to generate a concise summary combining all these summaries that captures the key issues.
</TASK>

<REQUIREMENTS>
- Provide a clear and detailed explanation of what each error fundamentally represents (e.g., data type mismatch, missing configuration, dependency failure).
- Describe the probable root cause of each error type, based strictly on patterns or signals present in the logs.
- Clearly identify the impacted components, modules, workflows, or system boundaries suggested by log context.
- Explain the operational impact or consequences of the errors (e.g., failed requests, interrupted background tasks, serialization failures).
- Highlight recurring patterns, repeated failures, or systemic issues indicated across multiple logs.
- Avoid vague, generic descriptions such as “critical error”, “application failed”, “something went wrong”.
- Do NOT include stack traces, raw log lines, or filenames in the final summary.
- Do NOT expand or explain abbreviations; keep all abbreviations exactly as they appear.
- Maintain strict factual accuracy; do not invent modules, components, or behavior that cannot be reasonably inferred from logs.
- If no logs are provided or input is empty, return only: `No logs to provide summary`
- Do NOT add any prefixes or suffixes around the output.
- Keep the summary less than 10 bullet points.
- Use a polished, professional tone suitable for engineering leadership.
</REQUIREMENTS>

<OUTPUT FORMAT>
- Produce the summary ONLY as bullet points.
- Each bullet point MUST strictly follow this exact structure:
  <li>SUMMARY CONTENT GOES HERE</li>
- The summary content MUST be placed inside the opening <li> and closing </li> tags.
- Do NOT generate empty <li></li> tags.
- Do NOT place any text outside of <li>...</li> tags.
- Bold important keywords or important phrases using: <b> ... </b> within the <li> tags only.
- Do NOT output any wrapper elements such as <ul>, <ol>, or <div>.
- If any content cannot be wrapped inside <li> tags, it MUST NOT be generated.
</OUTPUT FORMAT>
"""

SHORT_PARENT_SUMMARY_GENERATION_PROMPT = """
<TASK>
You are given a list of summaries of error logs. Your goal is to generate a concise summary combining all these summaries that captures the key issues.
</TASK>

<REQUIREMENTS>
- Summarize the main errors and their nature (e.g., common causes, affected components).
- Do not include raw log lines or technical stack traces.
- Keep the summary less than 5 bullet points.
- Use clear, professional language suitable for an engineering report.
- Focus on key patterns or recurring problems rather than individual details.
- Do not include any introductory phrases or prefixes like “Here is the summary” or “Summary:”. Output should only contain the summary text.
- Do NOT expand or explain any abbreviations or short forms. Keep them exactly as they appear
- If There are No/Empty Logs, Just Return `No logs to provide summary` and don't include any suffix or prefix phrases along with this
</REQUIREMENTS>

<OUTPUT FORMAT>
- Produce the summary ONLY as bullet points.
- Each bullet point MUST strictly follow this exact structure:
  <li>SUMMARY CONTENT GOES HERE</li>
- The summary content MUST be placed inside the opening <li> and closing </li> tags.
- Do NOT generate empty <li></li> tags.
- Do NOT place any text outside of <li>...</li> tags.
- Bold important keywords or important phrases using: <b> ... </b> within the <li> tags only.
- Do NOT output any wrapper elements such as <ul>, <ol>, or <div>.
- If any content cannot be wrapped inside <li> tags, it MUST NOT be generated.
</OUTPUT FORMAT>
"""

ERROR_LOGS_LIST = """
<LOGS>
Below are the error logs to generate summary of:
{logs}
</LOGS>
"""
