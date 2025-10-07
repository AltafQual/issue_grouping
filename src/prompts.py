CLUSTER_NAMING_SYS_MESSAGE = """

You are an expert in error log naming. You will be given a list of error logs from a cluster.

Your task is to:
- Suggest a meaningful cluster name that reflects the entire error type and context.
- Use PascalCase formatting for the cluster name (no spaces, each word capitalized).
- Don't include name `Cluster` for example TimeoutErrorCluster (should be TimeoutError), InitDeinitErrorCluster (should be InitDeinitError) ...

Return the result strictly as a JSON as shown below. Do not include any explanation or extra text:

  {{
  "cluster_name": "<name in PascalCase>"
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
- Use PascalCase formatting for the cluster name (no spaces, each word capitalized).
- Identify any logs that appear misclassified or unrelated to that cluster and return there ids as `misclassified_ids`.

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
- Use PascalCase formatting for the cluster name (no spaces, each word capitalized).
- Identify any logs that appear unrelated or wrongly grouped and doesn't match the theme/context of the merged cluster. These should be flagged as outliers and indices should be returned as `outlier_indices` (INDEX SHOULD BE CORRECT IT IS VERY CRITICAL)

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
- Return the result strictly as a JSON list. Do not include any explanation or text

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
2. **Code related Issue**: all the error related to failure of test due to bug in code, indexing error, value error are some example
3. **SDK related Issue**: logs related to hardware failures, sub modules failed or any type of failure in legacy code. All the errors which doesn't belong to either environment/code will fall in this category

Return your response in the following JSON format and out of 3 failure only one can be true at any given instance:
{{
  "environment_issue": true/false,
  "code_failure": true/false,
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
- Use PascalCase for cluster_name (short, no spaces, each word capitalized). Do not include the word "Cluster" in the name and Make sure the cluster name always starts with VerifierFailed<issue details> and no ManyImages/ManyVerifierImages should be added
- indices must be the exact "index" values from the provided logs. INDEX SHOULD BE CORRECT IT IS VERY CRITICAL.
- Only include logs that strongly fit the chosen subcluster; leave unrelated logs for later passes.
- Accuracy enforcement: Clusters must be highly precise. Single-log subclusters are permitted only when that single log uniquely and correctly fits the subcluster theme; do not assign incorrect logs. Append a concise failure reason at the end of the PascalCase name (shortened and specific), e.g., VerifierFailedTimeout, VerifierFailedConfigMissing, VerifierFailedInitDeinit.
- Return strictly and only this JSON format (no extra text):
{{
  "cluster_name": "<name in PascalCase format: `VerifierFailed<sub cluster type>`>",
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
