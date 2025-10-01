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

Return your response in the following JSON format:
{{
  "environment_issue": true/false,
  "code_failure": true/false,
  "sdk_issue": true/false,
  "reasoning": "Brief explanation of why this classification was chosen"
}}
"""
CLASSIFY_CLUSTER_TYPE_LOG_MESSAGE = """
Below are the logs to classify:
{logs}
"""
