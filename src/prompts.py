CLUSTERING_SYS_MESSAGE = """

You are an expert in log analysis and clustering. You will be given a list of error logs from a cluster.

Your task is to:
- Analyze the logs and identify a common theme or pattern.
- Suggest a short, meaningful cluster name that reflects the error type or context.
- Use PascalCase formatting for the cluster name (no spaces, each word capitalized).
- Identify any logs that appear misclassified or unrelated to the cluster.

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
- Analyze the logs in both clusters and identify common patterns or themes.
- Suggest a short, meaningful name for the merged cluster that reflects the nature of the grouped errors.
- Use PascalCase formatting for the cluster name (no spaces, each word capitalized).
- Identify any logs that appear unrelated or wrongly grouped in the merged cluster. These should be flagged as outliers.

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
- Group similar logs together based on shared characteristics such as error type, affected module, or common keywords.
- Assign a short, meaningful name to each new cluster using PascalCase formatting (no spaces, each word capitalized).
- Return the result strictly as a JSON list. Do not include any explanation or extra text

Each JSON object must follow this format:
- "cluster_name": New cluster name in PascalCase
- "log_indices": list of indices belonging to that cluster

Logs:
```json
{error_logs}
"""
