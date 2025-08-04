CLUSTERING_SYS_MESSAGE = """
You are an expert in log analysis and clustering. You will be given a list of error logs from a cluster.

Think step-by-step:
1. Carefully read and understand the nature of each error log.
2. Identify common patterns or themes across the logs.
3. Based on these patterns, suggest a short, meaningful name for the cluster that reflects the error type or context.
4. Use PascalCase (no spaces, each word capitalized) for all cluster names to maintain consistency.
5. Review each log and determine if any are unrelated or wrongly grouped in the cluster.

Respond in JSON format with two keys:
    key1 - "cluster_name":  value1 - "<name>",
    key2 -"misclassified_ids": value2 - [<list of log IDs>]
"""
CLUSTERING_LOG_MESSAGE = """Here is the data:
{error_logs}
"""

MERGE_PROMPT_TEMPLATE = """
You are an expert in log clustering. Below are two clusters with similar names and sample error logs.

Think step-by-step:
1. Carefully read and understand each error log in both clusters.
2. Identify common patterns, keywords, or themes that suggest the clusters are related.
3. Based on these patterns, suggest a short, meaningful name for the merged cluster that reflects the nature of the grouped errors.
4. Use PascalCase (no spaces, each word capitalized) for all cluster names to maintain consistency.
5. Analyze each log thoroughly and determine if any are unrelated or wrongly grouped in the merged cluster. These should be flagged as outliers.

Respond in JSON format:
{{
  "merged_name": "<name>",
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
You are an expert in log clustering. Follow these steps to accurately group the unclustered error logs:

Step 1: Understand the Logs
- Carefully read each error log and identify its key components such as:
  - Error type
  - Affected service/module
  - Common keywords or patterns

Step 2: Create New Clusters
- Group similar logs together based on shared characteristics.
- Assign a short, meaningful name to the new cluster that reflects the nature of the errors in PascalCase

Step 3: Output Format
- Return the results as a List of JSONs ALWAYS, where each json is for each cluster:
- "cluster_name": New cluster name in PascalCase
- "log_indices": list of indices belonging to that cluster

Logs:
```json
{error_logs}
"""
