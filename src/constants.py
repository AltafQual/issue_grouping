QGENEIE_API_KEY = "bad93c9e-c0da-4b3b-b0de-6dfb7f3db1c3"


class DataFrameKeys:
    embeddings_key: str = "embeddings"
    preprocessed_text_key: str = "preprocessed_reason"
    cluster_type_int: str = "int_cluster"
    cluster_name: str = "clusters"
    bins: str = "bins"
    error_logs_length: str = "logs_length"


class ClusterSpecificKeys:
    non_grouped_key: int = -1
