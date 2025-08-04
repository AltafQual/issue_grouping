import abc

import pandas as pd


class BaseDataLoader(abc.ABC):
    @abc.abstractmethod
    def load(self) -> pd.DataFrame:
        raise NotImplementedError


class ExcelLoader(BaseDataLoader):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def load(path=None, st_obj=None, **kwargs):
        if not path and not st_obj:
            raise ValueError("`path` is empty, please pass the Sheet path")
        if path:
            dataframe = pd.read_excel(path, **kwargs)
        else:
            dataframe = pd.read_excel(st_obj, **kwargs)
        if "result" not in dataframe.columns:
            raise KeyError("`result` key is not present in the dataframe, which is used to filter failed test cases")

        failure_filtered_df = dataframe[dataframe["result"] == "FAIL"]
        return failure_filtered_df if not failure_filtered_df.empty else None
