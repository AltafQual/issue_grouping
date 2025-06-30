import abc
import pandas as pd

class BaseDataLoader(abc.ABC):
    @abc.abstractmethod
    def load(self) -> pd.DataFrame:
        pass

class ExcelLoader(BaseDataLoader):
    def __init__(self, path: str = None) -> None:
        if not path:
            raise ValueError("`path` is empty, please pass the Sheet path")
        self.path = path
        super().__init__()

    def load(self, **kwargs):
        dataframe = pd.read_excel(self.path,**kwargs)
        if "result" not in dataframe.columns:
            raise KeyError("`result` key is not present in the dataframe, which is used to filter failed test cases")
        
        return dataframe[dataframe['result'] != "PASS"]

