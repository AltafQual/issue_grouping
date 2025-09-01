import abc

from qgenie.integrations.langchain import QGenieEmbeddings
from src.constants import QGENEIE_API_KEY


class BaseEmbeddings(abc.ABC):
    cache_dir = "./models"

    @abc.abstractmethod
    def embed(self):
        raise NotImplementedError


class QGenieBGEM3Embedding(BaseEmbeddings):
    name = "qgenie_embedd"

    def __init__(self):
        self.model = QGenieEmbeddings(model=self.name, api_key=QGENEIE_API_KEY)
        super().__init__()

    def embed(self, data: list):
        return self.model.embed_documents(data)

    async def aembed(self, data: list):
        return await self.model.aembed_documents(data)

    async def aembed_query(self, text: str):
        return await self.model.aembed_query(text)
