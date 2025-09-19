import abc

from langchain.embeddings.base import Embeddings

from qgenie.integrations.langchain import QGenieEmbeddings
from src.constants import QGENEIE_API_KEY


class QGenieBGEM3Embedding(Embeddings):
    name = "qgenie_embedd"

    def __init__(self):
        self.model = QGenieEmbeddings(model=self.name, api_key=QGENEIE_API_KEY)
        super().__init__()

    def embed(self, data: list):
        return self.model.embed_documents(data)

    def embed_documents(self, data: list[str]) -> list[list[float]]:
        return self.model.embed_documents(data)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)

    async def aembed(self, data: list):
        return await self.model.aembed_documents(data)

    async def aembed_query(self, text: str):
        return await self.model.aembed_query(text)
