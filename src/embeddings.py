import abc
import asyncio
import time

from langchain.embeddings.base import Embeddings

from qgenie.integrations.langchain import QGenieEmbeddings
from src.constants import QGENEIE_API_KEY
from src.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class QGenieBGEM3Embedding(Embeddings):
    name = "qgenie_embedd"

    def __init__(self):
        self.model = QGenieEmbeddings(model=self.name, api_key=QGENEIE_API_KEY)
        super().__init__()

    def _retry_sync(self, func, *args, **kwargs):
        for attempt in range(3):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise Exception(f"Failed to generate embeddings after 3 attempts: {e}")
                time.sleep(5)

    async def _retry_async(self, func, *args, **kwargs):
        for attempt in range(3):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Async attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise Exception(f"Failed to generate embeddings after 3 attempts: {e}")
                await asyncio.sleep(5)

    def embed(self, data: list):
        return self._retry_sync(self.model.embed_documents, data)

    def embed_documents(self, data: list[str]) -> list[list[float]]:
        return self._retry_sync(self.model.embed_documents, data)

    def embed_query(self, text: str) -> list[float]:
        return self._retry_sync(self.model.embed_query, text)

    async def aembed(self, data: list):
        return await self._retry_async(self.model.aembed_documents, data)

    async def aembed_query(self, text: str):
        return await self._retry_async(self.model.aembed_query, text)
