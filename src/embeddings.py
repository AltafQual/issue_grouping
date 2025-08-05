import abc

from sentence_transformers import SentenceTransformer

from qgenie.integrations.langchain import QGenieEmbeddings
from src.constants import QGENEIE_API_KEY
from src.helpers import load_cached_model


class BaseEmbeddings(abc.ABC):
    cache_dir = "./models"

    @abc.abstractmethod
    def embed(self):
        raise NotImplementedError


class BGEM3Embeddings(BaseEmbeddings):
    name = "BAAI/bge-m3"

    def __init__(self):
        self.model = load_cached_model()
        if not self.model:
            self.model = SentenceTransformer(self.name, cache_folder=self.cache_dir)
        super().__init__()

    def embed(self, data: list):
        return self.model.encode(data)


class ALLMiniLMV6(BaseEmbeddings):
    name = "all-MiniLM-L6-v2"

    def __init__(self):
        self.model = load_cached_model()
        if not self.model:
            self.model = SentenceTransformer(self.name, cache_folder=self.cache_dir)
        super().__init__()

    def embed(self, data: list):
        return self.model.encode(data)


class QGenieBGEM3Embedding(BaseEmbeddings):
    name = "QGenie/bge-m3"

    def __init__(self):
        self.model = QGenieEmbeddings(model="bge-large", api_key=QGENEIE_API_KEY)
        super().__init__()

    def embed(self, data: list):
        return self.model.embed_documents(data)
