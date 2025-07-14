# -*- coding: utf-8 -*-
from typing import List, Any
from langchain.embeddings.base import Embeddings

class BaseEmbeddingModel(Embeddings):
    def __init__(self, model: Any, **kwargs: Any):
        self.model = model

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Internal method to embed texts."""
        raise NotImplementedError

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self._embed([text])[0]
