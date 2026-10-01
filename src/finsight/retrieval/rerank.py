"""Optional reranking. A cross-encoder reads (question, chunk) TOGETHER, so it can judge relevance
better than first-stage retrieval, at the cost of an extra call. Default is off ("none"): the
hosted reranker is capped at 500 requests a month on the free plan, so every call is budgeted.
"""

from dataclasses import replace
from typing import Any

from finsight.config import Settings
from finsight.ingestion.embedder import TokenBudget
from finsight.schemas import Hit


class NoRerank:
    def rerank(self, question: str, hits: list[Hit], top_k: int) -> list[Hit]:
        return hits[:top_k]


class PineconeReranker:
    def __init__(self, settings: Settings, client: Any = None, budget: TokenBudget | None = None):
        self._s = settings
        self._client = client  # injectable for tests
        self.budget = budget or TokenBudget(
            settings.cache_dir / "rerank_usage.json",
            settings.rerank_request_budget,
            what="rerank request",
            setting="FINSIGHT_RERANK_REQUEST_BUDGET",
        )

    def _pc(self) -> Any:
        if self._client is None:
            if self._s.pinecone_api_key is None:
                raise RuntimeError("PINECONE_API_KEY is not set (add it to your local .env)")
            from pinecone import Pinecone

            self._client = Pinecone(api_key=self._s.pinecone_api_key.get_secret_value())
        return self._client

    def rerank(self, question: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        self.budget.check(1)  # raises BEFORE any call if the monthly cap would be exceeded
        result = self._pc().inference.rerank(
            model=self._s.rerank_model,
            query=question,
            documents=[h.text for h in hits],
            top_n=min(top_k, len(hits)),
            return_documents=False,
            parameters={"truncate": "END"},
        )
        self.budget.record(1)
        return [replace(hits[r.index], score=float(r.score)) for r in result.data]


def make_reranker(settings: Settings) -> NoRerank | PineconeReranker:
    return PineconeReranker(settings) if settings.rerank == "pinecone" else NoRerank()
