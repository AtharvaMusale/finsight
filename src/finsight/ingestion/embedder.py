"""Hosted embeddings via Pinecone Inference: dense (meaning) + sparse (exact keywords).

Both are stored on every chunk so hybrid search can weigh them at query time. Nothing is
downloaded or run locally. Because hosted tokens are metered (the free plan allows 5M a month),
every batch is checked against a monthly budget ledger BEFORE it is sent.
"""

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from finsight.config import Settings

log = logging.getLogger("finsight.embed")
Kind = Literal["passage", "query"]


@dataclass
class SparseVec:
    indices: list[int]
    values: list[float]


class BudgetExceeded(RuntimeError):
    """The monthly hosted-embedding token budget would be exceeded; nothing was sent."""


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1  # rough English average; only used as a pre-send safety check


class TokenBudget:
    """Per-calendar-month usage ledger (tokens, or requests) stored in a small JSON file."""

    def __init__(
        self,
        path: Path,
        limit: int,
        what: str = "embedding token",
        setting: str = "FINSIGHT_EMBED_TOKEN_BUDGET",
    ):
        self._path, self._limit, self._what, self._setting = path, limit, what, setting

    def _month(self) -> str:
        return date.today().strftime("%Y-%m")

    def used(self) -> int:
        try:
            return int(json.loads(self._path.read_text()).get(self._month(), 0))
        except (OSError, ValueError):
            return 0

    def check(self, estimated: int) -> None:
        if self.used() + estimated > self._limit:
            raise BudgetExceeded(
                f"{self._what} budget: {self.used():,} used + ~{estimated:,} needed exceeds "
                f"{self._limit:,} for {self._month()} ({self._setting})"
            )

    def record(self, tokens: int) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ledger = json.loads(self._path.read_text())
        except (OSError, ValueError):
            ledger = {}
        ledger[self._month()] = self.used() + tokens
        self._path.write_text(json.dumps(ledger))


def _get(obj: Any, name: str) -> Any:
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _retryable(e: BaseException) -> bool:
    # Bad input, an exhausted budget or a bug will not fix itself; only transient errors retry.
    return not isinstance(e, BudgetExceeded | ValueError | TypeError | KeyError)


class Embedder:
    def __init__(self, settings: Settings, client: Any = None, budget: TokenBudget | None = None):
        self._s = settings
        self._client = client  # injectable for tests; otherwise created lazily on first use
        self.budget = budget or TokenBudget(
            settings.cache_dir / "embed_usage.json", settings.embed_token_budget
        )

    def _pc(self) -> Any:
        if self._client is None:
            if self._s.pinecone_api_key is None:
                raise RuntimeError("PINECONE_API_KEY is not set (add it to your local .env)")
            from pinecone import Pinecone

            self._client = Pinecone(api_key=self._s.pinecone_api_key.get_secret_value())
        return self._client

    @retry(
        stop=stop_after_attempt(3),  # hard cap on retries for cost control
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception(_retryable),
        reraise=True,
    )
    def _embed_batch(self, model: str, texts: list[str], params: dict) -> Any:
        return self._pc().inference.embed(model=model, inputs=texts, parameters=params)

    def _embed(self, model: str, texts: list[str], params: dict) -> list[Any]:
        out: list[Any] = []
        for i in range(0, len(texts), self._s.embed_batch):
            batch = texts[i : i + self._s.embed_batch]
            self.budget.check(sum(estimate_tokens(t) for t in batch))
            resp = self._embed_batch(model, batch, params)
            self.budget.record(int(_get(_get(resp, "usage"), "total_tokens")))
            out.extend(_get(resp, "data"))
        return out

    def embed_dense(self, texts: list[str], kind: Kind = "passage") -> list[list[float]]:
        params = {"input_type": kind, "truncate": "END", "dimension": self._s.dense_dim}
        return [list(_get(e, "values")) for e in self._embed(self._s.dense_model, texts, params)]

    def embed_sparse(self, texts: list[str], kind: Kind = "passage") -> list[SparseVec]:
        params: dict = {"input_type": kind, "truncate": "END"}
        if kind == "passage":
            params["max_tokens_per_sequence"] = 2048  # our chunks can exceed the 512 default
        return [
            SparseVec(list(_get(e, "sparse_indices")), list(_get(e, "sparse_values")))
            for e in self._embed(self._s.sparse_model, texts, params)
        ]
