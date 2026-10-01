"""All configuration comes from environment variables (prefix FINSIGHT_) or a local .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SEC_HARD_MAX_RPS = 10.0  # SEC fair-access limit; never exceeded regardless of config


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FINSIGHT_", env_file=".env", extra="ignore")

    # No default on purpose: the SEC requires a real contact, so we fail fast if it's unset.
    sec_user_agent: str
    sec_max_rps: float = Field(default=5.0, gt=0)

    data_dir: Path = Path("data")
    tickers: list[str] = ["AAPL", "MSFT", "NVDA"]
    forms: list[str] = ["10-K", "10-Q"]
    years: int = 3

    # Vector store: Pinecone (hosted). The key comes from the environment, never from code.
    pinecone_api_key: SecretStr | None = Field(default=None, validation_alias="PINECONE_API_KEY")
    pinecone_index: str = "finsight"
    pinecone_namespace: str = "filings"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"  # the only region on the free (Starter) plan

    # Hosted embeddings (Pinecone Inference). Hybrid search needs the dotproduct metric.
    dense_model: str = "llama-text-embed-v2"
    dense_dim: int = 1024  # llama-text-embed-v2 supports 384, 512, 768, 1024, 2048
    sparse_model: str = "pinecone-sparse-english-v0"
    embed_batch: int = Field(default=64, ge=1, le=96)
    # Hard stop for hosted-embedding tokens per calendar month. The free plan allows 5M.
    embed_token_budget: int = 3_500_000

    # Retrieval (step 2)
    retrieve_k: int = 30  # candidates pulled from Pinecone
    final_k: int = 6  # chunks that go into the prompt
    hybrid_alpha: float = Field(default=0.5, ge=0, le=1)  # 1 = dense only, 0 = sparse only
    # Sparse (keyword) scores are ~30x larger than dense ones (measured: median top-1 of 16.1 vs
    # 0.51), so without this, alpha < ~0.95 is effectively sparse-only. Scaling the sparse query
    # brings both onto one scale so alpha behaves as a real balance.
    sparse_score_scale: float = Field(default=0.03, gt=0)
    # "none" = trust the hybrid ranking. "pinecone" = hosted cross-encoder (free plan: 500/month).
    rerank: Literal["none", "pinecone"] = "none"
    rerank_model: str = "bge-reranker-v2-m3"
    rerank_request_budget: int = 400  # hard stop per calendar month

    # Generation: a cheap Anthropic model by default; any LiteLLM model string works.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    llm_model: str = "anthropic/claude-haiku-4-5-20251001"
    llm_max_tokens: int = 700
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2  # hard cap on retries for cost control
    # Optional stronger model used ONLY for the final answer (unset = same as llm_model).
    synthesis_model: str | None = None
    verify: bool = True  # check the answer against its evidence before returning it
    max_revisions: int = Field(default=1, ge=0, le=3)  # hard cap on revise loops (cost control)
    request_timeout_s: float = 120.0  # whole /ask request, incl. verify/revise calls
    sql_timeout_s: float = 5.0  # one text-to-SQL query

    # A2A (step 7). Unset = run that capability in-process. Agents are local-only by default.
    a2a_retrieval_url: str | None = None  # e.g. http://127.0.0.1:9101
    a2a_facts_url: str | None = None  # e.g. http://127.0.0.1:9102
    a2a_verifier_url: str | None = None  # e.g. http://127.0.0.1:9103
    a2a_timeout_s: float = 60.0  # one agent-to-agent call
    a2a_max_payload_bytes: int = 512_000  # cap on any request or response body we accept

    tracing: bool = True  # servers write spans to <data_dir>/traces (never prompts or filing text)

    # Optional LangSmith export (off by default). Standard LangSmith names, read from .env.
    # Unlike the local trace files it sends CONTENT unless the hide flags are set.
    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: SecretStr | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="finsight", validation_alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str | None = Field(default=None, validation_alias="LANGSMITH_ENDPOINT")
    langsmith_hide_inputs: bool = Field(default=False, validation_alias="LANGSMITH_HIDE_INPUTS")
    langsmith_hide_outputs: bool = Field(default=False, validation_alias="LANGSMITH_HIDE_OUTPUTS")

    @field_validator("sec_user_agent")
    @classmethod
    def _needs_contact(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("FINSIGHT_SEC_USER_AGENT must include a contact email (SEC policy)")
        return v

    @field_validator("sec_max_rps")
    @classmethod
    def _cap_rps(cls, v: float) -> float:
        return min(v, SEC_HARD_MAX_RPS)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def trace_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def duckdb_path(self) -> Path:
        return self.data_dir / "finsight.duckdb"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from env
