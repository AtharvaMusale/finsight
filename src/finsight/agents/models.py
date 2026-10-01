"""Wire schemas for the A2A agents. They bound every field, because anything that arrives from
another process is untrusted: request args are validated by the agent that receives them, and
responses are validated again by the caller before they touch the graph."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from finsight.schemas import Hit
from finsight.tools.sql_tool import SqlResult

Question = Annotated[str, Field(min_length=3, max_length=500)]
Cell = str | int | float | bool | None
SECTION_PATTERN = r"^Item \d{1,2}[A-C]?$"


class HitModel(BaseModel):
    chunk_id: str = Field(max_length=200)
    text: str = Field(max_length=20_000)
    ticker: str = Field(max_length=10)
    form_type: str = Field(max_length=10)
    fiscal_year: int = Field(ge=1990, le=2100)
    section: str = Field(max_length=50)
    accession_no: str = Field(max_length=40)
    score: float = 0.0

    def to_hit(self) -> Hit:
        return Hit(**self.model_dump())


class SqlModel(BaseModel):
    sql: str = Field(max_length=5_000)
    columns: list[Annotated[str, Field(max_length=100)]] = Field(max_length=50)
    rows: list[Annotated[list[Cell], Field(max_length=50)]] = Field(max_length=200)
    truncated: bool = False
    warnings: list[Annotated[str, Field(max_length=300)]] = Field(default=[], max_length=10)

    def to_result(self) -> SqlResult:
        return SqlResult(self.sql, self.columns, self.rows, self.truncated, self.warnings)

    @classmethod
    def from_result(cls, r: SqlResult) -> "SqlModel":
        return cls(
            sql=r.sql, columns=r.columns, rows=r.rows, truncated=r.truncated, warnings=r.warnings
        )


class IssueModel(BaseModel):
    kind: Literal["unsupported_number", "invalid_citation", "uncited", "unsupported_claim"]
    detail: str = Field(max_length=300)
    claim: str = Field(default="", max_length=300)


# ---- requests (validated by the receiving agent)


class SearchArgs(BaseModel):
    query: Question
    tickers: list[str] = Field(default=[], max_length=3)
    fiscal_year: int | None = Field(default=None, ge=2000, le=2100)
    form_type: Literal["10-K", "10-Q"] | None = None
    section: str | None = Field(default=None, pattern=SECTION_PATTERN)
    limit: int = Field(default=30, ge=1, le=50)


class FactsArgs(BaseModel):
    question: Question


class VerifyArgs(BaseModel):
    answer: str = Field(min_length=1, max_length=20_000)
    hits: list[HitModel] = Field(default=[], max_length=50)
    sql: SqlModel | None = None


class AskArgs(BaseModel):
    question: Question


# ---- responses (validated by the calling side)


class HitsResult(BaseModel):
    hits: list[HitModel] = Field(max_length=50)


class VerificationResult(BaseModel):
    issues: list[IssueModel] = Field(max_length=100)
    claims_checked: int = Field(ge=0, le=1000)
    critic_ran: bool
