"""A2A plumbing shared by every FinSight agent: the server side and the client side.

Wire format. One JSON envelope in a text part (media type application/json), one back:
  request   {"trace_id": "...", "args": {...}}
  response  {"ok": true,  "trace_id": "...", "result": {...}}
            {"ok": false, "trace_id": "...", "error": "..."}
Why text, not an A2A data part: data parts are protobuf Structs, which store every number as a
float, so fiscal_year 2025 would come back as 2025.0. JSON text keeps ints as ints.

Why a single Message reply, not a Task: every skill here is one bounded call (seconds), so there is
no long-running state to track. Failures are returned as ok=false payloads, never as exceptions.

Trust. Everything a remote agent returns is untrusted data: bounded in size, re-validated against a
schema by the caller, and error text from the remote is truncated before anyone sees it.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

import httpx
from a2a.client import A2AClientError, ClientConfig, create_client
from a2a.helpers.proto_helpers import get_message_text, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue_v2 import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Role,
    SendMessageRequest,
)
from pydantic import BaseModel, ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from finsight.tracing import adopt, current_span_id, span

log = logging.getLogger("finsight.a2a")

RPC_PATH = "/rpc"
_TRACE_OK = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class AgentError(Exception):
    """A failure the caller can act on. The message is safe to return over the wire."""


class RemoteAgentError(RuntimeError):
    """A remote agent was unreachable, timed out, or answered badly. Message is safe to show."""


def _short(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:5])


def clean_remote_text(text: object, limit: int = 300) -> str:
    """Error text from another process is untrusted: one line, bounded."""
    return re.sub(r"\s+", " ", str(text)).strip()[:limit]


# ------------------------------------------------------------------ server side


class SkillExecutor(AgentExecutor):
    """Turns one envelope into one handler call, with validation, a timeout and safe errors."""

    def __init__(
        self,
        handler: Callable[[BaseModel, str], dict],
        args_model: type[BaseModel],
        *,
        skill: str,
        timeout_s: float,
        max_bytes: int,
    ):
        self._handler, self._args_model, self._skill = handler, args_model, skill
        self._timeout_s, self._max_bytes = timeout_s, max_bytes

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        trace_id = "-"
        try:
            raw = context.get_user_input()
            if len(raw.encode()) > self._max_bytes:
                raise AgentError("request too large")
            try:
                env = json.loads(raw)
            except ValueError:
                raise AgentError("request must be a JSON envelope") from None
            if not isinstance(env, dict) or not isinstance(env.get("args"), dict):
                raise AgentError('request must look like {"trace_id": ..., "args": {...}}')
            tid = str(env.get("trace_id") or "")
            trace_id = tid if _TRACE_OK.match(tid) else uuid.uuid4().hex
            parent = str(env.get("parent_span_id") or "")
            parent = parent if _TRACE_OK.match(parent) else None
            try:
                args = self._args_model.model_validate(env["args"])
            except ValidationError as e:
                raise AgentError(f"invalid arguments: {_short(e)}") from None
            started = time.monotonic()
            result = await asyncio.wait_for(
                asyncio.to_thread(self._traced_call, args, trace_id, parent),
                timeout=self._timeout_s,
            )
            log.info("trace=%s handled in %.2fs", trace_id, time.monotonic() - started)
            out = {"ok": True, "trace_id": trace_id, "result": result}
            payload = json.dumps(out)
        except AgentError as e:
            out = {"ok": False, "trace_id": trace_id, "error": str(e)}
            payload = json.dumps(out)
        except TimeoutError:
            log.warning("trace=%s timed out after %ss", trace_id, self._timeout_s)
            out = {
                "ok": False,
                "trace_id": trace_id,
                "error": f"timed out after {self._timeout_s:.0f}s",
            }
            payload = json.dumps(out)
        except Exception:
            log.exception("trace=%s handler failed", trace_id)  # details stay in the server log
            out = {"ok": False, "trace_id": trace_id, "error": f"internal error (trace {trace_id})"}
            payload = json.dumps(out)
        await event_queue.enqueue_event(new_text_message(payload, media_type="application/json"))

    def _traced_call(self, args: BaseModel, trace_id: str, parent: str | None) -> dict:
        # Runs in a worker thread: continue the caller's trace so one question is one tree.
        with adopt(trace_id, parent), span("a2a.handle", skill=self._skill):
            return self._handler(args, trace_id)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        return None  # single-shot skills: nothing to cancel


def make_card(
    *, name: str, description: str, skill_id: str, skill_name: str, skill_description: str,
    examples: list[str], base_url: str,
) -> AgentCard:  # fmt: skip
    return AgentCard(
        name=name,
        description=description,
        version="0.1.0",
        supported_interfaces=[
            AgentInterface(
                url=f"{base_url.rstrip('/')}{RPC_PATH}",
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=[
            AgentSkill(
                id=skill_id,
                name=skill_name,
                description=skill_description,
                tags=["finsight", "sec-filings"],
                examples=examples,
            )  # fmt: skip
        ],
    )


class _BodyLimit(BaseHTTPMiddleware):
    """Reject oversized bodies up front (the executor also re-checks the envelope size)."""

    def __init__(self, app, max_bytes: int):
        super().__init__(app)
        self._max = max_bytes

    async def dispatch(self, request: Request, call_next):
        length = request.headers.get("content-length", "0")
        if length.isdigit() and int(length) > self._max:
            return JSONResponse({"error": "request too large"}, status_code=413)
        return await call_next(request)


def build_agent_app(card: AgentCard, executor: SkillExecutor, max_bytes: int) -> Starlette:
    handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card
    )

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await handler.aclose()  # drain in-flight work on shutdown

    async def health(_request: Request) -> Response:
        return JSONResponse({"status": "ok", "agent": card.name})

    return Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(handler, RPC_PATH),
            Route("/health", health),
        ],
        middleware=[Middleware(_BodyLimit, max_bytes=max_bytes)],
        lifespan=lifespan,
    )


# ------------------------------------------------------------------ client side


class RemoteAgent:
    """Synchronous facade over the async A2A client (the graph runs in worker threads).

    One short-lived HTTP client per call: an httpx client is bound to one event loop, and each
    sync call runs its own. Connection errors are retried once; agent-reported errors never are."""

    def __init__(
        self,
        url: str,
        *,
        timeout_s: float,
        max_bytes: int,
        retries: int = 1,
        transport: httpx.AsyncBaseTransport | None = None,  # tests inject in-process ASGI
    ):
        self.url, self._timeout, self._max_bytes = url.rstrip("/"), timeout_s, max_bytes
        self._retries, self._transport = retries, transport

    def call(self, args: dict, trace_id: str) -> dict:
        """The agent's `result` dict. Raises RemoteAgentError with a safe message on any failure."""
        try:
            with adopt(trace_id, current_span_id()), span("a2a.call", agent=self.url):
                return asyncio.run(self._call(args, trace_id, current_span_id()))
        except RemoteAgentError:
            raise
        except TimeoutError:
            raise RemoteAgentError(f"agent call timed out after {self._timeout:.0f}s") from None
        except Exception:
            log.exception("trace=%s agent call failed url=%s", trace_id, self.url)
            raise RemoteAgentError(f"agent at {self.url} is unavailable") from None

    async def _call(self, args: dict, trace_id: str, parent: str | None) -> dict:
        attempts = 1 + self._retries
        for attempt in range(1, attempts + 1):
            try:
                return await asyncio.wait_for(
                    self._once(args, trace_id, parent), timeout=self._timeout
                )
            except (httpx.TransportError, A2AClientError) as e:
                if attempt == attempts:
                    log.warning(
                        "trace=%s agent %s unreachable: %s", trace_id, self.url, type(e).__name__
                    )
                    raise RemoteAgentError(f"agent at {self.url} is unavailable") from None
        raise AssertionError("unreachable")  # pragma: no cover

    async def _once(self, args: dict, trace_id: str, parent: str | None) -> dict:
        body = json.dumps({"trace_id": trace_id, "parent_span_id": parent, "args": args})
        if len(body.encode()) > self._max_bytes:
            raise RemoteAgentError("request too large to send")
        async with httpx.AsyncClient(
            transport=self._transport, timeout=self._timeout, base_url=self.url
        ) as http:
            client = await create_client(self.url, ClientConfig(streaming=False, httpx_client=http))
            try:
                req = SendMessageRequest(
                    message=new_text_message(
                        body, media_type="application/json", role=Role.ROLE_USER
                    )
                )
                async for resp in client.send_message(req):
                    if resp.HasField("message"):
                        return self._parse(get_message_text(resp.message))
                raise RemoteAgentError("agent returned no message")
            finally:
                await client.close()

    def _parse(self, text: str) -> dict:
        if len(text.encode()) > self._max_bytes:
            raise RemoteAgentError("agent response too large")
        try:
            out = json.loads(text)
        except ValueError:
            raise RemoteAgentError("agent returned invalid JSON") from None
        if not isinstance(out, dict) or not isinstance(out.get("ok"), bool):
            raise RemoteAgentError("agent returned an unexpected response shape")
        if not out["ok"]:
            raise RemoteAgentError(f"agent error: {clean_remote_text(out.get('error', ''))}")
        result = out.get("result")
        if not isinstance(result, dict):
            raise RemoteAgentError("agent returned no result")
        return result
