from __future__ import annotations

from contextlib import asynccontextmanager
import json
import time

import httpx
from fastapi import Body, FastAPI, Header, HTTPException

from app.agent.service import AgentService
from app.agent.state import StateStore
from app.clients.houses import HousesClient
from app.clients.landmarks import LandmarksClient
from app.infra.cache import CacheManager
from app.infra.logging import setup_logging
from app.schemas import CaseType, ChatRequest, ChatResponse, HealthResponse, InvokeRequest, InvokeResponse
from app.settings import AgentSettings, load_settings

LOGGER = logging.getLogger(__name__)


def _build_model_base_url(model_ip: str) -> str:
    if model_ip.startswith(("http://", "https://")):
        return model_ip
    return f"http://{model_ip}:8888"


async def _forward_chat_completion(
    http_client: httpx.AsyncClient,
    *,
    model_ip: str,
    message: str,
    session_id: str | None = None,
) -> str | None:
    headers: dict[str, str] = {}
    if session_id:
        headers["Session-ID"] = session_id

    payload = {
        "model": "",
        "messages": [{"role": "user", "content": message}],
        "stream": False,
    }
    resp = await http_client.post(
        f"{_build_model_base_url(model_ip)}/v1/chat/completions",
        json=payload,
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        return None
    msg = choices[0].get("message", {})
    return msg.get("content")


def _build_model_base_url(model_ip: str) -> str:
    if model_ip.startswith(("http://", "https://")):
        return model_ip
    return f"http://{model_ip}:8888"


async def _forward_chat_completion(
    http_client: httpx.AsyncClient,
    *,
    model_ip: str,
    message: str,
    session_id: str | None = None,
) -> str | None:
    headers: dict[str, str] = {}
    if session_id:
        headers["Session-ID"] = session_id

    payload = {
        "model": "",
        "messages": [{"role": "user", "content": message}],
        "stream": False,
    }
    resp = await http_client.post(
        f"{_build_model_base_url(model_ip)}/v1/chat/completions",
        json=payload,
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        return None
    msg = choices[0].get("message", {})
    return msg.get("content")


def create_app(settings: AgentSettings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = httpx.Timeout(
            connect=cfg.timeout.connect,
            read=cfg.timeout.read,
            write=cfg.timeout.write,
            pool=cfg.timeout.pool,
        )
        limits = httpx.Limits(
            max_connections=cfg.limits.max_connections,
            max_keepalive_connections=cfg.limits.max_keepalive_connections,
        )
        http_client = httpx.AsyncClient(timeout=timeout, limits=limits)

        cache = CacheManager(cfg)
        state_store = StateStore(cfg)
        landmarks_client = LandmarksClient(cfg.api_base_url, cfg.default_user_id, http_client)
        houses_client = HousesClient(cfg.api_base_url, cfg.default_user_id, http_client)
        service = AgentService(
            settings=cfg,
            state_store=state_store,
            landmarks_client=landmarks_client,
            houses_client=houses_client,
            cache=cache,
        )

        app.state.settings = cfg
        app.state.http_client = http_client
        app.state.agent_service = service
        app.state.session_model_ip: dict[str, str] = {}

        try:
            yield
        finally:
            await http_client.aclose()

    app = FastAPI(title="Smart Rental Agent", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def log_http_requests(request: Request, call_next):
        started = time.perf_counter()
        raw_body = await request.body()

        async def receive() -> dict[str, bytes | bool]:
            return {"type": "http.request", "body": raw_body, "more_body": False}

        request._receive = receive

        body_preview = raw_body.decode("utf-8", errors="replace") if raw_body else ""
        if len(body_preview) > 500:
            body_preview = body_preview[:500] + "...(truncated)"

        LOGGER.info(
            "incoming request method=%s path=%s query=%s body=%s",
            request.method,
            request.url.path,
            str(request.query_params),
            body_preview or "<empty>",
        )

        response = await call_next(request)
        duration_ms = int((time.perf_counter() - started) * 1000)
        LOGGER.info(
            "request completed method=%s path=%s status=%s duration_ms=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/invoke", response_model=InvokeResponse)
    async def invoke(req: InvokeRequest) -> InvokeResponse:
        service: AgentService = app.state.agent_service
        return await service.handle(req)

    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        started_ms = int(time.time() * 1000)
        service: AgentService = app.state.agent_service
        app.state.session_model_ip[req.session_id] = req.model_ip
        invoke_resp = await service.handle(
            InvokeRequest(
                session_id=req.session_id,
                case_type=CaseType.single,
                message=req.message,
            )
        )

        output = invoke_resp.text
        if invoke_resp.candidates:
            output = json.dumps(
                {
                    "message": invoke_resp.text,
                    "houses": [item.house_id for item in invoke_resp.candidates],
                },
                ensure_ascii=False,
            )
        else:
            try:
                http_client: httpx.AsyncClient = app.state.http_client
                llm_text = await _forward_chat_completion(
                    http_client,
                    model_ip=req.model_ip,
                    message=req.message,
                    session_id=req.session_id,
                )
                if llm_text:
                    output = llm_text
            except httpx.HTTPError:
                pass

        now_ms = int(time.time() * 1000)
        return ChatResponse(
            session_id=req.session_id,
            response=output,
            status="success",
            tool_results=[invoke_resp.debug] if invoke_resp.debug else [],
            timestamp=now_ms,
            duration_ms=now_ms - started_ms,
        )

    @app.post("/vi/chat/completions")
    @app.post("/v1/chat/completions")
    async def proxy_chat_completions(
        payload: dict = Body(default_factory=dict),
        model_ip: str | None = Header(default=None, alias="Model-IP"),
        session_id: str | None = Header(default=None, alias="Session-ID"),
    ) -> dict:
        resolved_model_ip = model_ip
        if not resolved_model_ip and session_id:
            resolved_model_ip = app.state.session_model_ip.get(session_id)
        if not resolved_model_ip:
            raise HTTPException(status_code=400, detail="model_ip is required (Model-IP header or known session)")

        http_client: httpx.AsyncClient = app.state.http_client
        headers: dict[str, str] = {}
        if session_id:
            headers["Session-ID"] = session_id

        response = await http_client.post(
            f"{_build_model_base_url(resolved_model_ip)}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    return app


app = create_app()
