from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
import json
import logging
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import httpx
from fastapi import Body, FastAPI, Header, HTTPException, Request

from app.agent.service import AgentService
from app.agent.state import StateStore
from app.clients.houses import HousesClient
from app.clients.landmarks import LandmarksClient
from app.infra.cache import CacheManager
from app.infra.logging import bind_log_context, log_event, preview_payload, preview_text, reset_log_context, setup_logging
from app.schemas import CaseType, ChatRequest, ChatResponse, HealthResponse, InvokeRequest, InvokeResponse
from app.settings import AgentSettings, load_settings

LOGGER = logging.getLogger(__name__)

STEP_RECV_USER_QUERY = "STEP-01-RECV-USER-QUERY"
STEP_AGENT_PIPELINE = "STEP-02-AGENT-PIPELINE"
STEP_LLM_FALLBACK = "STEP-03-LLM-FALLBACK"
STEP_LLM_NLU = "STEP-02A-LLM-NLU"
STEP_FINAL_RESPONSE = "STEP-04-FINAL-RESPONSE"
STEP_HTTP = "STEP-00-HTTP"

_TOOL_ROUTING_OPERATION_IDS = {
    "get_houses_by_platform",
    "get_houses_by_community",
    "get_houses_nearby",
    "get_house_by_id",
    "get_house_listings",
    "rent_house",
    "terminate_rental",
    "take_offline",
    "get_nearby_landmarks",
    "get_landmark_by_name",
    "search_landmarks",
}


def _build_model_base_url(model_ip: str) -> str:
    if model_ip.startswith(("http://", "https://")):
        return model_ip
    return f"http://{model_ip}:8888"


async def _forward_chat_completion(
    http_client: httpx.AsyncClient,
    *,
    model_ip: str,
    messages: list[dict[str, str]],
    session_id: str | None = None,
) -> str | None:
    headers: dict[str, str] = {}
    if session_id:
        headers["Session-ID"] = session_id

    payload = {
        "model": "",
        "messages": messages,
        "stream": False,
    }
    log_event(
        LOGGER,
        "chat.llm.forward.request",
        step=STEP_LLM_FALLBACK,
        model_ip=model_ip,
        payload_preview=preview_payload(payload),
    )

    resp = await http_client.post(
        f"{_build_model_base_url(model_ip)}/v1/chat/completions",
        json=payload,
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()
    log_event(
        LOGGER,
        "chat.llm.forward.response",
        step=STEP_LLM_FALLBACK,
        status_code=resp.status_code,
        body_preview=preview_payload(data),
    )
    choices = data.get("choices", [])
    if not choices:
        return None
    msg = choices[0].get("message", {})
    return msg.get("content")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if not cleaned:
        return None

    candidates = [cleaned]
    if "```" in cleaned:
        candidates.append(cleaned.replace("```json", "").replace("```", "").strip())

    left = cleaned.find("{")
    right = cleaned.rfind("}")
    if left != -1 and right != -1 and right > left:
        candidates.append(cleaned[left : right + 1])

    for item in candidates:
        try:
            value = json.loads(item)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


@lru_cache(maxsize=1)
def _load_tool_schema_summary() -> str:
    schema_path = Path(__file__).resolve().parents[1] / "fake_app_agent_tools.json"
    try:
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""

    paths = payload.get("paths")
    if not isinstance(paths, dict):
        return ""

    lines: list[str] = []
    for route, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, config in methods.items():
            if not isinstance(config, dict):
                continue
            operation_id = config.get("operationId")
            if operation_id not in _TOOL_ROUTING_OPERATION_IDS:
                continue
            parameters = config.get("parameters", [])
            required_names: list[str] = []
            all_names: list[str] = []
            if isinstance(parameters, list):
                for item in parameters:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    if not isinstance(name, str):
                        continue
                    all_names.append(name)
                    if item.get("required") is True:
                        required_names.append(name)
            required_text = ",".join(required_names) if required_names else "-"
            all_text = ",".join(all_names) if all_names else "-"
            lines.append(f"{operation_id}|{method.upper()} {route}|required={required_text}|params={all_text}")

    return "\n".join(lines[:30])


def _build_llm_nlu_messages(message: str, summary: str) -> list[dict[str, str]]:
    tool_summary = _load_tool_schema_summary()
    prompt = (
        "你是租房Agent意图解析器。"
        "请把输入解析为严格JSON，不要输出任何额外文字或Markdown。\n"
        "intent可选值：chat/search/compare/amenities/house_detail/listings/rent/terminate/offline。\n"
        "你必须基于可用API工具目录判断本轮最匹配的operationId，并抽取调用参数。\n"
        "只返回如下结构："
        '{"intent":"search","tool_plan":{"operationId":"get_houses_by_platform","arguments":{}},"hard":{},"soft":{},"confidence":0.0}。\n'
        "operationId必须从工具目录中选择；若是纯闲聊可填 none。\n"
        "hard可含字段：district,area,community,landmark_id,landmark_name,landmark_category,budget_min,budget_max,rent_type,"
        "layout,area_min,max_subway_dist,max_commute_min,utilities_type,move_in_date,listing_platform,house_id。\n"
        "soft可含字段：decoration,elevator,orientation,noise_preference,amenities,value_for_money,prioritize_subway_distance。\n"
        "tool_plan.arguments请使用API参数名（如 max_price/rental_type/listing_platform/house_id）。\n"
        "未提及字段不要猜测，可以不返回或设为null。\n"
        f"工具目录:\n{tool_summary}"
    )
    user_payload = {
        "history_summary": summary[:700] if summary else "",
        "message": message,
    }
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


async def _analyze_intent_with_llm(
    http_client: httpx.AsyncClient,
    *,
    model_ip: str,
    session_id: str,
    message: str,
    summary: str,
) -> dict[str, Any] | None:
    llm_text = await _forward_chat_completion(
        http_client,
        model_ip=model_ip,
        messages=_build_llm_nlu_messages(message, summary),
        session_id=session_id,
    )
    if not llm_text:
        return None
    return _extract_json_object(llm_text)


def create_app(settings: AgentSettings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    setup_logging(cfg.log_level)

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

        log_event(
            LOGGER,
            "http.request.in",
            step=STEP_HTTP,
            method=request.method,
            path=request.url.path,
            query=str(request.query_params),
            body=body_preview or "<empty>",
        )

        response = await call_next(request)
        duration_ms = int((time.perf_counter() - started) * 1000)
        log_event(
            LOGGER,
            "http.request.out",
            step=STEP_HTTP,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/invoke", response_model=InvokeResponse)
    async def invoke(req: InvokeRequest) -> InvokeResponse:
        started = time.perf_counter()
        trace_id = uuid4().hex[:12]
        user_id = req.user_id or cfg.default_user_id
        tokens = bind_log_context(
            trace_id=trace_id,
            session_id=req.session_id,
            case_type=req.case_type.value,
            user_id=user_id,
        )
        try:
            log_event(
                LOGGER,
                "invoke.received",
                step=STEP_RECV_USER_QUERY,
                message=preview_text(req.message, limit=300),
                history_len=len(req.history),
                meta_keys=sorted(req.meta.keys()),
            )
            service: AgentService = app.state.agent_service
            resp = await service.handle(req)
            log_event(
                LOGGER,
                "invoke.completed",
                step=STEP_FINAL_RESPONSE,
                duration_ms=int((time.perf_counter() - started) * 1000),
                candidate_count=len(resp.candidates),
                clarify_count=len(resp.clarify_questions),
                response=preview_text(resp.text, limit=300),
            )
            return resp
        except Exception as exc:
            log_event(
                LOGGER,
                "invoke.failed",
                step=STEP_FINAL_RESPONSE,
                duration_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )
            raise
        finally:
            reset_log_context(tokens)

    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        started_ms = int(time.time() * 1000)
        trace_id = uuid4().hex[:12]
        tokens = bind_log_context(
            trace_id=trace_id,
            session_id=req.session_id,
            case_type=CaseType.single.value,
            user_id=cfg.default_user_id,
        )
        service: AgentService = app.state.agent_service
        try:
            log_event(
                LOGGER,
                "chat.received",
                step=STEP_RECV_USER_QUERY,
                model_ip=req.model_ip,
                message=preview_text(req.message, limit=300),
            )
            app.state.session_model_ip[req.session_id] = req.model_ip

            invoke_meta: dict[str, Any] = {"model_ip": req.model_ip}
            if hasattr(service, "allow_llm_fallback") and hasattr(service, "rough_token_estimate"):
                try:
                    allow_llm_nlu_by_policy = True
                    policy_reason = "service_policy_absent"
                    if hasattr(service, "should_use_llm_nlu"):
                        allow_llm_nlu_by_policy, policy_reason = service.should_use_llm_nlu(req.session_id, req.message)

                    state_summary = ""
                    if hasattr(service, "state_store"):
                        state = service.state_store.get(req.session_id)  # type: ignore[attr-defined]
                        if state is not None and getattr(state, "conversation_summary", ""):
                            state_summary = state.conversation_summary

                    prompt_tokens = service.rough_token_estimate(req.message + state_summary)  # type: ignore[attr-defined]
                    allow_llm_nlu = service.allow_llm_fallback(req.session_id, prompt_tokens)  # type: ignore[attr-defined]
                    if allow_llm_nlu and allow_llm_nlu_by_policy:
                        http_client: httpx.AsyncClient = app.state.http_client
                        llm_parse = await _analyze_intent_with_llm(
                            http_client,
                            model_ip=req.model_ip,
                            session_id=req.session_id,
                            message=req.message,
                            summary=state_summary,
                        )
                        if llm_parse:
                            invoke_meta["llm_parse"] = llm_parse
                            log_event(
                                LOGGER,
                                "chat.llm_nlu.applied",
                                step=STEP_LLM_NLU,
                                llm_parse=llm_parse,
                            )
                            if hasattr(service, "record_llm_fallback_usage"):
                                completion_tokens = service.rough_token_estimate(json.dumps(llm_parse, ensure_ascii=False))  # type: ignore[attr-defined]
                                service.record_llm_fallback_usage(req.session_id, prompt_tokens + completion_tokens)  # type: ignore[attr-defined]
                    else:
                        log_event(
                            LOGGER,
                            "chat.llm_nlu.skipped",
                            step=STEP_LLM_NLU,
                            reason="budget_exceeded" if not allow_llm_nlu else policy_reason,
                            estimated_prompt_tokens=prompt_tokens,
                        )
                except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                    log_event(
                        LOGGER,
                        "chat.llm_nlu.failed",
                        step=STEP_LLM_NLU,
                        error=str(exc),
                    )

            log_event(
                LOGGER,
                "chat.agent.invoke.start",
                step=STEP_AGENT_PIPELINE,
                input_message=preview_text(req.message, limit=300),
            )
            invoke_resp = await service.handle(
                InvokeRequest(
                    session_id=req.session_id,
                    case_type=CaseType.single,
                    message=req.message,
                    meta=invoke_meta,
                )
            )
            log_event(
                LOGGER,
                "chat.agent.invoke.done",
                step=STEP_AGENT_PIPELINE,
                invoke_text=preview_text(invoke_resp.text, limit=300),
                candidate_count=len(invoke_resp.candidates),
                clarify_count=len(invoke_resp.clarify_questions),
                debug=invoke_resp.debug,
            )

            output = invoke_resp.text
            response_kind = str(invoke_resp.debug.get("response_kind", "chat"))
            if response_kind == "search" or bool(invoke_resp.candidates):
                output = json.dumps(
                    {
                        "message": invoke_resp.text,
                        "houses": [item.house_id for item in invoke_resp.candidates],
                    },
                    ensure_ascii=False,
                )
            elif response_kind == "chat":
                try:
                    prompt_tokens = (
                        service.rough_token_estimate(req.message)
                        if hasattr(service, "rough_token_estimate")
                        else max(1, int(len(req.message) / 2))
                    )
                    allow_llm = (
                        service.allow_llm_fallback(req.session_id, prompt_tokens)
                        if hasattr(service, "allow_llm_fallback")
                        else True
                    )
                    if allow_llm:
                        http_client: httpx.AsyncClient = app.state.http_client
                        llm_messages = (
                            service.build_llm_fallback_messages(req.session_id, req.message)
                            if hasattr(service, "build_llm_fallback_messages")
                            else [{"role": "user", "content": req.message}]
                        )
                        llm_text = await _forward_chat_completion(
                            http_client,
                            model_ip=req.model_ip,
                            messages=llm_messages,
                            session_id=req.session_id,
                        )
                        if llm_text:
                            output = llm_text
                            completion_tokens = (
                                service.rough_token_estimate(llm_text)
                                if hasattr(service, "rough_token_estimate")
                                else max(1, int(len(llm_text) / 2))
                            )
                            if hasattr(service, "record_llm_fallback_usage"):
                                service.record_llm_fallback_usage(req.session_id, prompt_tokens + completion_tokens)
                            log_event(
                                LOGGER,
                                "chat.llm_fallback.applied",
                                step=STEP_LLM_FALLBACK,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                llm_output=preview_text(llm_text, limit=300),
                            )
                    else:
                        log_event(
                            LOGGER,
                            "chat.llm_fallback_skipped",
                            step=STEP_LLM_FALLBACK,
                            reason="budget_exceeded",
                            estimated_prompt_tokens=prompt_tokens,
                        )
                except httpx.HTTPError as exc:
                    log_event(
                        LOGGER,
                        "chat.llm_fallback_failed",
                        step=STEP_LLM_FALLBACK,
                        error=str(exc),
                    )

            now_ms = int(time.time() * 1000)
            log_event(
                LOGGER,
                "chat.completed",
                step=STEP_FINAL_RESPONSE,
                duration_ms=now_ms - started_ms,
                invoke_output=preview_text(invoke_resp.text, limit=300),
                response=preview_text(output, limit=300),
            )
            return ChatResponse(
                session_id=req.session_id,
                response=output,
                status="success",
                tool_results=[invoke_resp.debug] if invoke_resp.debug else [],
                timestamp=now_ms,
                duration_ms=now_ms - started_ms,
            )
        finally:
            reset_log_context(tokens)

    @app.post("/vi/chat/completions")
    @app.post("/v1/chat/completions")
    async def proxy_chat_completions(
        payload: dict = Body(default_factory=dict),
        model_ip: str | None = Header(default=None, alias="Model-IP"),
        session_id: str | None = Header(default=None, alias="Session-ID"),
    ) -> dict:
        if not session_id:
            raise HTTPException(status_code=400, detail="Session-ID header is required for /v1/chat/completions")

        resolved_model_ip = model_ip
        if not resolved_model_ip:
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

    @app.post("/v2/chat/completions")
    async def proxy_chat_completions_v2(
        payload: dict = Body(default_factory=dict),
        model_ip: str | None = Header(default=None, alias="Model-IP"),
    ) -> dict:
        if not model_ip:
            raise HTTPException(status_code=400, detail="Model-IP header is required for /v2/chat/completions")

        http_client: httpx.AsyncClient = app.state.http_client
        response = await http_client.post(
            f"{_build_model_base_url(model_ip)}/v1/chat/completions",
            json=payload,
            headers={},
        )
        response.raise_for_status()
        return response.json()

    return app


app = create_app()
