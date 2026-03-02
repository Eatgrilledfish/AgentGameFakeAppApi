from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import time
from uuid import uuid4

import httpx
from fastapi import Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from app.agent.service import AgentService
from app.agent.state import StateStore
from app.clients.houses import HousesClient
from app.clients.landmarks import LandmarksClient
from app.infra.cache import CacheManager
from app.infra.logging import (
    bind_log_context,
    clear_trace_events,
    get_trace_events,
    log_event,
    preview_payload,
    preview_text,
    reset_log_context,
    setup_logging,
)
from app.schemas import CaseType, ChatRequest, ChatResponse, HealthResponse, InvokeRequest, InvokeResponse
from app.settings import AgentSettings, load_settings

LOGGER = logging.getLogger(__name__)

STEP_RECV_USER_QUERY = "STEP-01-RECV-USER-QUERY"
STEP_AGENT_PIPELINE = "STEP-02-AGENT-PIPELINE"
STEP_LLM_FALLBACK = "STEP-03-LLM-FALLBACK"
STEP_FINAL_RESPONSE = "STEP-04-FINAL-RESPONSE"
STEP_HTTP = "STEP-00-HTTP"



DEBUG_CHAT_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>实时链路监控面板</title>
  <style>
    body{font-family:Arial,sans-serif;margin:0;background:#f5f7fb;color:#223;}
    .wrap{max-width:1200px;margin:24px auto;padding:0 16px;}
    .card{background:#fff;border-radius:10px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:16px;}
    .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;}
    input,textarea,button{padding:10px;border:1px solid #cfd7e6;border-radius:8px;font-size:14px;}
    input,textarea{flex:1;min-width:240px;}
    button{background:#2f6feb;color:#fff;border:none;cursor:pointer;}
    button.secondary{background:#54627a;}
    .timeline{display:grid;grid-template-columns:1fr;gap:10px;}
    .event{border-left:4px solid #2f6feb;background:#f8faff;padding:10px 12px;border-radius:6px;}
    .event.upstream{border-left-color:#9c27b0;background:#fbf6ff;}
    .event.llm{border-left-color:#ff9800;background:#fffaf1;}
    .event.final{border-left-color:#2e7d32;background:#f3fff3;}
    .meta{font-size:12px;color:#556;}
    .cols{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
    @media (max-width:900px){.cols{grid-template-columns:1fr;}}
    pre{white-space:pre-wrap;word-break:break-word;margin:6px 0 0;font-size:12px;}
    .tip{font-size:13px;color:#445;}
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h2>8080 实时交互可视化面板</h2>
    <p class="tip">用途：监听 session 的真实处理过程。建议把本服务运行在 <b>8080</b>，然后打开 <code>http://127.0.0.1:8080/debug/chat-view</code>。</p>
    <div class="row">
      <input id="apiBase" placeholder="监听地址，例如 http://127.0.0.1:8080" value=""/>
      <input id="modelIp" placeholder="model_ip（发送消息时需要）"/>
      <input id="sessionId" placeholder="session_id（监听必填）"/>
    </div>
    <div class="row" style="margin-top:8px;">
      <textarea id="message" rows="3" placeholder="可选：在此发送消息并观察全链路"></textarea>
    </div>
    <div class="row" style="margin-top:8px;">
      <button onclick="send()">发送测试消息</button>
      <button class="secondary" onclick="startWatch()">开始监听</button>
      <button class="secondary" onclick="stopWatch()">停止监听</button>
      <button class="secondary" onclick="clearTrace()">清空当前会话轨迹</button>
    </div>
    <p id="resp"></p>
  </div>

  <div class="cols">
    <div class="card">
      <h3>关键输入输出</h3>
      <div id="summary"></div>
    </div>
    <div class="card">
      <h3>完整事件时间线</h3>
      <div id="events" class="timeline"></div>
    </div>
  </div>
</div>
<script>
let timer = null;
let rendered = new Set();
let lastSeq = 0;

function baseUrl(){
  const val = document.getElementById('apiBase').value.trim();
  return val || window.location.origin;
}
function sessionId(){
  return document.getElementById('sessionId').value.trim();
}
function cls(ev){
  const name = String(ev.event||'');
  if(name.startsWith('upstream.')) return 'event upstream';
  if(name.includes('llm')) return 'event llm';
  if(name.includes('completed') || name.includes('invoke.completed')) return 'event final';
  return 'event';
}
function esc(s){ return String(s ?? '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function renderSummary(events){
  const pick = (name) => events.filter(e => e.event === name).at(-1);
  const recv = pick('chat.received');
  const invokeDone = pick('chat.agent.invoke.done');
  const llm = pick('chat.llm_fallback.applied') || pick('chat.llm.forward.response');
  const final = pick('chat.completed') || pick('invoke.completed');
  const rows = [
    ['用户输入', recv?.message || '-'],
    ['Agent输出', invokeDone?.invoke_text || final?.invoke_output || '-'],
    ['大模型输出', llm?.llm_output || llm?.body_preview || '-'],
    ['最终返回用户', final?.response || '-'],
  ];
  document.getElementById('summary').innerHTML = rows.map(r => `<div class="event"><b>${esc(r[0])}</b><pre>${esc(r[1])}</pre></div>`).join('');
}

function appendEvents(events){
  const el = document.getElementById('events');
  for(const ev of events){
    const key = `${ev.trace_id}-${ev.seq}-${ev.event}`;
    if(rendered.has(key)) continue;
    rendered.add(key);
    lastSeq = Math.max(lastSeq, Number(ev.seq||0));
    const item = document.createElement('div');
    item.className = cls(ev);
    item.innerHTML = `<div><b>${esc(ev.step||'-')}</b> · <span>${esc(ev.event||'-')}</span></div><div class="meta">seq=${esc(ev.seq)} trace_id=${esc(ev.trace_id)}</div><pre>${esc(JSON.stringify(ev,null,2))}</pre>`;
    el.appendChild(item);
  }
}

async function refreshTrace(){
  const sid = sessionId();
  if(!sid){ return; }
  const url = `${baseUrl()}/debug/traces/${encodeURIComponent(sid)}?since_seq=${lastSeq}`;
  const res = await fetch(url);
  const data = await res.json();
  const events = data.events || [];
  appendEvents(events);

  const full = await fetch(`${baseUrl()}/debug/traces/${encodeURIComponent(sid)}`);
  const fullData = await full.json();
  renderSummary(fullData.events || []);
}

async function send(){
  const model_ip=document.getElementById('modelIp').value.trim();
  const sid=sessionId();
  const message=document.getElementById('message').value.trim();
  if(!model_ip||!sid||!message){alert('请填写 model_ip/session_id/message');return;}
  const res=await fetch(`${baseUrl()}/api/v1/chat`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_ip,session_id:sid,message})});
  const data=await res.json();
  document.getElementById('resp').innerText='最终返回：'+(data.response||JSON.stringify(data));
  await refreshTrace();
}

function startWatch(){
  if(timer) clearInterval(timer);
  timer = setInterval(refreshTrace, 1200);
  refreshTrace();
}
function stopWatch(){
  if(timer) clearInterval(timer);
  timer = null;
}

async function clearTrace(){
  const sid=sessionId();
  if(!sid){alert('先输入 session_id');return;}
  await fetch(`${baseUrl()}/debug/traces/${encodeURIComponent(sid)}`,{method:'DELETE'});
  rendered = new Set();
  lastSeq = 0;
  document.getElementById('events').innerHTML='';
  document.getElementById('summary').innerHTML='';
}

document.getElementById('apiBase').value = window.location.origin;
</script>
</body>
</html>
"""


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


    @app.get("/debug/chat-view", response_class=HTMLResponse)
    async def debug_chat_view() -> HTMLResponse:
        return HTMLResponse(content=DEBUG_CHAT_PAGE)

    @app.get("/debug/traces/{session_id}")
    async def debug_get_trace(session_id: str, since_seq: int = Query(default=0, ge=0)) -> dict:
        events = get_trace_events(session_id)
        if since_seq > 0:
            events = [event for event in events if int(event.get("seq", 0)) > since_seq]
        return {"session_id": session_id, "events": events}

    @app.delete("/debug/traces/{session_id}")
    async def debug_clear_trace(session_id: str) -> dict:
        removed = clear_trace_events(session_id)
        return {"session_id": session_id, "cleared": removed}

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
                        llm_text = await _forward_chat_completion(
                            http_client,
                            model_ip=req.model_ip,
                            message=req.message,
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
