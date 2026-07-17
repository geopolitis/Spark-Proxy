import json
import asyncio
import time
import uuid
from collections import deque
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from openai import AsyncOpenAI
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from bs4 import BeautifulSoup
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

import os
import httpx
# --- Configuration ---
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8080/v1")
MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen3.6-35B-A3B-FP8")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8081"))
CONFIG_PATH = os.environ.get("PROXY_CONFIG_PATH", "/home/geo/Gemini/models.yaml")
REQUEST_LOG_PATH = os.environ.get("PROXY_REQUEST_LOG_PATH", "/home/geo/Gemini/proxy_requests.jsonl")
METRICS_PATH = os.environ.get("PROXY_METRICS_PATH", "/home/geo/Gemini/proxy_metrics.json")
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", str(24 * 1024 * 1024)))
STREAM_INACTIVITY_TIMEOUT_SECONDS = float(os.environ.get("STREAM_INACTIVITY_TIMEOUT_SECONDS", "120"))
FORCE_STREAM_USAGE = os.environ.get("FORCE_STREAM_USAGE", "1").lower() not in {"0", "false", "no"}
DEFAULT_CLIENT_PROFILE = os.environ.get("DEFAULT_CLIENT_PROFILE", "generic").strip().lower() or "generic"
COPILOT_MAX_OUTPUT_TOKENS = int(os.environ.get("COPILOT_MAX_OUTPUT_TOKENS", "4096"))
KILO_MAX_OUTPUT_TOKENS = int(os.environ.get("KILO_MAX_OUTPUT_TOKENS", "8192"))
CIRCUIT_BREAKER_FAILURES = int(os.environ.get("CIRCUIT_BREAKER_FAILURES", "5"))
CIRCUIT_BREAKER_WINDOW_SECONDS = float(os.environ.get("CIRCUIT_BREAKER_WINDOW_SECONDS", "60"))
CIRCUIT_BREAKER_COOLDOWN_SECONDS = float(os.environ.get("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "30"))
BACKEND_CONNECT_TIMEOUT_SECONDS = float(os.environ.get("BACKEND_CONNECT_TIMEOUT_SECONDS", "10"))
BACKEND_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("BACKEND_REQUEST_TIMEOUT_SECONDS", "180"))
STRICT_UNSUPPORTED_PARAMS = os.environ.get("STRICT_UNSUPPORTED_PARAMS", "1").lower() not in {"0", "false", "no"}

DEFAULT_CONFIG = {
    "default_model": MODEL_NAME,
    "models": {
        "Qwen/Qwen3-Coder-Next": {
            "backend_model": "Qwen/Qwen3-Coder-Next",
            "max_context_tokens": 262144,
            "default_max_tokens": 4096,
            "hard_max_tokens": 16384,
            "tool_parser": "qwen3_coder",
            "supports_tools": True,
            "supports_stream_usage": True,
            "supports_parallel_tools": False,
            "recommended_temperature": 0,
            "client_overrides": {
                "kilo": {"max_tokens": 8192, "stream": True},
                "copilot": {"max_tokens": 4096, "parallel_tool_calls": False},
            },
        },
        "qwen36-nvfp4-plain-b8192": {
            "backend_model": "qwen36-nvfp4-plain-b8192",
            "max_context_tokens": 262144,
            "default_max_tokens": 4096,
            "hard_max_tokens": 8192,
            "supports_tools": True,
            "tool_parser": "qwen3_xml",
            "supports_stream_usage": True,
            "supports_parallel_tools": False,
            "recommended_temperature": 0,
        },
    },
    "client_profiles": {
        "generic": {"rewrite": "minimal"},
        "kilo": {"force_stream_usage": True},
        "copilot": {"force_stream_usage": True, "disable_parallel_tools": True},
        "openwebui": {"rewrite": "minimal"},
        "vscode": {"rewrite": "minimal"},
        "benchmark": {"rewrite": "none", "allow_large_outputs": True},
    },
}

app = FastAPI(title="Spark Search Proxy")

# Add CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI(base_url=VLLM_URL, api_key="not-needed")

STATS_LOCK = asyncio.Lock()
LOG_LOCK = asyncio.Lock()
CONFIG = dict(DEFAULT_CONFIG)
CIRCUIT_LOCK = asyncio.Lock()
CIRCUIT = {
    "state": "closed",
    "opened_at": 0.0,
    "failures": [],
    "last_error": None,
}
STATS = {
    "started_at": time.time(),
    "chat_requests": 0,
    "stream_requests": 0,
    "stream_done_events": 0,
    "stream_finish_chunks": 0,
    "stream_missing_done": 0,
    "stream_timeouts": 0,
    "stream_bytes": 0,
    "client_profile_generic": 0,
    "client_profile_kilo": 0,
    "client_profile_copilot": 0,
    "client_profile_openwebui": 0,
    "client_profile_vscode": 0,
    "client_profile_unknown": 0,
    "backend_requests": 0,
    "backend_errors": 0,
    "backend_retries": 0,
    "backend_unavailable": 0,
    "validation_errors": 0,
    "context_overflows": 0,
    "request_size_rejections": 0,
    "circuit_open_rejections": 0,
    "tool_rounds": 0,
    "web_search_calls": 0,
    "web_search_success": 0,
    "web_search_empty": 0,
    "web_search_errors": 0,
    "total_latency_ms": 0.0,
    "total_search_latency_ms": 0.0,
    "http_requests": 0,
    "http_2xx": 0,
    "http_3xx": 0,
    "http_4xx": 0,
    "http_5xx": 0,
    "http_exceptions": 0,
    "total_http_latency_ms": 0.0,
    "last_error": None,
}
RECENT_SEARCHES = deque(maxlen=25)
RECENT_REQUESTS = deque(maxlen=50)
RECENT_FAILURES = deque(maxlen=50)


def load_proxy_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        return dict(DEFAULT_CONFIG)
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(text) or {}
    except Exception:
        loaded = json.loads(text)
    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    config["models"] = {**DEFAULT_CONFIG.get("models", {}), **(loaded.get("models") or {})}
    config["client_profiles"] = {**DEFAULT_CONFIG.get("client_profiles", {}), **(loaded.get("client_profiles") or {})}
    return config


def load_persisted_metrics():
    if not os.path.exists(METRICS_PATH):
        return
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as fh:
            persisted = json.load(fh)
        counters = persisted.get("counters", {})
        for key, value in counters.items():
            if key in STATS and isinstance(value, (int, float)):
                STATS[key] = value
        STATS["started_at"] = time.time()
    except Exception as exc:
        STATS["last_error"] = f"failed to load metrics: {exc}"


def persist_metrics_snapshot(stats_copy: Dict[str, Any] | None = None):
    try:
        snapshot = {"ts": time.time(), "counters": stats_copy or STATS}
        tmp_path = f"{METRICS_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, sort_keys=True)
        os.replace(tmp_path, METRICS_PATH)
    except Exception:
        pass


async def append_request_log(row: Dict[str, Any]):
    try:
        async with LOG_LOCK:
            with open(REQUEST_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    except Exception:
        pass


def openai_error(message: str, status_code: int, error_type: str = "BadRequestError", param: str | None = None, code: str | int | None = None):
    payload = {"error": {"message": message, "type": error_type, "param": param, "code": code or status_code}}
    return JSONResponse(payload, status_code=status_code)


def exception_to_openai_error(exc: Exception):
    text = str(exc)
    lower = text.lower()
    status = 500
    kind = "ProxyError"
    code = "proxy_error"
    if isinstance(exc, HTTPException):
        status = int(exc.status_code)
        text = str(exc.detail)
        lower = text.lower()
    if "maximum context length" in lower or "context" in lower and "tokens" in lower:
        status, kind, code = 400, "ContextOverflowError", "context_overflow"
    elif "timed out" in lower or "timeout" in lower:
        status, kind, code = 504, "BackendTimeoutError", "backend_timeout"
    elif "connection" in lower or "connect" in lower or "unavailable" in lower:
        status, kind, code = 503, "BackendUnavailableError", "backend_unavailable"
    elif "tool" in lower and ("malformed" in lower or "json" in lower):
        status, kind, code = 400, "MalformedToolCallError", "malformed_tool_call"
    elif status == 400:
        kind, code = "BadRequestError", "bad_request"
    return openai_error(text, status, kind, code=code)


try:
    CONFIG = load_proxy_config()
except Exception as exc:
    STATS["last_error"] = f"failed to load config: {exc}"
load_persisted_metrics()


async def update_stats(**values):
    async with STATS_LOCK:
        for key, value in values.items():
            if key.endswith("_inc"):
                STATS[key[:-4]] = STATS.get(key[:-4], 0) + value
            else:
                STATS[key] = value
        snapshot = dict(STATS)
    persist_metrics_snapshot(snapshot)


def model_profiles() -> Dict[str, Any]:
    return CONFIG.get("models") or {}


def default_model_name() -> str:
    return CONFIG.get("default_model") or MODEL_NAME


def resolve_model_profile(model_name: str | None) -> tuple[str, Dict[str, Any]]:
    requested = model_name or default_model_name()
    profiles = model_profiles()
    if requested in profiles:
        profile = dict(profiles[requested])
        profile.setdefault("backend_model", requested)
        return requested, profile
    for name, profile in profiles.items():
        if requested == profile.get("backend_model"):
            merged = dict(profile)
            merged.setdefault("backend_model", requested)
            return name, merged
    raise HTTPException(
        status_code=400,
        detail=f"Unknown model '{requested}'. Available models: {', '.join(sorted(profiles))}",
    )


def output_token_value(body: Dict[str, Any]) -> int | None:
    value = body.get("max_tokens", body.get("max_completion_tokens"))
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def set_output_token_value(body: Dict[str, Any], value: int):
    if "max_completion_tokens" in body and "max_tokens" not in body:
        body["max_completion_tokens"] = value
    else:
        body["max_tokens"] = value


def estimate_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return max(1, (len(value) + 3) // 4)
    if isinstance(value, list):
        return sum(estimate_tokens(item) for item in value)
    if isinstance(value, dict):
        return sum(estimate_tokens(v) for v in value.values()) + len(value)
    return max(1, (len(str(value)) + 3) // 4)


def estimate_prompt_tokens(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    return sum(estimate_tokens(message) + 4 for message in messages)


async def circuit_allows_request() -> bool:
    async with CIRCUIT_LOCK:
        if CIRCUIT["state"] != "open":
            return True
        if time.time() - float(CIRCUIT["opened_at"]) >= CIRCUIT_BREAKER_COOLDOWN_SECONDS:
            CIRCUIT["state"] = "half_open"
            return True
        return False


async def record_backend_success():
    async with CIRCUIT_LOCK:
        CIRCUIT["state"] = "closed"
        CIRCUIT["failures"] = []
        CIRCUIT["last_error"] = None


async def record_backend_failure(error: str):
    now = time.time()
    async with CIRCUIT_LOCK:
        failures = [ts for ts in CIRCUIT["failures"] if now - ts <= CIRCUIT_BREAKER_WINDOW_SECONDS]
        failures.append(now)
        CIRCUIT["failures"] = failures
        CIRCUIT["last_error"] = error
        if len(failures) >= CIRCUIT_BREAKER_FAILURES:
            CIRCUIT["state"] = "open"
            CIRCUIT["opened_at"] = now
        RECENT_FAILURES.appendleft({"ts": now, "error": error, "circuit_state": CIRCUIT["state"]})


async def record_search(query: str, ok: bool, latency_ms: float, result_count: int, error: str | None = None):
    async with STATS_LOCK:
        RECENT_SEARCHES.appendleft({
            "ts": time.time(),
            "query": query,
            "ok": ok,
            "latency_ms": latency_ms,
            "result_count": result_count,
            "error": error,
        })


async def record_request(request: Request, status_code: int, latency_ms: float, error: str | None = None):
    client = request.client
    profile = getattr(request.state, "client_profile", None)
    request_id = getattr(request.state, "request_id", None)
    status_family = f"http_{int(status_code / 100)}xx"
    async with STATS_LOCK:
        STATS["http_requests"] = STATS.get("http_requests", 0) + 1
        STATS[status_family] = STATS.get(status_family, 0) + 1
        STATS["total_http_latency_ms"] = STATS.get("total_http_latency_ms", 0.0) + latency_ms
        if error:
            STATS["http_exceptions"] = STATS.get("http_exceptions", 0) + 1
            STATS["last_error"] = error
        RECENT_REQUESTS.appendleft({
            "ts": time.time(),
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "client_host": client.host if client else None,
            "client_port": client.port if client else None,
            "client_profile": profile,
            "request_id": request_id,
            "error": error,
        })


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    started = time.time()
    status_code = 500
    error = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        await record_request(request, status_code, (time.time() - started) * 1000.0, error)

# --- Search Logic ---
def clean_text(value):
    return " ".join(value.get_text(" ", strip=True).split()) if value else ""


def parse_bing_results(html: str):
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for item in soup.select("li.b_algo")[:6]:
        link = item.select_one("h2 a")
        if not link:
            continue
        snippet = item.select_one(".b_caption p") or item.select_one("p")
        title = clean_text(link)
        text = clean_text(snippet)
        href = link.get("href", "")
        if title and text:
            results.append({"title": title, "url": href, "snippet": text})
    return results[:4]


def parse_duckduckgo_results(html: str):
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for item in soup.select(".result, .web-result")[:6]:
        link = item.select_one(".result__title a, a.result__a")
        snippet = item.select_one(".result__snippet")
        title = clean_text(link)
        text = clean_text(snippet)
        href = link.get("href", "") if link else ""
        if title and text:
            results.append({"title": title, "url": href, "snippet": text})
    return results[:4]


async def perform_web_search(query: str):
    print(f"\n[Proxy] Searching for: {query}")
    started = time.time()
    await update_stats(web_search_calls_inc=1)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        try:
            encoded = query.replace(" ", "+")
            search_targets = [
                (f"https://www.bing.com/search?q={encoded}", parse_bing_results),
                (f"https://duckduckgo.com/html/?q={encoded}", parse_duckduckgo_results),
            ]
            last_error = None
            for url, parser in search_targets:
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                    results = parser(await page.content())
                    if results:
                        await browser.close()
                        latency_ms = (time.time() - started) * 1000.0
                        await update_stats(web_search_success_inc=1, total_search_latency_ms_inc=latency_ms)
                        await record_search(query, True, latency_ms, len(results))
                        return json.dumps(results)
                except Exception as e:
                    last_error = e

            await browser.close()
            if last_error:
                latency_ms = (time.time() - started) * 1000.0
                error = str(last_error)
                await update_stats(web_search_errors_inc=1, total_search_latency_ms_inc=latency_ms, last_error=error)
                await record_search(query, False, latency_ms, 0, error)
                return f"Search failed: {error}"
            latency_ms = (time.time() - started) * 1000.0
            await update_stats(web_search_empty_inc=1, total_search_latency_ms_inc=latency_ms)
            await record_search(query, True, latency_ms, 0)
            return "No results found."
        except Exception as e:
            try: await browser.close()
            except: pass
            latency_ms = (time.time() - started) * 1000.0
            error = str(e)
            await update_stats(web_search_errors_inc=1, total_search_latency_ms_inc=latency_ms, last_error=error)
            await record_search(query, False, latency_ms, 0, error)
            return f"Search failed: {error}"

# --- Tool Definition ---
TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Access real-time internet information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
}]

# --- Proxy Endpoints ---
CHAT_COMPLETION_KEYS = {
    "model",
    "messages",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_p",
    "user",
    "extra_body",
    "parallel_tool_calls",
    "client_profile",
    "client",
    "profile",
}
FORWARDED_CHAT_KEYS = CHAT_COMPLETION_KEYS - {"client_profile", "client", "profile", "parallel_tool_calls"}


def sanitize_chat_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only fields accepted by the OpenAI chat-completions client."""
    return {key: value for key, value in body.items() if key in FORWARDED_CHAT_KEYS}


def unsupported_chat_keys(body: Dict[str, Any]) -> list[str]:
    return sorted(key for key in body if key not in CHAT_COMPLETION_KEYS)


def header_text(request: Request) -> str:
    interesting = [
        "user-agent",
        "x-client-name",
        "x-client-version",
        "x-stainless-package-version",
        "x-title",
        "x-requested-with",
        "referer",
        "origin",
    ]
    return " ".join(str(request.headers.get(name, "")) for name in interesting).lower()


def body_profile_hint(body: Dict[str, Any]) -> str | None:
    extra = body.get("extra_body") if isinstance(body.get("extra_body"), dict) else {}
    for key in ("client_profile", "client", "profile"):
        value = body.get(key) or extra.get(key)
        if value:
            return str(value).strip().lower()
    return None


def detect_client_profile(request: Request, raw_body: Dict[str, Any]) -> Dict[str, str]:
    explicit = (
        request.headers.get("x-client-profile")
        or request.headers.get("x-llm-client-profile")
        or body_profile_hint(raw_body)
    )
    if explicit:
        return {"name": explicit.strip().lower(), "source": "explicit"}

    text = header_text(request)
    if "kilo" in text or "kilocode" in text or "kilo-code" in text:
        return {"name": "kilo", "source": "headers"}
    if "copilot" in text or "llm-gateway" in text or "github.copilot" in text:
        return {"name": "copilot", "source": "headers"}
    if "open-webui" in text or "openwebui" in text:
        return {"name": "openwebui", "source": "headers"}
    if "vscode" in text or "visual studio code" in text or "electron" in text:
        return {"name": "vscode", "source": "headers"}

    has_tools = bool(raw_body.get("tools"))
    has_parallel_tools = "parallel_tool_calls" in raw_body
    max_tokens = raw_body.get("max_tokens") or raw_body.get("max_completion_tokens") or 0
    try:
        max_tokens = int(max_tokens)
    except Exception:
        max_tokens = 0
    if raw_body.get("stream") and has_tools and max_tokens >= 16000:
        return {"name": "copilot", "source": "request_shape"}
    if raw_body.get("stream") and (has_tools or has_parallel_tools):
        return {"name": "kilo", "source": "request_shape"}
    return {"name": DEFAULT_CLIENT_PROFILE, "source": "default"}


async def record_client_profile(profile: str):
    normalized = profile if profile in {"generic", "kilo", "copilot", "openwebui", "vscode"} else "unknown"
    await update_stats(**{f"client_profile_{normalized}_inc": 1})


def profile_max_output(profile: str, model_profile: Dict[str, Any] | None = None) -> int | None:
    model_profile = model_profile or {}
    overrides = (model_profile.get("client_overrides") or {}).get(profile) or {}
    if overrides.get("max_tokens") is not None:
        try:
            return int(overrides["max_tokens"])
        except Exception:
            pass
    if profile == "copilot":
        return COPILOT_MAX_OUTPUT_TOKENS
    if profile == "kilo":
        return KILO_MAX_OUTPUT_TOKENS
    return None


def normalize_body_for_profile(body: Dict[str, Any], profile: str, model_profile: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(body)
    if normalized.get("tools") == []:
        normalized.pop("tools", None)
        if normalized.get("tool_choice") in (None, "auto", "required"):
            normalized.pop("tool_choice", None)
    if isinstance(normalized.get("extra_body"), dict):
        extra_body = {
            key: value for key, value in normalized["extra_body"].items()
            if key not in {"client_profile", "client", "profile"}
        }
        if extra_body:
            normalized["extra_body"] = extra_body
        else:
            normalized.pop("extra_body", None)
    default_max = int(model_profile.get("default_max_tokens") or 4096)
    hard_max = int(model_profile.get("hard_max_tokens") or default_max)
    output_cap = min(hard_max, profile_max_output(profile, model_profile) or hard_max)
    requested_output = output_token_value(normalized)
    if requested_output is None:
        set_output_token_value(normalized, min(default_max, output_cap))
    else:
        set_output_token_value(normalized, min(requested_output, output_cap))

    if profile == "copilot":
        normalized.setdefault("temperature", model_profile.get("recommended_temperature", 0))
    elif profile == "kilo":
        normalized.setdefault("temperature", model_profile.get("recommended_temperature", 0))
        if normalized.get("stream"):
            stream_options = dict(normalized.get("stream_options") or {})
            stream_options.setdefault("include_usage", True)
            normalized["stream_options"] = stream_options

    if not model_profile.get("supports_parallel_tools", False):
        normalized.pop("parallel_tool_calls", None)
    if not model_profile.get("supports_tools", True):
        normalized.pop("tools", None)
        normalized.pop("tool_choice", None)
    normalized["model"] = model_profile.get("backend_model") or normalized.get("model")
    return normalized


def validate_chat_request(raw_body: Dict[str, Any], body: Dict[str, Any], model_name: str, model_profile: Dict[str, Any]):
    if STRICT_UNSUPPORTED_PARAMS:
        unsupported = unsupported_chat_keys(raw_body)
        if unsupported:
            raise HTTPException(status_code=400, detail=f"Unsupported chat completion parameters: {', '.join(unsupported)}")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")
    max_context = int(model_profile.get("max_context_tokens") or 0)
    prompt_tokens = estimate_prompt_tokens(messages)
    output_tokens = output_token_value(body) or int(model_profile.get("default_max_tokens") or 4096)
    if max_context and prompt_tokens + output_tokens > max_context:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This model's maximum context length is {max_context} tokens. "
                f"Estimated prompt tokens are {prompt_tokens} and requested output tokens are {output_tokens}, "
                f"for an estimated total of {prompt_tokens + output_tokens}. "
                "Reduce prompt size or max output tokens."
            ),
        )
    return {"model": model_name, "prompt_tokens_estimated": prompt_tokens, "output_tokens": output_tokens, "max_context_tokens": max_context}


def sse_event(data: str) -> str:
    return f"data: {data}\n\n"


def prepare_stream_body(body: Dict[str, Any]) -> Dict[str, Any]:
    stream_body = dict(body)
    extra_body = stream_body.pop("extra_body", None)
    if isinstance(extra_body, dict):
        for key, value in extra_body.items():
            stream_body.setdefault(key, value)
    stream_body["stream"] = True
    if FORCE_STREAM_USAGE:
        stream_options = dict(stream_body.get("stream_options") or {})
        stream_options.setdefault("include_usage", True)
        stream_body["stream_options"] = stream_options
    return stream_body


async def open_backend_stream(body: Dict[str, Any]):
    if not await circuit_allows_request():
        await update_stats(circuit_open_rejections_inc=1)
        raise HTTPException(status_code=503, detail=f"Backend circuit breaker is open: {CIRCUIT.get('last_error')}")
    timeout = httpx.Timeout(
        connect=BACKEND_CONNECT_TIMEOUT_SECONDS,
        read=STREAM_INACTIVITY_TIMEOUT_SECONDS,
        write=30.0,
        pool=10.0,
    )
    http_client = httpx.AsyncClient(timeout=timeout)
    try:
        response = None
        for attempt in range(2):
            try:
                await update_stats(backend_requests_inc=1)
                response = await http_client.send(
                    http_client.build_request(
                        "POST",
                        f"{VLLM_URL.rstrip('/')}/chat/completions",
                        json=prepare_stream_body(body),
                        headers={
                            "Authorization": "Bearer not-needed",
                            "Accept": "text/event-stream",
                            "Content-Type": "application/json",
                            "Cache-Control": "no-cache",
                        },
                    ),
                    stream=True,
                )
                break
            except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as exc:
                if attempt == 0:
                    await update_stats(backend_retries_inc=1)
                    continue
                raise exc
        if response is None:
            raise HTTPException(status_code=503, detail="Backend did not return a response")
        if response.status_code >= 400:
            error_body = (await response.aread()).decode("utf-8", errors="replace")
            await response.aclose()
            await http_client.aclose()
            await update_stats(backend_errors_inc=1, last_error=error_body)
            await record_backend_failure(error_body)
            raise HTTPException(status_code=response.status_code, detail=error_body)
        await record_backend_success()
        return http_client, response
    except Exception as exc:
        await http_client.aclose()
        if isinstance(exc, HTTPException):
            raise
        await update_stats(backend_errors_inc=1, last_error=str(exc))
        await record_backend_failure(str(exc))
        raise


async def stream_backend_raw(http_client: httpx.AsyncClient, response: httpx.Response):
    saw_done = False
    saw_finish = False
    try:
        async for chunk in response.aiter_raw():
            if not chunk:
                continue
            text = chunk.decode("utf-8", errors="ignore")
            if "data: [DONE]" in text:
                saw_done = True
            if '"finish_reason"' in text and '"finish_reason":null' not in text:
                saw_finish = True
            await update_stats(stream_bytes_inc=len(chunk))
            yield chunk
        if saw_done:
            await update_stats(stream_done_events_inc=1)
        else:
            await update_stats(stream_missing_done_inc=1, last_error="upstream stream ended without data: [DONE]")
        if saw_finish:
            await update_stats(stream_finish_chunks_inc=1)
    except httpx.ReadTimeout:
        message = f"Proxy upstream stream inactive for {STREAM_INACTIVITY_TIMEOUT_SECONDS:g}s"
        await update_stats(stream_timeouts_inc=1, backend_errors_inc=1, last_error=message)
        await record_backend_failure(message)
        yield sse_event(json.dumps({"error": {"message": message, "type": "stream_timeout"}})).encode("utf-8")
        yield sse_event("[DONE]").encode("utf-8")
    except Exception as exc:
        await update_stats(backend_errors_inc=1, last_error=str(exc))
        await record_backend_failure(str(exc))
        raise
    finally:
        await response.aclose()
        await http_client.aclose()


async def check_vllm_models() -> Dict[str, Any]:
    try:
        timeout = httpx.Timeout(connect=BACKEND_CONNECT_TIMEOUT_SECONDS, read=10.0, write=10.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout) as http_client:
            response = await http_client.get(f"{VLLM_URL.rstrip('/')}/models")
            data = response.json()
        model_ids = [item.get("id") for item in data.get("data", []) if isinstance(item, dict)]
        configured = sorted(model_profiles())
        return {
            "ok": response.status_code == 200,
            "status_code": response.status_code,
            "models": model_ids,
            "configured_models": configured,
            "configured_model_loaded": any((profile.get("backend_model") or name) in model_ids for name, profile in model_profiles().items()),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "models": [], "configured_models": sorted(model_profiles())}


async def check_streaming_probe() -> Dict[str, Any]:
    try:
        body = {
            "model": default_model_name(),
            "messages": [{"role": "user", "content": "Reply ok."}],
            "max_tokens": 4,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        http_client, response = await open_backend_stream(body)
        saw_done = False
        saw_finish = False
        try:
            async for chunk in response.aiter_raw():
                text = chunk.decode("utf-8", errors="ignore")
                saw_done = saw_done or "data: [DONE]" in text
                saw_finish = saw_finish or ('"finish_reason"' in text and '"finish_reason":null' not in text)
                if saw_done and saw_finish:
                    break
        finally:
            await response.aclose()
            await http_client.aclose()
        return {"ok": saw_done and saw_finish, "saw_done": saw_done, "saw_finish_reason": saw_finish}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/health")
async def health():
    vllm = await check_vllm_models()
    ok = bool(vllm.get("ok")) and CIRCUIT.get("state") != "open"
    return {
        "status": "ok" if ok else "degraded",
        "backend": VLLM_URL,
        "model": MODEL_NAME,
        "vllm": vllm,
        "circuit": CIRCUIT,
    }

@app.get("/stats")
async def stats():
    async with STATS_LOCK:
        stats_copy = dict(STATS)
        recent = list(RECENT_SEARCHES)
        requests = list(RECENT_REQUESTS)
    uptime = max(0.001, time.time() - stats_copy["started_at"])
    chat_requests = stats_copy.get("chat_requests", 0)
    web_search_calls = stats_copy.get("web_search_calls", 0)
    http_requests = stats_copy.get("http_requests", 0)
    http_5xx = stats_copy.get("http_5xx", 0)
    return {
        "status": "ok",
        "backend": VLLM_URL,
        "model": MODEL_NAME,
        "config_path": CONFIG_PATH,
        "listen": {"host": "0.0.0.0", "port": PROXY_PORT},
        "client_profiles": {
            "default": DEFAULT_CLIENT_PROFILE,
            "known": ["generic", "kilo", "copilot", "openwebui", "vscode"],
            "explicit_override": "x-client-profile header, x-llm-client-profile header, or extra_body.client_profile",
            "profile_output_caps": {
                "kilo": KILO_MAX_OUTPUT_TOKENS,
                "copilot": COPILOT_MAX_OUTPUT_TOKENS,
            },
        },
        "streaming": {
            "mode": "raw_sse_passthrough",
            "force_include_usage": FORCE_STREAM_USAGE,
            "inactivity_timeout_seconds": STREAM_INACTIVITY_TIMEOUT_SECONDS,
        },
        "request_limits": {
            "max_request_bytes": MAX_REQUEST_BYTES,
            "strict_unsupported_params": STRICT_UNSUPPORTED_PARAMS,
        },
        "circuit": CIRCUIT,
        "models": model_profiles(),
        "uptime_seconds": uptime,
        "counters": stats_copy,
        "derived": {
            "chat_requests_per_min": chat_requests / uptime * 60.0,
            "web_searches_per_min": web_search_calls / uptime * 60.0,
            "avg_chat_latency_ms": stats_copy.get("total_latency_ms", 0.0) / chat_requests if chat_requests else 0.0,
            "avg_search_latency_ms": stats_copy.get("total_search_latency_ms", 0.0) / web_search_calls if web_search_calls else 0.0,
            "avg_http_latency_ms": stats_copy.get("total_http_latency_ms", 0.0) / http_requests if http_requests else 0.0,
            "search_success_pct": stats_copy.get("web_search_success", 0) / web_search_calls * 100.0 if web_search_calls else 0.0,
            "http_5xx_pct": http_5xx / http_requests * 100.0 if http_requests else 0.0,
        },
        "last_request": requests[0] if requests else None,
        "recent_requests": requests,
        "recent_searches": recent,
        "recent_failures": list(RECENT_FAILURES),
    }


@app.get("/diagnostics")
async def diagnostics():
    async with STATS_LOCK:
        stats_copy = dict(STATS)
        recent_requests = list(RECENT_REQUESTS)
        recent_searches = list(RECENT_SEARCHES)
    vllm = await check_vllm_models()
    return {
        "status": "ok" if vllm.get("ok") and CIRCUIT.get("state") != "open" else "degraded",
        "backend": VLLM_URL,
        "active_model": MODEL_NAME,
        "config_path": CONFIG_PATH,
        "request_log_path": REQUEST_LOG_PATH,
        "metrics_path": METRICS_PATH,
        "streaming": {
            "mode": "raw_sse_passthrough",
            "force_include_usage": FORCE_STREAM_USAGE,
            "inactivity_timeout_seconds": STREAM_INACTIVITY_TIMEOUT_SECONDS,
        },
        "vllm": vllm,
        "tool_search": {
            "enabled": True,
            "tool_name": "web_search",
            "recent_searches": recent_searches[:5],
        },
        "circuit": CIRCUIT,
        "models": model_profiles(),
        "counters": stats_copy,
        "recent_requests": recent_requests[:10],
        "recent_failures": list(RECENT_FAILURES)[:10],
    }


@app.get("/diagnostics/stream")
async def diagnostics_stream():
    return await check_streaming_probe()


@app.get("/metrics")
async def prometheus_metrics():
    async with STATS_LOCK:
        stats_copy = dict(STATS)
    lines = [
        "# HELP spark_proxy_counter Proxy cumulative counters.",
        "# TYPE spark_proxy_counter counter",
    ]
    for key, value in sorted(stats_copy.items()):
        if isinstance(value, (int, float)) and key != "started_at":
            lines.append(f'spark_proxy_counter{{name="{key}"}} {float(value)}')
    lines.extend([
        "# HELP spark_proxy_circuit_open Whether the backend circuit breaker is open.",
        "# TYPE spark_proxy_circuit_open gauge",
        f"spark_proxy_circuit_open {1.0 if CIRCUIT.get('state') == 'open' else 0.0}",
        "# HELP spark_proxy_uptime_seconds Proxy uptime in seconds.",
        "# TYPE spark_proxy_uptime_seconds gauge",
        f"spark_proxy_uptime_seconds {max(0.0, time.time() - float(stats_copy.get('started_at', time.time())))}",
    ])
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

@app.get("/v1/models")
async def list_models():
    return await client.models.list()

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request):
    started = time.time()
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    await update_stats(chat_requests_inc=1)
    log_row = {"request_id": request_id, "ts": time.time(), "path": "/v1/chat/completions"}
    try:
        raw_bytes = await request.body()
        if len(raw_bytes) > MAX_REQUEST_BYTES:
            await update_stats(request_size_rejections_inc=1)
            raise HTTPException(status_code=413, detail=f"Request body is {len(raw_bytes)} bytes; limit is {MAX_REQUEST_BYTES} bytes")
        try:
            raw_body = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Malformed JSON request body: {exc}")

        detected_profile = detect_client_profile(request, raw_body)
        profile = detected_profile["name"]
        request.state.client_profile = profile
        await record_client_profile(profile)

        model_name, model_profile = resolve_model_profile(raw_body.get("model"))
        body = normalize_body_for_profile(sanitize_chat_body(raw_body), profile, model_profile)
        validation = validate_chat_request(raw_body, body, model_name, model_profile)
        messages = body.get("messages", [])
        log_row.update({
            "client_profile": profile,
            "client_profile_source": detected_profile["source"],
            "model": model_name,
            "backend_model": body.get("model"),
            "stream": bool(body.get("stream")),
            "prompt_tokens_estimated": validation["prompt_tokens_estimated"],
            "output_tokens": validation["output_tokens"],
            "max_context_tokens": validation["max_context_tokens"],
        })

        if body.get("stream"):
            await update_stats(stream_requests_inc=1)
            http_client, response = await open_backend_stream(body)
            await append_request_log({**log_row, "status": "stream_started"})
            return StreamingResponse(
                stream_backend_raw(http_client, response),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Request-ID": request_id,
                    "X-Client-Profile": profile,
                    "X-Client-Profile-Source": detected_profile["source"],
                },
            )

        if not await circuit_allows_request():
            await update_stats(circuit_open_rejections_inc=1)
            raise HTTPException(status_code=503, detail=f"Backend circuit breaker is open: {CIRCUIT.get('last_error')}")

        if model_profile.get("supports_tools", True) and profile != "benchmark":
            body["tools"] = TOOLS
            body["tool_choice"] = "auto"

        while True:
            try:
                await update_stats(backend_requests_inc=1)
                response = await client.chat.completions.create(**body)
                await record_backend_success()
            except Exception as exc:
                await update_stats(backend_errors_inc=1, last_error=str(exc))
                await record_backend_failure(str(exc))
                raise
            assistant_msg = response.choices[0].message

            if assistant_msg.tool_calls:
                await update_stats(tool_rounds_inc=1)
                messages.append(assistant_msg)

                for tool_call in assistant_msg.tool_calls:
                    if tool_call.function.name == "web_search":
                        try:
                            args = json.loads(tool_call.function.arguments)
                        except Exception as exc:
                            await update_stats(validation_errors_inc=1)
                            raise HTTPException(status_code=400, detail=f"Malformed tool call arguments: {exc}")
                        search_result = await perform_web_search(args["query"])

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": search_result
                        })

                body["messages"] = messages
                continue
            latency_ms = (time.time() - started) * 1000.0
            await update_stats(total_latency_ms_inc=latency_ms)
            payload = response.model_dump(exclude_none=True)
            finish_reason = payload.get("choices", [{}])[0].get("finish_reason")
            await append_request_log({**log_row, "status": "ok", "latency_ms": latency_ms, "finish_reason": finish_reason})
            return JSONResponse(payload, headers={"X-Request-ID": request_id, "X-Client-Profile": profile})
    except Exception as exc:
        latency_ms = (time.time() - started) * 1000.0
        text = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
        if "maximum context length" in text.lower() or "estimated total" in text.lower():
            await update_stats(context_overflows_inc=1)
        elif isinstance(exc, HTTPException) and exc.status_code == 400:
            await update_stats(validation_errors_inc=1)
        await append_request_log({**log_row, "status": "error", "latency_ms": latency_ms, "error": text})
        return exception_to_openai_error(exc)

if __name__ == "__main__":
    print(f"Starting Search Proxy on port {PROXY_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
