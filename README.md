# Spark Proxy Gateway

OpenAI-compatible proxy gateway for local vLLM models on Spark nodes.

The gateway sits between clients such as Kilo Code, Copilot LLM Gateway, Open WebUI, and direct benchmark scripts, then forwards requests to vLLM with model-aware validation and raw SSE streaming passthrough.

## Features

- Raw SSE passthrough for `stream: true`
- Automatic client profile detection
- Model profile table in `models.yaml`
- Preflight validation for model name, context size, output caps, request size, and unsupported parameters
- OpenAI-style translated errors
- Safe retry before stream start only
- Circuit breaker for repeated backend failures
- Kilo request-storm protection with one active stream per node by default
- Kilo-specific 128K context and 4K output safety limits
- Per-request IDs and JSONL structured logs
- Persisted metrics and Prometheus `/metrics`
- Diagnostics endpoints for vLLM health, streaming, model load, and recent failures

## Files

- `search_proxy.py` - FastAPI proxy gateway
- `models.yaml` - model and client-profile configuration
- `.env.example` - runtime configuration template
- `requirements.txt` - Python dependencies
- `start_proxy.sh` - local start helper

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
./start_proxy.sh
```

The default proxy listens on port `8081` and expects vLLM at `http://127.0.0.1:8080/v1`.

## Client Endpoints

Use the `/v1` base URL from your client:

```text
http://<node-ip>:8081/v1
```

Useful checks:

```bash
curl http://127.0.0.1:8081/health
curl http://127.0.0.1:8081/diagnostics
curl http://127.0.0.1:8081/diagnostics/stream
curl http://127.0.0.1:8081/metrics
```

## Client Profiles

Profiles are detected automatically from headers and request shape. You can force a profile with:

```text
X-Client-Profile: kilo
X-Client-Profile: copilot
X-Client-Profile: benchmark
```

Supported profiles:

- `generic`
- `kilo`
- `copilot`
- `openwebui`
- `vscode`
- `benchmark`

## Runtime Data

By default, logs and metrics go under `./data/` when started via `start_proxy.sh`.

- `data/proxy_requests.jsonl`
- `data/proxy_metrics.json`

These are ignored by git.
