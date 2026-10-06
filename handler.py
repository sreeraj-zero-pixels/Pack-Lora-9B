"""RunPod Serverless handler that proxies jobs to a local vLLM OpenAI server.

At startup (in __main__): the LoRA adapter is merged into the base model via
src/model_setup.py, `vllm serve` is spawned on 127.0.0.1:VLLM_PORT, and the
RunPod serverless loop starts once vLLM is healthy. Importing this module
starts nothing, so tests can import it freely.

Accepted job input shapes (all under job["input"]):

1. RunPod OpenAI passthrough (what the platform sends on /openai/v1/...):
       {"openai_route": "/v1/chat/completions", "openai_input": {...}}
2. Generic proxy to any vLLM route:
       {"route": "/v1/completions", "body": {...}, "method": "POST"}
3. Shorthand:
       {"prompt": "...", "sampling_params": {...}, "stream": false}
       {"messages": [...], "sampling_params": {...}, "stream": true}

"sampling_params" is merged into the request body; "max_new_tokens" (top level
or inside sampling_params) maps to "max_tokens"; "chat_template_kwargs" is
passed through; "stream": true yields raw SSE chunks.
"""

import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, AsyncGenerator, Optional, Tuple

import aiohttp
import runpod

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import model_setup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("handler")

VLLM_PORT = os.getenv("VLLM_PORT", "8000")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", f"http://127.0.0.1:{VLLM_PORT}")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "qwen3.5-9b-pack-lora")
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "16384")
GPU_MEMORY_UTILIZATION = os.getenv("GPU_MEMORY_UTILIZATION", "0.90")
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "16"))
VLLM_STARTUP_TIMEOUT = float(os.getenv("VLLM_STARTUP_TIMEOUT", "1800"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "3600"))
VLLM_EXTRA_ARGS = shlex.split(os.getenv("VLLM_EXTRA_ARGS", ""))

DEFAULT_CHAT_ROUTE = "/v1/chat/completions"
DEFAULT_COMPLETION_ROUTE = "/v1/completions"

vllm_process: Optional[subprocess.Popen] = None


def start_vllm(model_dir: str) -> subprocess.Popen:
    """Spawn `vllm serve` for the merged model on 127.0.0.1:VLLM_PORT."""
    cmd = [
        "vllm", "serve", model_dir,
        "--host", "127.0.0.1",
        "--port", VLLM_PORT,
        "--served-model-name", SERVED_MODEL_NAME,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        *VLLM_EXTRA_ARGS,
    ]
    log.info("Starting vLLM: %s", " ".join(cmd))
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, env=env)

    def _term(signum, _frame):
        log.info("Signal %s received; terminating vLLM", signum)
        proc.terminate()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    return proc


def wait_for_health(proc: subprocess.Popen) -> None:
    """Poll vLLM /health until 200; raise on process exit or timeout."""
    import urllib.request
    import urllib.error

    deadline = time.time() + VLLM_STARTUP_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{VLLM_BASE_URL}/health", timeout=5) as resp:
                if resp.status == 200:
                    log.info("vLLM healthy after %.1fs", VLLM_STARTUP_TIMEOUT - (deadline - time.time()))
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become healthy within {VLLM_STARTUP_TIMEOUT}s")


def _is_vllm_alive() -> bool:
    return vllm_process is None or vllm_process.poll() is None


def build_request(job_input: dict) -> Tuple[str, str, Optional[dict]]:
    """Return (route, method, body) for any accepted job input shape."""
    if job_input.get("openai_input"):
        route = job_input.get("openai_route") or DEFAULT_CHAT_ROUTE
        body = dict(job_input["openai_input"])
        if "model" not in body:
            body["model"] = SERVED_MODEL_NAME
        return route, "POST", body

    if job_input.get("openai_route"):
        return job_input["openai_route"], "GET", None

    if job_input.get("route"):
        body = job_input.get("body")
        if body is not None and "model" not in body:
            body = {**body, "model": SERVED_MODEL_NAME}
        method = (job_input.get("method") or ("POST" if body else "GET")).upper()
        return job_input["route"], method, body

    messages = job_input.get("messages")
    prompt = job_input.get("prompt")
    if messages is None and prompt is None:
        raise ValueError(
            "Job input must contain one of: openai_input (+openai_route), "
            "route (+body), or prompt/messages."
        )

    sampling_params = dict(job_input.get("sampling_params") or {})
    max_new_tokens = job_input.get("max_new_tokens", sampling_params.pop("max_new_tokens", None))
    if max_new_tokens is not None:
        sampling_params["max_tokens"] = max_new_tokens
    body = {
        **sampling_params,
        "model": SERVED_MODEL_NAME,
        "stream": bool(job_input.get("stream", False)),
    }
    if job_input.get("chat_template_kwargs"):
        body["chat_template_kwargs"] = job_input["chat_template_kwargs"]
    if messages is not None:
        body["messages"] = messages
        return DEFAULT_CHAT_ROUTE, "POST", body
    body["prompt"] = prompt
    return DEFAULT_COMPLETION_ROUTE, "POST", body


def _error(message: str, error_type: str = "worker_error") -> dict:
    return {"error": {"message": message, "type": error_type, "code": None}}


def _add_convenience_fields(result: Any, route: str, body: Optional[dict]) -> Any:
    """For shorthand non-stream requests, expose text/reasoning_content at top level."""
    if not isinstance(result, dict) or "choices" not in result:
        return result
    choices = result.get("choices") or []
    if not choices:
        return result
    choice = choices[0] or {}
    message = choice.get("message") or {}
    text = message.get("content") or choice.get("text")
    if text is not None:
        result["text"] = text
    if message.get("reasoning_content") is not None:
        result["reasoning_content"] = message["reasoning_content"]
    return result


def _is_shorthand(job_input: dict) -> bool:
    return not (job_input.get("openai_input") or job_input.get("openai_route") or job_input.get("route"))


async def handler(job: dict) -> AsyncGenerator[Any, None]:
    job_input = job.get("input") or {}

    try:
        route, method, body = build_request(job_input)
    except ValueError as e:
        yield _error(str(e), error_type="invalid_input")
        return

    if not _is_vllm_alive():
        yield _error("vLLM server process is not running; worker is unhealthy")
        return

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, f"{VLLM_BASE_URL}{route}", json=body) as resp:
                if resp.status >= 400:
                    detail = await resp.text()
                    log.error("vLLM %s %s returned HTTP %s: %s", method, route, resp.status, detail)
                    yield _error(f"vLLM returned HTTP {resp.status}: {detail}")
                    return

                wants_stream = isinstance(body, dict) and body.get("stream") is True
                if wants_stream:
                    async for chunk in resp.content.iter_any():
                        yield chunk.decode("utf-8", errors="replace")
                else:
                    result = await resp.json(content_type=None)
                    if _is_shorthand(job_input):
                        result = _add_convenience_fields(result, route, body)
                    yield result
    except aiohttp.ClientError as e:
        log.exception("Request to vLLM failed")
        yield _error(f"Request to vLLM failed: {e}")


if __name__ == "__main__":
    try:
        model_dir = model_setup.prepare_merged_model()
        vllm_process = start_vllm(model_dir)
        wait_for_health(vllm_process)
    except Exception:
        log.exception("Startup failed")
        sys.exit(1)

    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": lambda _: MAX_CONCURRENCY,
        "return_aggregate_stream": True,
    })
