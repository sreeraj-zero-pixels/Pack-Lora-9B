import asyncio
import os
import sys

import pytest
from aiohttp import web

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import handler


def test_openai_input_passthrough_fills_model():
    route, method, body = handler.build_request({
        "openai_input": {"messages": [{"role": "user", "content": "hi"}]},
    })
    assert route == "/v1/chat/completions"
    assert method == "POST"
    assert body["model"] == handler.SERVED_MODEL_NAME
    assert body["messages"][0]["content"] == "hi"


def test_openai_input_respects_route_and_model():
    route, method, body = handler.build_request({
        "openai_route": "/v1/embeddings",
        "openai_input": {"model": "custom", "input": "x"},
    })
    assert route == "/v1/embeddings"
    assert body["model"] == "custom"


def test_bare_openai_route_is_get():
    route, method, body = handler.build_request({"openai_route": "/v1/models"})
    assert (route, method, body) == ("/v1/models", "GET", None)


def test_route_with_body_posts_and_fills_model():
    route, method, body = handler.build_request({
        "route": "/v1/completions", "body": {"prompt": "hi"},
    })
    assert method == "POST"
    assert body["model"] == handler.SERVED_MODEL_NAME


def test_messages_shorthand_merges_sampling_params():
    route, method, body = handler.build_request({
        "messages": [{"role": "user", "content": "hi"}],
        "sampling_params": {"temperature": 0.2, "max_tokens": 64},
    })
    assert route == "/v1/chat/completions"
    assert method == "POST"
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 64
    assert body["model"] == handler.SERVED_MODEL_NAME
    assert body["stream"] is False
    assert body["messages"][0]["role"] == "user"


def test_stream_flag_top_level():
    _, _, body = handler.build_request({
        "messages": [{"role": "user", "content": "hi"}], "stream": True,
    })
    assert body["stream"] is True


def test_chat_template_kwargs_passed_through():
    _, _, body = handler.build_request({
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"enable_thinking": False},
    })
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_max_new_tokens_maps_to_max_tokens():
    _, _, body = handler.build_request({
        "messages": [{"role": "user", "content": "hi"}],
        "max_new_tokens": 32,
    })
    assert body["max_tokens"] == 32
    assert "max_new_tokens" not in body

    _, _, body = handler.build_request({
        "messages": [{"role": "user", "content": "hi"}],
        "sampling_params": {"max_new_tokens": 48},
    })
    assert body["max_tokens"] == 48
    assert "max_new_tokens" not in body


def test_prompt_shorthand_uses_completions():
    route, method, body = handler.build_request({"prompt": "once upon a"})
    assert route == "/v1/completions"
    assert body["prompt"] == "once upon a"


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        handler.build_request({"foo": "bar"})


CHAT_RESPONSE = {
    "id": "chatcmpl-1",
    "choices": [{"index": 0, "message": {
        "role": "assistant",
        "content": "MDM keeps core business data consistent.",
        "reasoning_content": "thinking...",
    }}],
}


async def _run_against_fake_vllm(monkeypatch, job):
    """Tiny aiohttp server standing in for vLLM; collect handler() output."""
    async def chat(request):
        return web.json_response(CHAT_RESPONSE)

    async def boom(request):
        return web.Response(status=500, text="internal boom")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/boom", boom)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setattr(handler, "VLLM_BASE_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setattr(handler, "_is_vllm_alive", lambda: True)
    try:
        return [item async for item in handler.handler(job)]
    finally:
        await runner.cleanup()


def test_handler_chat_returns_text_convenience(monkeypatch):
    job = {"input": {"messages": [{"role": "user", "content": "hi"}]}}
    results = asyncio.run(_run_against_fake_vllm(monkeypatch, job))
    assert len(results) == 1
    assert results[0]["text"] == "MDM keeps core business data consistent."
    assert results[0]["reasoning_content"] == "thinking..."
    assert results[0]["choices"][0]["message"]["content"] == results[0]["text"]


def test_handler_http_500_yields_error(monkeypatch):
    job = {"input": {"route": "/boom", "body": {"x": 1}}}
    results = asyncio.run(_run_against_fake_vllm(monkeypatch, job))
    assert len(results) == 1
    assert "error" in results[0]
    assert "HTTP 500" in results[0]["error"]["message"]
    assert "internal boom" in results[0]["error"]["message"]
