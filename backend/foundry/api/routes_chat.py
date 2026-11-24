"""对话路由 —— SSE 流式生成。

## 旧版的两个并发缺陷，这里都修掉

### 1. ``asyncio.Queue`` 跨线程不安全

旧版 ``chat.py:79-81``::

    def sync_callback(token, perplexity=None):
        queue.put_nowait((token, perplexity))   # 在 threading.Thread 里调用

    # 消费端
    token, pp = await asyncio.wait_for(queue.get(), timeout=0.1)

``asyncio.Queue`` 不是线程安全的。从工作线程 put 不会唤醒挂起的
``await queue.get()``，只能靠 100ms 超时轮询兜底 —— **流式输出被硬节流到
约 20 token/s**。

本版改用 ``asyncio.run_coroutine_threadsafe`` + ``loop.call_soon_threadsafe``
投递到事件循环，无轮询、无延迟。

### 2. 生成线程无法取消

旧版前端断开后，generator 被 dispose，但 ``gen_thread`` 仍会跑到
``max_tokens``，白烧 GPU。

本版通过 ``should_stop`` 回调把客户端断开的信号传给引擎，引擎在
``callback`` 里抛 ``_StopGeneration`` 提前退出，并正确保存 state。
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from foundry.api import service
from foundry.api.deps import AppState, get_state
from foundry.api.schemas import (
    ChatRequest,
    SessionCreateRequest,
    SessionResponse,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])

#: 事件循环停止时的哨兵
_STOP = object()


def _sse(payload: Any) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _run_stream(
    state: AppState,
    request: ChatRequest,
    *,
    regenerate: bool,
) -> AsyncIterator[str]:
    """把同步生成器的 token 流转换为 SSE。

    线程安全要点：
      - 工作线程只通过 ``loop.call_soon_threadsafe`` 与事件循环交互
      - 客户端断开时置 ``cancel`` 事件，引擎会尽快退出
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    cancel = threading.Event()
    error_holder: list[str] = []

    def emit(item: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def on_token(chunk: str) -> None:
        emit({"content": chunk})

    def worker() -> None:
        try:
            if regenerate:
                result = service.regenerate(
                    state,
                    model_ref=request.model_ref,
                    message=request.message,
                    history=[m.model_dump() for m in request.history],
                    params=request.params,
                    thinking_mode=request.thinking_mode,
                    session_name=request.session_name,
                    role_id=request.role_id,
                    on_token=on_token,
                    should_stop=cancel.is_set,
                )
            else:
                result = service.chat(
                    state,
                    model_ref=request.model_ref,
                    message=request.message,
                    history=[m.model_dump() for m in request.history],
                    params=request.params,
                    thinking_mode=request.thinking_mode,
                    session_name=request.session_name,
                    role_id=request.role_id,
                    on_token=on_token,
                    should_stop=cancel.is_set,
                )
            emit({"result": result})
        except Exception as exc:  # noqa: BLE001
            error_holder.append(str(exc))
            emit({"error": str(exc)})
        finally:
            emit(_STOP)

    threading.Thread(target=worker, daemon=True, name="foundry-chat").start()

    try:
        while True:
            item = await queue.get()
            if item is _STOP:
                break
            yield _sse(item)
    except asyncio.CancelledError:
        # 客户端断开：请求引擎停止，避免生成线程白跑到 max_tokens
        cancel.set()
        raise
    finally:
        cancel.set()

    yield "data: [DONE]\n\n"


@router.post("/chat")
async def chat(request: ChatRequest, state: AppState = Depends(get_state)) -> StreamingResponse:
    """流式对话。"""
    return StreamingResponse(
        _run_stream(state, request, regenerate=False),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲，否则流式失效
        },
    )


@router.post("/regenerate")
async def regenerate(request: ChatRequest, state: AppState = Depends(get_state)) -> StreamingResponse:
    """重生成：回滚 state 后重新生成，让模型"忘记"上一轮回复。"""
    return StreamingResponse(
        _run_stream(state, request, regenerate=True),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/preload-model")
def preload_model(model_ref: str, state: AppState = Depends(get_state)) -> dict:
    """预加载模型，避免首次对话等待。"""
    resolved = service.ensure_model_loaded(state, model_ref)
    return {"message": "模型加载成功", "model": resolved}


@router.post("/reset-state")
def reset_state(
    model_ref: str, session: str = "", state: AppState = Depends(get_state)
) -> dict:
    """清空该会话的模型循环态（让模型彻底忘记历史）。"""
    service.reset_state(state, model_ref, session)
    return {"message": "模型状态已重置"}


@router.get("/sessions", response_model=list[SessionResponse])
def list_sessions(
    model_ref: str | None = None, state: AppState = Depends(get_state)
) -> list[dict]:
    from foundry.storage import repo

    rows = repo.list_sessions(state.db, model_ref=model_ref)
    out = []
    for r in rows:
        try:
            params = json.loads(r["params"] or "{}")
        except json.JSONDecodeError:
            params = {}
        out.append({**r, "params": params})
    return out


@router.post("/sessions")
def create_session(req: SessionCreateRequest, state: AppState = Depends(get_state)) -> dict:
    from foundry.storage import repo

    sid = repo.create_session(
        state.db,
        model_ref=req.model_ref,
        name=req.name,
        role_id=req.role_id,
        params=json.dumps(req.params, ensure_ascii=False),
    )
    return {"id": sid, "message": "会话已创建"}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: int, state: AppState = Depends(get_state)) -> dict:
    from foundry.storage import repo

    repo.delete_session(state.db, session_id)
    return {"message": "会话已删除"}
