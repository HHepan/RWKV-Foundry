"""会话状态与推理引擎。

## 设计约束（老大明确定下的）

> 没有多并发需求，一次只加载并测试一个对话。但不同对话之间信息隔离要做好。

这句话把架构直接定死了，而且是**正确的**定死：

- **不需要并发推理** → 不做任务队列、不做并发调度、不做多路复用。
  一张卡本来也无法真并行，假装支持并发只会引入一堆竞态。
- **必须隔离** → state / 对话历史 / 推理参数 全部 per-session，
  引擎内部**不保存任何"当前会话"或"当前参数"**。

## 旧版的串味根源

``RWKVInferenceManager`` 是单例，且：

    self.current_params         共享可变
    self.last_response          共享可变
    self._session_states.clear() 换模型时清掉所有会话的 state

旧版 ``chat.py:123`` 用 ``manager.get_last_response()`` 取回复再落盘。
两个会话交替时，A 的请求可能把 B 的回复存进 A 的对话文件。

本版：``generate()`` **直接返回**结果，引擎不留"最后一次回复"。
没有共享状态，串味在结构上就不存在。

## 隔离单元

``SessionState`` 持有::

    state / occurrence              模型循环态（每个会话独立）
    fed_text                        增量 prefill 基线
    checkpoint_state / checkpoint_* 重生成用的回滚快照
    messages                        对话历史
    params                          推理参数
    round_count                     显存回收计数

``SessionStore`` 负责按 key 存取，并在模型切换时整体作废。
"""

from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from foundry.errors import FoundryError
from foundry.infer.params import InferParams
from foundry.infer.prefill import rebuild_fed_text
from foundry.infer.prompt import ChatMessage, build_prompt, split_thinking


class InferError(FoundryError):
    """推理失败。"""


class ModelNotLoadedError(InferError):
    """未加载模型就请求生成。"""


class GenResult:
    """一次生成的结果。**不保存在引擎上**，由调用方持有。"""

    __slots__ = ("raw", "thinking", "content", "stopped", "tokens_fed", "tokens_reused")

    def __init__(
        self,
        *,
        raw: str,
        thinking: str,
        content: str,
        stopped: bool,
        tokens_fed: int,
        tokens_reused: int,
    ):
        self.raw = raw
        self.thinking = thinking
        self.content = content
        self.stopped = stopped
        self.tokens_fed = tokens_fed
        self.tokens_reused = tokens_reused

    def as_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "thinking": self.thinking,
            "stopped": self.stopped,
            "tokens_fed": self.tokens_fed,
            "tokens_reused": self.tokens_reused,
        }


class _StopGeneration(Exception):
    """内部信号：调用方要求提前结束生成。

    ## 为什么不用 StopIteration

    ``StopIteration`` 在 Python 里有特殊语义：如果它从 generator 内部逃逸，
    解释器会把它替换成 ``RuntimeError("generator raised StopIteration")``。
    用它当控制流信号，行为取决于上游 ``generate`` 的实现细节，极不可靠。

    上游 RWKV 的 ``pipeline.generate`` 会 ``except StopIteration`` 并正常返回，
    但我们的 fake/未来实现不一定如此。用专用异常类语义明确，且上游的
    ``except StopIteration`` 不会误吞它——因此本层必须自己拦。
    """


class _Pipeline(Protocol):
    """``rwkv.utils.PIPELINE`` 的结构（便于测试注入 fake）。"""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


@dataclass
class SessionState:
    """一个会话的全部可变状态。**这是隔离边界。**"""

    key: str
    params: InferParams
    messages: list[ChatMessage] = field(default_factory=list)

    #: 模型循环态
    state: Any = None
    occurrence: Any = None

    #: 增量 prefill 基线：state 中已吃进的完整文本
    fed_text: str = ""

    #: 重生成用的回滚快照
    checkpoint_state: Any = None
    checkpoint_occurrence: Any = None
    checkpoint_fed_text: str = ""

    round_count: int = 0

    def snapshot_for_rollback(self) -> None:
        """在生成前保存快照。

        ``occurrence`` 每轮重置，但 state 必须深拷贝——``model.forward``
        会原地修改传入的 state（RWKV 的 state 是 list of tensors，
        不是不可变结构）。
        """
        self.checkpoint_state = copy.deepcopy(self.state)
        self.checkpoint_occurrence = copy.deepcopy(self.occurrence)
        self.checkpoint_fed_text = self.fed_text

    def rollback(self) -> None:
        """回滚到生成前，让模型"忘记"上一轮回复。"""
        self.state = self.checkpoint_state
        self.occurrence = self.checkpoint_occurrence
        self.fed_text = self.checkpoint_fed_text

    def reset(self) -> None:
        """彻底清空，让模型忘掉全部历史。"""
        self.state = None
        self.occurrence = None
        self.fed_text = ""
        self.checkpoint_state = None
        self.checkpoint_occurrence = None
        self.checkpoint_fed_text = ""
        self.round_count = 0


class SessionStore:
    """会话状态容器，带 LRU 上限。

    **不保存模型**，只保存 state。模型由 InferenceEngine 单独持有。
    """

    def __init__(self, *, max_sessions: int = 32):
        if max_sessions <= 0:
            raise ValueError(f"max_sessions 必须为正整数，当前 {max_sessions}")
        self._max = max_sessions
        self._items: OrderedDict[str, SessionState] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str, *, params: InferParams) -> SessionState:
        """取会话；不存在则用给定参数新建。"""
        with self._lock:
            existing = self._items.get(key)
            if existing is not None:
                self._items.move_to_end(key)
                return existing
            created = SessionState(key=key, params=params)
            self._items[key] = created
            self._evict_locked()
            return created

    def put(self, state: SessionState) -> None:
        with self._lock:
            self._items[state.key] = state
            self._items.move_to_end(state.key)
            self._evict_locked()

    def drop(self, key: str) -> bool:
        with self._lock:
            return self._items.pop(key, None) is not None

    def clear(self) -> None:
        """模型切换时整体作废——不同模型的 state 绝不通用。"""
        with self._lock:
            self._items.clear()

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def _evict_locked(self) -> None:
        while len(self._items) > self._max:
            self._items.popitem(last=False)


class InferenceEngine:
    """模型与推理的持有者。

    **单模型、串行推理。** 一把锁保护生成过程，因为：

    1. 一张卡无法真并行
    2. RWKV 的 state 被 forward 原地修改，并发必炸
    3. 串行 + 明确拒绝，比假装并发诚实
    """

    def __init__(self, *, max_sessions: int = 32):
        self.sessions = SessionStore(max_sessions=max_sessions)
        self._pipeline: _Pipeline | None = None
        self._model_ref: str | None = None
        self._tokenizer: Any = None

        #: 生成锁。整个 generate() 期间持有。
        self._gen_lock = threading.Lock()
        self._load_lock = threading.Lock()

    # ── 模型生命周期 ────────────────────────────────────────────────────
    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def model_ref(self) -> str | None:
        return self._model_ref

    def attach(self, pipeline: _Pipeline, tokenizer: Any, *, model_ref: str) -> None:
        """绑定一个已就绪的 pipeline（由 loader 或测试注入）。

        换模型时**清空全部会话 state**——state 与权重强相关，绝不通用。
        """
        with self._load_lock:
            changed = self._model_ref != model_ref
            self._pipeline = pipeline
            self._tokenizer = tokenizer
            self._model_ref = model_ref
            if changed:
                self.sessions.clear()

    def detach(self) -> None:
        with self._load_lock:
            self._pipeline = None
            self._tokenizer = None
            self._model_ref = None
            self.sessions.clear()

    def _require_pipeline(self) -> _Pipeline:
        if self._pipeline is None:
            raise ModelNotLoadedError("模型尚未加载，请先调用 load_model")
        return self._pipeline

    # ── 会话操作 ────────────────────────────────────────────────────────
    def get_session(self, key: str, *, params: InferParams | None = None) -> SessionState:
        return self.sessions.get(key or "default", params=params or InferParams.default())

    def reset_session(self, key: str) -> bool:
        """清空模型态但保留对话历史与参数。"""
        state = self.sessions.get(key or "default", params=InferParams.default())
        state.reset()
        return True

    def drop_session(self, key: str) -> bool:
        return self.sessions.drop(key or "default")

    # ── 生成 ────────────────────────────────────────────────────────────
    def generate(
        self,
        *,
        session_key: str,
        user_message: str,
        history: list[ChatMessage],
        params: InferParams,
        thinking_mode: bool = False,
        on_token: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        rollback: bool = False,
    ) -> GenResult:
        """生成回复。

        **参数显式传入，引擎不保存任何"当前参数"。** 这是隔离的关键。

        :param session_key: 会话标识（不同会话 state 互不影响）
        :param history: 该会话的**历史**消息，**不含**本轮用户消息。
            本轮消息由 ``user_message`` 追加。这与前端 regenerate 的行为一致：
            前端会先移除上轮 assistant 回复，再以 "不新增消息" 的方式重发。
        :param params: 本次生成使用的参数
        :param rollback: 重生成时先回滚到上一轮生成前的 state
        :param should_stop: 前端断开时返回 True，尽早退出
        """
        pipeline = self._require_pipeline()
        params.validate()

        key = session_key or "default"
        session = self.sessions.get(key, params=params)

        # 每次生成都同步参数与历史，确保会话状态与调用方一致
        session.params = params
        session.messages = list(history)

        with self._gen_lock:
            if rollback:
                session.rollback()

            messages = [*history, ChatMessage(role="user", content=user_message)]
            prompt = build_prompt(messages, params=params, thinking_mode=thinking_mode)

            # ── 增量 prefill ────────────────────────────────────────────
            from foundry.infer.prefill import plan_prefill

            plan = plan_prefill(
                prompt=prompt,
                fed_text=session.fed_text,
                has_state=session.state is not None,
                encode=pipeline.encode,
            )

            if plan.full_rebuild:
                # 前缀不匹配，旧 state 不可复用
                session.state = None
                session.occurrence = None

            # 保存回滚快照（在 state 被修改之前）
            session.snapshot_for_rollback()

            raw, new_state, new_occurrence = self._run(
                pipeline=pipeline,
                feed_tokens=plan.feed_tokens,
                state=session.state,
                params=params,
                on_token=on_token,
                should_stop=should_stop,
            )

            # 提交新状态
            session.state = new_state
            session.occurrence = None  # occurrence 每轮重置，防跨轮惩罚过载
            session.fed_text = rebuild_fed_text(prompt=prompt, raw_output=raw)
            session.round_count += 1

            thinking, content = split_thinking(raw)
            return GenResult(
                raw=raw,
                thinking=thinking,
                content=content,
                stopped=bool(should_stop and should_stop()),
                tokens_fed=plan.fed_tokens,
                tokens_reused=plan.reused_tokens,
            )

    def _run(
        self,
        *,
        pipeline: Any,
        feed_tokens: list[int],
        state: Any,
        params: InferParams,
        on_token: Callable[[str], None] | None,
        should_stop: Callable[[], bool] | None,
    ) -> tuple[str, Any, Any]:
        """调用上游 pipeline.generate。

        上游签名::

            generate(ctx, token_count, args, callback, state, occurrence)
                -> (out_str, state, occurrence)

        ``callback`` 抛 ``StopIteration`` 可提前终止，上游内部会捕获它并
        返回当前 state —— 这正是我们能正确保存中间态的原因。

        ## 为什么这里不能再捕获 StopIteration

        上游已经处理了这个异常并正常返回。如果我们在外层也 ``except
        StopIteration``，就会把**正常的提前停止**误判成故障。
        本层只捕获真正的错误。
        """
        from foundry.infer.sampler import make_pipeline_args

        args = make_pipeline_args(params)

        def callback(chunk: str) -> None:
            if should_stop is not None and should_stop():
                raise _StopGeneration
            if on_token is not None:
                on_token(chunk)

        try:
            raw, new_state, new_occ = pipeline.generate(
                feed_tokens,
                token_count=params.max_tokens,
                args=args,
                callback=callback if (on_token or should_stop) else None,
                state=state,
                occurrence=None,
            )
        except _StopGeneration:
            # 上游兼容 StopIteration 的实现会先返回，走不到这里；
            # 不兼容的实现会把它冒出来，此时用已有的 state 收尾。
            return "", state, None
        except Exception as exc:  # noqa: BLE001
            raise InferError(f"生成失败: {exc}") from exc

        return raw or "", new_state, new_occ
