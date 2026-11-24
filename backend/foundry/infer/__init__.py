"""推理层。

    params.py   推理参数（frozen，per-session）
    prompt.py   prompt 构建（含 thinking 还原）
    prefill.py  ★ 增量 prefill —— 技术护城河，纯函数可独立测试
    sampler.py  采样参数适配 + 跨 chunk stop 序列检测
    session.py  会话隔离与引擎（单模型、串行推理）

边界：**本层不碰数据库、不碰 HTTP。**
"""

from __future__ import annotations

from foundry.infer.params import InferParams
from foundry.infer.prefill import (
    PrefillPlan,
    longest_common_prefix,
    plan_prefill,
    rebuild_fed_text,
    token_prefix_chars,
)
from foundry.infer.prompt import (
    PRESYSTEM_PREFIX,
    ChatMessage,
    build_prompt,
    count_rounds,
    split_thinking,
)
from foundry.infer.sampler import (
    STOP_SEQUENCES,
    StopSequenceFilter,
    make_pipeline_args,
)
from foundry.infer.session import (
    GenResult,
    InferenceEngine,
    InferError,
    ModelNotLoadedError,
    SessionState,
    SessionStore,
)

__all__ = [
    # params
    "InferParams",
    # prefill
    "PrefillPlan",
    "longest_common_prefix",
    "plan_prefill",
    "rebuild_fed_text",
    "token_prefix_chars",
    # prompt
    "PRESYSTEM_PREFIX",
    "ChatMessage",
    "build_prompt",
    "count_rounds",
    "split_thinking",
    # sampler
    "STOP_SEQUENCES",
    "StopSequenceFilter",
    "make_pipeline_args",
    # session
    "GenResult",
    "InferError",
    "InferenceEngine",
    "ModelNotLoadedError",
    "SessionState",
    "SessionStore",
]
