"""Prompt 构建 —— 纯函数，无副作用，可独立测试。

## 必须原样保留的两个关键点

### 1. 历史 thinking 还原

RWKV 是循环模型，state 里存的是**带 ``<think>`` 标记的完整 token 序列**。
如果重建 prompt 时只放正式回答（丢掉 think 部分），会出现两个问题：

- prompt 前缀与 state 里已吃进的 token 序列失配 → 增量 prefill 完全失效
  （只能复用极少 token，退化成每轮全量重算 O(n²)）
- 模型看到的"自己说过的话"与实际生成过的不一致 → 后续推理质量下降

因此 assistant 的历史回复必须还原成 ``<think>\\n...\\n</think>\\n<正文>``。

### 2. 默认不裁剪历史

``max_rounds=0`` 时保留全部对话。裁剪会让 prompt 开头每轮都变，
增量 prefill 的长公共前缀消失——这是拿 O(n) 换 O(n²)，反向优化。

## 格式

对齐 RWKV-7 G1x 与 RWKV-Runner 的 TextRWKV::

    User: hi

    Assistant: Hi. I am your assistant ...

    System: <角色卡>

    User: 消息

    Assistant: 回复

    User: 消息

    Assistant:
"""

from __future__ import annotations

from dataclasses import dataclass

from foundry.config import THINK_END, THINK_START
from foundry.infer.params import InferParams

#: 对齐 RWKV-Runner 的 __preload 示范对话。
#:
#: Runner 在加载模型时先用这段对话喂一遍 state，让模型进入「详细、专业回答」
#: 的模式。这里作为 prompt 前缀做等价效果。
PRESYSTEM_PREFIX = (
    "User: hi\n\n"
    "Assistant: Hi. I am your assistant and I will provide expert full response "
    "in full details. Please feel free to ask any question and I will always answer it.\n\n"
)


@dataclass(frozen=True)
class ChatMessage:
    """一条对话消息。``thinking`` 与 ``content`` 分离存储。"""

    role: str
    content: str
    thinking: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> ChatMessage:
        return cls(
            role=str(raw.get("role", "user")),
            content=str(raw.get("content", "")),
            thinking=str(raw.get("thinking", "") or ""),
        )

    def to_dict(self) -> dict:
        out = {"role": self.role, "content": self.content}
        if self.thinking:
            out["thinking"] = self.thinking
        return out


def _role_label(role: str) -> str:
    """映射为 RWKV 模板的角色标签。非 system/user 一律视为 assistant。"""
    if role == "system":
        return "System"
    if role == "user":
        return "User"
    return "Assistant"


def build_prompt(
    messages: list[ChatMessage],
    *,
    params: InferParams,
    thinking_mode: bool = False,
) -> str:
    """构建完整 prompt。

    :param messages: 完整对话历史（可含开头的 system 角色卡）
    :param params: 决定 presystem / max_rounds
    :param thinking_mode: True 时末尾追加 ``<think>`` 触发推理输出
    """
    system_content: str | None = None
    body = list(messages)

    # 1. 提取 system 角色卡（始终保留，不参与裁剪）
    if body and body[0].role == "system":
        system_content = body[0].content
        body = body[1:]

    # 2. 滑动窗口（仅在显式配置时启用）
    if params.max_rounds > 0:
        max_messages = params.max_rounds * 2 + 1  # N 轮 + 当前用户消息
        if len(body) > max_messages:
            body = body[-max_messages:]

    parts: list[str] = []

    # 3. 示范对话前缀
    if params.presystem:
        parts.append(PRESYSTEM_PREFIX)

    # 4. 角色卡
    if system_content:
        parts.append(f"System: {system_content}\n\n")

    # 5. 对话历史
    for msg in body:
        label = _role_label(msg.role)
        content = msg.content

        # ★ 核心：还原 thinking 标记
        if label == "Assistant" and msg.thinking:
            content = f"{THINK_START}\n{msg.thinking}\n{THINK_END}\n{content}"

        parts.append(f"{label}: {content}\n\n")

    # 6. 以 Assistant: 结尾触发生成（注意尾随空格，对齐 Runner）
    parts.append("Assistant: ")
    if thinking_mode:
        parts.append(THINK_START)

    return "".join(parts)


def split_thinking(raw: str) -> tuple[str, str]:
    """把模型原始输出切分为 (thinking, answer)。

    模型可能只输出 thinking 就被截断——此时 answer 为空字符串，
    不能当成错误。
    """
    if THINK_END in raw:
        thinking, _, answer = raw.partition(THINK_END)
        return thinking.strip(), answer.strip()
    if THINK_START in raw:
        _, _, thinking = raw.partition(THINK_START)
        return thinking.strip(), ""
    return "", raw.strip()


def count_rounds(messages: list[ChatMessage]) -> int:
    """统计有效对话轮数（user 消息数，排除 system）。"""
    return sum(1 for m in messages if m.role == "user")
