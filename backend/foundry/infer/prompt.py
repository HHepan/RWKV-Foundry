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


# ── 续写模式 ────────────────────────────────────────────────────────────
#
# 续写与对话是两种不同的用法，prompt 结构完全不同：
#
#   对话:   User: xxx  Assistant: yyy  User: zzz  Assistant: <生成>
#   续写:   <原文文本……>                          <模型直接接着写>
#
# 续写**不加任何 User/Assistant 模板**，也不加 presystem 前缀——那些是
# 对话格式的一部分，塞进续写会污染文风，模型会开始模仿"助手"腔调，
# 而不是延续原文的叙事风格。
#
# 这也是 RWKV 作为 base model 的天然优势：它本身就是文档续写器，
# 不需要包装成问答格式。

#: 续写场景预设。prompt_prefix 会拼在用户输入之前，用于设定体裁与基调。
#:
#: 这些不是 RWKV-Runner 的原样拷贝——Runner 用的是预设文件系统，
#: 结构复杂。这里提炼出它的核心思路：给模型一个体裁锚点，
#: 让续写方向可控，而不是让它自由发散。
COMPLETION_PRESETS: dict[str, dict[str, str]] = {
    "novel": {
        "label": "小说续写",
        "description": "延续小说的叙事、人物与文风",
        "prompt_prefix": "",
        "hint": "把小说已有段落贴进来，模型会接着往下写。",
    },
    "scene": {
        "label": "情景续写",
        "description": "围绕一个场景继续展开描写",
        "prompt_prefix": "",
        "hint": "贴入场景描写，模型会补充环境、动作与氛围。",
    },
    "story": {
        "label": "操作故事续写",
        "description": "按已有情节推进故事发展",
        "prompt_prefix": "",
        "hint": "贴入故事开头或当前段落，模型会推进情节。",
    },
}


def build_completion_prompt(
    text: str,
    *,
    preset: str = "novel",
    system_content: str | None = None,
) -> str:
    """构建续写 prompt。

    与 ``build_prompt`` 的区别：

      - **不加** ``User:`` / ``Assistant:`` 模板
      - **不加** ``presystem`` 示范对话前缀（那是对话专用）
      - 原文原样作为 prompt，模型在其后直接续写
      - 可选的角色卡（system）以 ``System:`` 段置于最前

    ## 为什么角色卡仍用 System 段

    续写时若设定「你是古龙风格的武侠小说家」，需要让模型知道这个约束。
    用 ``System:`` 段而不是散文式说明，是因为 RWKV 训练数据里
    ``System:`` 是明确的分隔标记，模型认得这个结构。

    :param text: 用户提供的待续写原文
    :param preset: 预设场景 key（当前不影响输出，保留用于未来扩展）
    :param system_content: 可选的风格/角色设定
    """
    parts: list[str] = []

    if system_content:
        parts.append(f"System: {system_content}\n\n")

    # 原文原样保留：续写的全部上下文就在这段文本里
    parts.append(text)

    # 结尾补一个换行，避免模型把最后一行与续写内容粘在一起。
    # 若原文本身以换行结尾则不重复添加。
    if not parts[-1].endswith("\n"):
        parts.append("\n")

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
