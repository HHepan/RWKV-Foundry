"""采样参数适配与 stop 序列检测。

## 两件事

### 1. 把我们的 ``InferParams`` 转成上游 ``PIPELINE_ARGS``

上游 ``PIPELINE_ARGS`` 定义在 vendored 的 ``rwkv_utils_patch`` 里（不是官方
``rwkv`` 包的版本——官方版本不返回 state，无法做增量 prefill）。本模块
做这一层转换，使 ``session.py`` 不直接依赖 vendored 模块的内部结构。

### 2. stop 序列检测

模型不知道何时该停——它只是在预测下一个 token。必须在**文本层面**检测
对话模板泄漏，例如模型开始模仿格式输出 ``\\n\\nUser:`` 时就要截断。

关键难点：**跨 chunk 边界**。增量输出的一个 chunk 可能只含 ``"\\n\\nUs"``，
真正的 ``"er:"`` 下一个 chunk 才到。因此必须保留一个尾窗缓冲，
长度等于最长 stop 序列。

旧版用固定 12 字符尾窗 + 字符级拼接，这里保留同样的思路但做成可复用类。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from foundry.infer.params import InferParams

#: 对话模板泄漏的截断标记。
#:
#: 模型偶尔会"替用户说话"或重复模板，遇到这些就停。顺序无关——只要
#: 任意一个被检测到即截断。``"\nAss"`` 等短前缀是给模型打字过程中的
#: 早期阻断（防止先输出半截再被截断导致内容不完整）。
STOP_SEQUENCES: tuple[str, ...] = (
    "\n\nUser:",
    "\nAssistant:",
    "\n\nAssistant:",
    "\n\n\nAssistant:",
    "\nAss",
    "\n\nAss",
    "\n\n\nAss",
)

#: 尾窗长度 = 最长 stop 序列长度 - 1。保证任何跨边界的 stop 都能被拼出来。
_TAIL_WINDOW = max(len(s) for s in STOP_SEQUENCES) - 1

#: token id 0 = EOD，RWKV 词表的结束符
EOD_ID = 0


@dataclass
class PipelineArgs:
    """上游 ``PIPELINE_ARGS`` 的结构等价物。

    ## 为什么不直接 import vendored 的 PIPELINE_ARGS

    vendored 的 ``rwkv_utils_patch`` 在模块顶层 ``import torch``。若本模块
    直接 import 它，任何用到采样参数的单元测试都必须先装好 CUDA 版 torch
    ——这让"参数转换"这种纯逻辑无法独立验证。

    字段与上游**逐一对齐**（上游 ``pipeline.generate`` 只按属性名读取，
    不检查类型），因此本类可直接传给上游 ``generate(args=...)``。
    """

    temperature: float = 1.0
    top_p: float = 0.3
    top_k: int = 0
    alpha_frequency: float = 1.0
    alpha_presence: float = 0.0
    alpha_decay: float = 0.996
    token_ban: list[int] = field(default_factory=list)
    token_stop: list[int] = field(default_factory=lambda: [EOD_ID])
    #: 分块喂入长度，短则省显存但更慢
    chunk_len: int = 256
    newline_adj: int = 0
    global_penalty: bool = False


def make_pipeline_args(params: InferParams) -> PipelineArgs:
    """把 ``InferParams`` 转成上游可直接消费的参数对象。"""
    return PipelineArgs(
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        alpha_frequency=params.alpha_frequency,
        alpha_presence=params.alpha_presence,
        alpha_decay=params.alpha_decay,
        token_stop=[EOD_ID],
        newline_adj=params.newline_adj,
        global_penalty=params.global_penalty,
    )


@dataclass
class StopSequenceFilter:
    """跨 chunk 的 stop 序列检测。

    用法::

        f = StopSequenceFilter()
        for chunk in stream:
            safe = f.feed(chunk)      # 可安全输出的文本（不含 stop 序列）
            if safe:
                emit(safe)
            if f.stopped:
                break
        tail = f.flush()              # 流结束时的残留

    设计要点：``feed`` 返回的是**已确认安全**的部分。末尾可能是某个 stop
    序列的前缀，因此要暂存，等下一个 chunk 到了再判断。
    """

    sequences: tuple[str, ...] = STOP_SEQUENCES
    _buffer: str = field(default="", repr=False)
    _stopped: bool = field(default=False, repr=False)
    _emitted: list[str] = field(default_factory=list, repr=False)

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def emitted(self) -> str:
        return "".join(self._emitted)

    def feed(self, chunk: str) -> str:
        """喂入新文本，返回可安全输出的部分。检测到 stop 时置 ``stopped``。"""
        if self._stopped:
            return ""

        self._buffer += chunk

        # ── 检测完整 stop 序列 ──────────────────────────────────────────
        earliest = -1
        for seq in self.sequences:
            pos = self._buffer.find(seq)
            if pos != -1 and (earliest == -1 or pos < earliest):
                earliest = pos

        if earliest != -1:
            safe = self._buffer[:earliest]
            self._buffer = ""
            self._stopped = True
            if safe:
                self._emitted.append(safe)
            return safe

        # ── 没有完整匹配：只输出"绝不可能是 stop 前缀"的部分 ────────────
        keep = _tail_window(self._buffer, self.sequences)
        if keep <= 0:
            return ""

        safe = self._buffer[:keep]
        self._buffer = self._buffer[keep:]
        if safe:
            self._emitted.append(safe)
        return safe

    def flush(self) -> str:
        """流结束时吐出残留缓冲（若已 stop 则丢弃）。"""
        if self._stopped:
            self._buffer = ""
            return ""
        safe = self._buffer
        self._buffer = ""
        if safe:
            self._emitted.append(safe)
        return safe


def _tail_window(buffer: str, sequences: tuple[str, ...]) -> int:
    """计算可安全输出的长度（从头算起）。

    找出 buffer 末尾有多少字符可能是某个 stop 序列的前缀，这些必须保留。
    返回 ``len(buffer) - 需要的保留长度``。
    """
    max_keep = 0
    for seq in sequences:
        # 检查 seq 的每个前缀是否与 buffer 的尾部匹配
        for length in range(min(len(seq) - 1, len(buffer)), 0, -1):
            if buffer.endswith(seq[:length]):
                max_keep = max(max_keep, length)
                break
    return len(buffer) - max_keep
