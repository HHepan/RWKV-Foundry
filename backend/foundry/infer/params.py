"""推理参数 —— 每个会话独立持有。

## 为什么参数必须 per-session

旧版 ``RWKVInferenceManager`` 是单例，且把 ``self.current_params`` 作为**共享可变
状态**。后果:

- 会话 A 把 temperature 调到 1.2，会话 B 的下一次生成就用 1.2
- ``generate()`` 内部还有一句 ``self.current_params = self.load_params_from_json(...)``
  每轮重读盘并覆盖，两个会话互相踩
- 前端传来的 ``request.params`` 被完全忽略（真 bug：参数面板是装饰品）

本版把参数绑在 ``SessionState`` 上，并通过 ``generate(..., params=...)``
**显式传入**。引擎内部不保存任何"当前参数"——没有共享状态，就没有串味。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from foundry.config import InferDefaults


@dataclass(frozen=True)
class InferParams:
    """一次生成使用的全部采样参数。

    frozen 是刻意的：修改参数必须用 ``with_overrides()`` 产生新对象。
    这样"某个请求偷偷改了全局参数"在类型层面就不可能发生。
    """

    max_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.3
    top_k: int = 0
    alpha_frequency: float = 1.0
    alpha_presence: float = 0.0
    alpha_decay: float = 0.996
    newline_adj: int = 0
    global_penalty: bool = False
    #: 是否注入示范对话前缀（对齐 RWKV-Runner 的 __preload）
    presystem: bool = True
    #: 滑动窗口轮数。0 = 不裁剪（RWKV 循环模型应保留全部历史）
    max_rounds: int = 0
    #: 每 N 轮回收一次显存
    clean_rounds: int = 10

    @classmethod
    def default(cls) -> InferParams:
        d = InferDefaults()
        return cls(
            max_tokens=d.max_tokens,
            temperature=d.temperature,
            top_p=d.top_p,
            top_k=d.top_k,
            alpha_frequency=d.alpha_frequency,
            alpha_presence=d.alpha_presence,
            alpha_decay=d.alpha_decay,
            newline_adj=d.newline_adj,
            global_penalty=d.global_penalty,
            presystem=d.presystem,
            max_rounds=d.max_rounds,
            clean_rounds=d.clean_rounds,
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> InferParams:
        """从请求体/数据库 JSON 构造，未知键忽略，缺失键用默认值。"""
        base = cls.default()
        if not raw:
            return base
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in raw.items() if k in known}
        if not clean:
            return base
        return replace(base, **clean)

    def with_overrides(self, **kwargs: Any) -> InferParams:
        """返回应用了覆盖的新对象；值为 None 或未知键的覆盖被忽略。"""
        known = {f for f in self.__dataclass_fields__}
        applied = {k: v for k, v in kwargs.items() if k in known and v is not None}
        return replace(self, **applied) if applied else self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ── 校验 ────────────────────────────────────────────────────────────
    def problems(self) -> list[str]:
        found: list[str] = []
        if self.max_tokens <= 0:
            found.append(f"max_tokens 必须为正整数，当前 {self.max_tokens}")
        if self.max_tokens > 32768:
            found.append(
                f"max_tokens={self.max_tokens} 过大。RWKV 是循环模型，"
                f"虽然显存恒定但生成时间线性增长，建议 <= 8192"
            )
        if self.temperature < 0:
            found.append(f"temperature 不能为负，当前 {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            found.append(f"top_p 必须落在 (0, 1]，当前 {self.top_p}")
        if self.top_k < 0:
            found.append(f"top_k 不能为负，当前 {self.top_k}")
        if not 0.0 <= self.alpha_decay <= 1.0:
            found.append(f"alpha_decay 必须落在 [0, 1]，当前 {self.alpha_decay}")
        if self.alpha_frequency < 0:
            found.append(f"alpha_frequency 不能为负，当前 {self.alpha_frequency}")
        if self.alpha_presence < 0:
            found.append(f"alpha_presence 不能为负，当前 {self.alpha_presence}")
        if self.max_rounds < 0:
            found.append(f"max_rounds 不能为负，当前 {self.max_rounds}")
        if self.clean_rounds <= 0:
            found.append(f"clean_rounds 必须为正整数，当前 {self.clean_rounds}")
        return found

    def validate(self) -> InferParams:
        found = self.problems()
        if found:
            from foundry.errors import FoundryError

            raise FoundryError("推理参数无效:\n" + "\n".join(f"  - {p}" for p in found))
        return self
