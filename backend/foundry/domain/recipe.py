"""训练配方（Recipe）—— Foundry 的核心概念。

配方是一次训练的可复现描述。**核心资产是配方，不是熔炉。**
同样的配方训练两次，应得到同一个 ``hash``，从而能直接横向对比 loss 曲线。

## 什么进入哈希，什么不进入

分界线是：**是否会改变产出的权重**。

    进入哈希：
        base_model, model_size, dataset, precision, my_testing,
        ctx_len, micro_bsz, epoch_steps, epoch_count,
        lr_init, lr_final, lora_r, lora_alpha, lora_dropout, epoch_save

    不进入哈希：
        save_folder    —— 输出位置，同一配方可铸到任意目录
        accelerator / strategy / devices / grad_cp
                       —— 影响速度与显存，不改变数学结果

如果把 save_folder 算进哈希，同一配方换个目录就是另一个配方，
对比实验立刻失去意义。这是本模块最重要的一条判据。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from foundry.errors import FoundryError

#: RWKV-7 "Goose" G1x 系列的固定架构参数。
#:
#: 这些数值必须与基底权重严格匹配——写错会直接导致加载失败或训练产出胡言乱语。
#: 它们来自 RWKV-PEFT / RWKV-LM 的官方模型定义，不可自行"优化"。
MODEL_SIZE_MAP: dict[str, tuple[int, int]] = {
    "0.1B": (12, 768),
    "0.4B": (24, 1024),
    "1.5B": (24, 2048),
    "2.9B": (32, 2560),
    "7B": (32, 4096),
    "13.3B": (61, 4096),
}

#: 兼容旧项目的 ``14B`` 别名（实为 13.3B 的 61 层配置）
MODEL_SIZE_ALIASES: dict[str, str] = {
    "14B": "13.3B",
}

#: 上游 RWKV-PEFT 的 CUDA 内核拒绝 fp16，只能 bf16 或 fp32。
VALID_PRECISIONS = ("bf16", "fp32")
VALID_MY_TESTING = ("x050", "x060", "x070")


class RecipeError(FoundryError):
    """配方参数非法。``problems`` 一次列出全部问题。"""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("\n".join(problems))


def normalize_model_size(model_size: str) -> str:
    """把别名折叠为规范名，未知尺寸直接报错（不要静默回退到默认值）。"""
    canonical = MODEL_SIZE_ALIASES.get(model_size, model_size)
    if canonical not in MODEL_SIZE_MAP:
        known = ", ".join(sorted(MODEL_SIZE_MAP))
        raise RecipeError([f"未知的 model_size: {model_size!r}。可用值: {known}"])
    return canonical


@dataclass(frozen=True)
class Recipe:
    """一次训练的可复现描述。

    ``dataset`` 用 ``"<folder>/<filename>"`` 形式表示（folder 可为空）。
    数据集内容不同即配方不同，因此它参与哈希。
    """

    base_model: str
    dataset: str
    model_size: str = "2.9B"

    # 序列与批量
    ctx_len: int = 512
    micro_bsz: int = 1
    epoch_steps: int = 1000
    epoch_count: int = 1
    epoch_save: int = 1

    # 学习率
    lr_init: float = 2e-5
    lr_final: float = 2e-5

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.01

    # 引擎
    precision: str = "bf16"
    my_testing: str = "x070"

    # ── 派生属性 ────────────────────────────────────────────────────────
    @property
    def n_layer(self) -> int:
        return MODEL_SIZE_MAP[normalize_model_size(self.model_size)][0]

    @property
    def n_embd(self) -> int:
        return MODEL_SIZE_MAP[normalize_model_size(self.model_size)][1]

    @property
    def peft_config(self) -> dict[str, Any]:
        """传给 ``--peft_config`` 的 JSON 对象。"""
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
        }

    @property
    def peft_config_json(self) -> str:
        """紧凑 JSON，键序固定，便于比对。"""
        return json.dumps(self.peft_config, separators=(",", ":"), sort_keys=True)

    # ── 哈希 ────────────────────────────────────────────────────────────
    def canonical_payload(self) -> dict[str, Any]:
        """参与哈希的字段，已按类型归一。

        类型归一很重要：``1`` 与 ``1.0``、``2e-5`` 与 ``0.00002`` 应产生同一哈希，
        否则用户改一下输入框就可能得到"新配方"。
        """
        return {
            "base_model": str(self.base_model),
            "dataset": str(self.dataset),
            "model_size": normalize_model_size(self.model_size),
            "ctx_len": int(self.ctx_len),
            "micro_bsz": int(self.micro_bsz),
            "epoch_steps": int(self.epoch_steps),
            "epoch_count": int(self.epoch_count),
            "epoch_save": int(self.epoch_save),
            "lr_init": float(self.lr_init),
            "lr_final": float(self.lr_final),
            "lora_r": int(self.lora_r),
            "lora_alpha": int(self.lora_alpha),
            "lora_dropout": float(self.lora_dropout),
            "precision": str(self.precision),
            "my_testing": str(self.my_testing),
        }

    @property
    def hash(self) -> str:
        """配方指纹：canonical JSON 的 SHA-256 前 16 位。"""
        blob = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # ── 校验 ────────────────────────────────────────────────────────────
    def problems(self) -> list[str]:
        """返回全部参数问题，而非只报第一个。"""
        found: list[str] = []

        try:
            normalize_model_size(self.model_size)
        except RecipeError as exc:
            found.extend(exc.problems)

        if self.precision not in VALID_PRECISIONS:
            found.append(
                f"precision={self.precision!r} 无效。"
                f"可选: {', '.join(VALID_PRECISIONS)}（上游 CUDA 内核拒绝 fp16）"
            )
        if self.my_testing not in VALID_MY_TESTING:
            found.append(
                f"my_testing={self.my_testing!r} 无效。可选: {', '.join(VALID_MY_TESTING)}"
            )

        if self.ctx_len <= 0:
            found.append(f"ctx_len 必须为正整数，当前 {self.ctx_len}")
        if self.micro_bsz <= 0:
            found.append(f"micro_bsz 必须为正整数，当前 {self.micro_bsz}")
        if self.epoch_steps <= 0:
            found.append(f"epoch_steps 必须为正整数，当前 {self.epoch_steps}")
        if self.epoch_count <= 0:
            found.append(f"epoch_count 必须为正整数，当前 {self.epoch_count}")
        if self.epoch_save <= 0:
            found.append(f"epoch_save 必须为正整数，当前 {self.epoch_save}")
        if self.epoch_save > self.epoch_count:
            found.append(
                f"epoch_save={self.epoch_save} 大于 epoch_count={self.epoch_count}，"
                f"训练结束前不会保存任何中间权重（最终权重仍会保存）"
            )

        if self.lr_init <= 0:
            found.append(f"lr_init 必须为正数，当前 {self.lr_init}")
        if self.lr_final < 0:
            found.append(f"lr_final 不能为负，当前 {self.lr_final}")

        if self.lora_r <= 0:
            found.append(f"lora_r 必须为正整数，当前 {self.lora_r}")
        if self.lora_alpha <= 0:
            found.append(f"lora_alpha 必须为正整数，当前 {self.lora_alpha}")
        if not 0.0 <= self.lora_dropout < 1.0:
            found.append(f"lora_dropout 必须落在 [0, 1)，当前 {self.lora_dropout}")

        if not self.base_model.strip():
            found.append("base_model 不能为空")
        if not self.dataset.strip():
            found.append("dataset 不能为空")

        return found

    def validate(self) -> Recipe:
        """校验通过则返回自身，否则抛 RecipeError。"""
        found = self.problems()
        if found:
            raise RecipeError(found)
        return self

    # ── 序列化 ──────────────────────────────────────────────────────────
    def to_payload(self) -> dict[str, Any]:
        """完整序列化（含不参与哈希的派生信息，便于人读）。"""
        payload = asdict(self)
        payload["_hash"] = self.hash
        payload["_n_layer"] = self.n_layer
        payload["_n_embd"] = self.n_embd
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Recipe:
        """从 ``to_payload`` 或数据库读出的 JSON 还原。

        忽略 ``_`` 开头的元字段，因此可以安全地把 to_payload 的结果喂回来。
        """
        fields = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in payload.items() if k in fields}
        return cls(**clean)

    def diff(self, other: Recipe) -> dict[str, tuple[Any, Any]]:
        """对比两个配方的差异，返回 ``{字段: (self, other)}``。

        用于「这个配方和上次那个差在哪」这类问题——直接比对参数表太笨。
        """
        a, b = self.canonical_payload(), other.canonical_payload()
        return {k: (a[k], b[k]) for k in a if a[k] != b[k]}
