"""HTTP 请求/响应模型。

## 一处刻意的设计：请求模型与领域模型分离

``InferParams`` / ``Recipe`` 是**领域对象**（frozen、有校验、无 IO）。
这里的 ``*Request`` 是**传输对象**（可部分填充、容忍未知字段）。

不直接复用领域对象做请求体，因为：

1. 前端可能只传部分字段，需要区分"未传"和"传了默认值"
2. 领域校验抛 ``FoundryError``，API 层需要转成 HTTP 状态码
3. 领域对象演化时不应牵动 HTTP 契约

反过来，**响应模型直接由领域对象派生**（``from_*`` 类方法），
保证前端拿到的字段与内部状态永远一致。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from foundry.domain.recipe import Recipe
from foundry.infer.params import InferParams


# ── 数据集 ───────────────────────────────────────────────────────────────
class DatasetPageResponse(BaseModel):
    total: int | None = Field(None, description="总行数；无索引时为 null")
    page: int
    size: int
    has_more: bool
    data: list[dict[str, Any]]


class DatasetStatsResponse(BaseModel):
    filename: str
    folder: str
    line_count: int
    rounds_single: int
    rounds_multi: int
    cached: bool = Field(description="是否命中索引缓存")


class DatasetRowUpdate(BaseModel):
    text: str


class MergeRequest(BaseModel):
    source_files: list[dict[str, str]] = Field(
        description='例如 [{"filename": "a.jsonl", "folder": "out"}]'
    )
    new_name: str
    folder: str = ""
    shuffle: bool = True
    counts: dict[str, int] = Field(
        default_factory=dict,
        description='按源文件限制采样数，键为 "folder/filename" 或 "filename"；空表示全取',
    )


# ── 配方 ─────────────────────────────────────────────────────────────────
class RecipePayload(BaseModel):
    """配方请求体。所有字段可选，未传则用默认值。"""

    base_model: str
    dataset: str
    model_size: str = "2.9B"
    ctx_len: int = 512
    micro_bsz: int = 1
    epoch_steps: int = 1000
    epoch_count: int = 1
    epoch_save: int = 1
    lr_init: float = 2e-5
    lr_final: float = 2e-5
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.01
    precision: str = "bf16"
    my_testing: str = "x070"

    def to_recipe(self) -> Recipe:
        return Recipe(
            base_model=self.base_model,
            dataset=self.dataset,
            model_size=self.model_size,
            ctx_len=self.ctx_len,
            micro_bsz=self.micro_bsz,
            epoch_steps=self.epoch_steps,
            epoch_count=self.epoch_count,
            epoch_save=self.epoch_save,
            lr_init=self.lr_init,
            lr_final=self.lr_final,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            precision=self.precision,
            my_testing=self.my_testing,
        )


class RecipeValidationResponse(BaseModel):
    hash: str
    valid: bool
    problems: list[str]
    n_layer: int | None = None
    n_embd: int | None = None
    peft_config: dict[str, Any] | None = None


class ScriptPreviewResponse(BaseModel):
    hash: str
    script: str


# ── 训练 ─────────────────────────────────────────────────────────────────
class TrainStartResponse(BaseModel):
    folder: str
    run_id: int
    recipe_hash: str
    status: str
    pid: int | None = None
    pgid: int | None = None


class TrainStatusResponse(BaseModel):
    folder: str | None = None
    status: str
    pid: int | None = None
    exit_code: int | None = None
    error: str | None = None
    elapsed_seconds: float | None = None
    recipe_hash: str | None = None


class TrainStopResponse(BaseModel):
    status: str
    note: str


class LossPoint(BaseModel):
    step: int
    epoch: int | None = None
    loss: float
    lr: float | None = None
    kt_s: float | None = None


class LogLine(BaseModel):
    seq: int
    line: str


class RunRecord(BaseModel):
    id: int
    folder: str
    base_model: str
    dataset: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None
    recipe_hash: str | None = None


# ── 会话与对话 ───────────────────────────────────────────────────────────
class SessionCreateRequest(BaseModel):
    model_ref: str = Field(description='例如 "20260414/rwkv-10" 或 "base_models/rwkv7-g1a-0.1b"')
    name: str = ""
    role_id: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    id: int
    model_ref: str
    name: str
    role_id: str | None
    params: dict[str, Any]
    created_at: str
    updated_at: str
    message_count: int = 0


class MessagePayload(BaseModel):
    role: str
    content: str
    thinking: str = ""


class ChatRequest(BaseModel):
    """对话请求。

    ``history`` **不含**本轮用户消息——由 ``message`` 追加。
    这与前端 regenerate 的语义一致：先移除上轮 assistant 回复，再重发。
    """

    model_ref: str
    message: str
    history: list[MessagePayload] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    thinking_mode: bool = False
    session_name: str = ""
    role_id: str | None = None


class CompletionRequest(BaseModel):
    """续写请求。

    ``text`` 是待续写的原文，**不是用户提问**。后端不会把它包成
    ``User: ...`` —— 续写直接以原文为 prompt，让模型接着写。
    """

    model_ref: str
    text: str
    preset: str = Field(
        default="novel",
        description="场景预设: novel(小说) / scene(情景) / story(故事)",
    )
    params: dict[str, Any] = Field(default_factory=dict)
    thinking_mode: bool = False
    session_name: str = ""
    role_id: str | None = None


class ChatResultResponse(BaseModel):
    content: str
    thinking: str
    stopped: bool
    tokens_fed: int
    tokens_reused: int


# ── 角色卡 ───────────────────────────────────────────────────────────────
class RolePayload(BaseModel):
    id: str | None = None
    name: str
    content: str


class RoleResponse(BaseModel):
    id: str
    name: str
    content: str
    updated_at: str


# ── 配置与健康 ───────────────────────────────────────────────────────────
class ConfigResponse(BaseModel):
    version: str
    paths: dict[str, str]
    infer_defaults: dict[str, Any]
    train_defaults: dict[str, Any]
    vocab_size: int
    model_sizes: list[str]


class HealthResponse(BaseModel):
    ok: bool
    model_loaded: bool
    model_ref: str | None
    active_train: str | None
    schema_version: int


def infer_params_from(raw: dict[str, Any]) -> InferParams:
    """统一入口：把请求里的 params 字典转成领域对象。"""
    return InferParams.from_dict(raw)
