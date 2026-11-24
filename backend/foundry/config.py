"""应用配置 —— 服务端口、CORS、推理默认值、训练默认值。

配置优先级（后者覆盖前者）：
    1. 代码内默认值（本文件）
    2. 环境变量
    3. 调用方显式传入

所有环境变量均以 ``FOUNDRY_`` 为前缀，避免与宿主环境冲突。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# ── 版本 ────────────────────────────────────────────────────────────────
__version__ = "0.1.0"

#: RWKV 词表大小。RWKV-7 World 词表固定为 65536，训练命令需要显式传入。
VOCAB_SIZE = 65536

#: thinking 标记。拆开拼接是为了避免本文件自身被模型训练数据扫描时误匹配。
THINK_START = "<" + "think" + ">"
THINK_END = "<" + "/think" + ">"


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是整数，当前为 {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServerConfig:
    """HTTP 服务配置。

    旧版把 CORS 写死为 ``http://localhost:5173``，同时又让 vite ``--host 0.0.0.0``，
    两者矛盾：用局域网 IP 打开前端时所有请求都会被 CORS 拒绝。这里把
    允许来源显式化，默认值与 vite 默认端口保持一致，并支持逗号分隔多项。
    """

    host: str = "127.0.0.1"
    port: int = 8000
    #: 逗号分隔。设为 ``*`` 表示允许全部（仅建议在本机调试时使用）。
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")

    @classmethod
    def from_env(cls) -> ServerConfig:
        raw_origins = _env_str("FOUNDRY_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
        origins = tuple(o.strip() for o in raw_origins.split(",") if o.strip())
        return cls(
            host=_env_str("FOUNDRY_HOST", "127.0.0.1"),
            port=_env_int("FOUNDRY_PORT", 8000),
            cors_origins=origins or ("http://localhost:5173",),
        )


@dataclass(frozen=True)
class InferDefaults:
    """推理默认值 —— 对齐 RWKV-Runner 的 TextRWKV。

    这些数值不是随手填的：temperature=1.0 / top_p=0.3 / top_k=0 是 RWKV-Runner
    的默认组合；alpha_presence=0 表示不惩罚首次出现的 token，alpha_frequency=1
    表示只对高频重复重罚，用词更自然。
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
    #: 是否注入示范对话前缀（对齐 Runner 的 __preload）。
    presystem: bool = True
    #: 滑动窗口保留轮数。0 = 不裁剪。
    #:
    #: 注意：RWKV 是循环模型，state 已累积完整历史。裁剪早期对话会使 prompt
    #: 开头每轮都变化，导致增量 prefill 完全失效（只能复用极少 token），
    #: 因此默认关闭。
    max_rounds: int = 0
    #: 每 N 轮触发一次显存回收。
    clean_rounds: int = 10

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "alpha_frequency": self.alpha_frequency,
            "alpha_presence": self.alpha_presence,
            "alpha_decay": self.alpha_decay,
            "newline_adj": self.newline_adj,
            "global_penalty": self.global_penalty,
            "presystem": self.presystem,
            "max_rounds": self.max_rounds,
            "clean_rounds": self.clean_rounds,
        }


@dataclass(frozen=True)
class TrainDefaults:
    """训练默认值。

    ``my_testing`` 是 RWKV 架构版本标识，必须与基底权重匹配（x070 = RWKV-7）。
    ``precision`` 上游 CUDA 内核拒绝 fp16，只能 bf16。
    """

    model_size: str = "2.9B"
    micro_bsz: int = 1
    epoch_save: int = 1
    epoch_steps: int = 1000
    ctx_len: int = 512
    epoch_count: int = 1
    lr_init: float = 2e-5
    lr_final: float = 2e-5
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.01
    precision: str = "bf16"
    accelerator: str = "gpu"
    strategy: str = "deepspeed_stage_1"
    devices: int = 1
    grad_cp: int = 1
    my_testing: str = "x070"
    #: binidx 转换后追加 eod。
    append_eod: bool = True
    #: 单次训练的采样上限，0 = 不限。
    data_sample_ratio: int = 0


@dataclass(frozen=True)
class InferRuntime:
    """推理运行期配置（非采样参数）。"""

    #: 设备策略，格式对齐 rwkv_x070_dual._parse_strategy。
    #:   单卡:  "cuda:0 fp16"
    #:   双卡:  "cuda:0 fp16 -> cuda:1 fp16"
    #:   CPU:   "cpu fp32"
    strategy: str = "cuda:0 fp16"
    #: 单模型状态下最多保留多少个会话缓存。超出按 LRU 淘汰。
    max_sessions: int = 32
    #: 生成队列上界。一张卡无法真正并行，超过即拒绝而不是排队堆积。
    max_queue: int = 4

    @classmethod
    def from_env(cls) -> InferRuntime:
        return cls(
            strategy=_env_str("FOUNDRY_INFER_STRATEGY", "cuda:0 fp16"),
            max_sessions=_env_int("FOUNDRY_MAX_SESSIONS", 32),
            max_queue=_env_int("FOUNDRY_MAX_QUEUE", 4),
        )


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    #: 是否把每个请求的完整 prompt 打到日志。
    #:
    #: 默认关闭 —— 旧项目无条件 print 完整 prompt，多轮对话后单次上万字符，
    #: 纯粹浪费 IO 且泄露用户内容。
    dump_prompt: bool = False

    @classmethod
    def from_env(cls) -> LoggingConfig:
        return cls(
            level=_env_str("FOUNDRY_LOG_LEVEL", "INFO").upper(),
            dump_prompt=_env_bool("FOUNDRY_DUMP_PROMPT", False),
        )


@dataclass(frozen=True)
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    infer: InferRuntime = field(default_factory=InferRuntime)
    infer_defaults: InferDefaults = field(default_factory=InferDefaults)
    train_defaults: TrainDefaults = field(default_factory=TrainDefaults)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            server=ServerConfig.from_env(),
            infer=InferRuntime.from_env(),
            logging=LoggingConfig.from_env(),
        )
