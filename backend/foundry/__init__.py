"""RWKV-Foundry 后端核心包。

模块职责边界：

    errors   异常类型（能明确报错就绝不静默失败）
    paths    文件系统布局，全项目唯一路径真相
    env      RWKV 环境变量引导（必须在 import torch 前调用）
    config   运行配置（端口/CORS/推理/训练默认值）

依赖方向严格单向，本包不依赖 FastAPI，因此可在无 web 环境下单测。
"""

from __future__ import annotations

from foundry.config import (
    THINK_END,
    THINK_START,
    VOCAB_SIZE,
    AppConfig,
    InferDefaults,
    InferRuntime,
    LoggingConfig,
    ServerConfig,
    TrainDefaults,
    __version__,
)
from foundry.errors import (
    ConfigError,
    FoundryError,
    PathTraversalError,
    PeftNotReadyError,
)
from foundry.paths import (
    ENV_HOME,
    ENV_PEFT_DIR,
    PEFT_REQUIRED_FILES,
    WorkspacePaths,
    get_paths,
    repo_root,
    resolve_paths,
    safe_join,
    safe_name,
)

__all__ = [
    # config
    "AppConfig",
    "InferDefaults",
    "InferRuntime",
    "LoggingConfig",
    "ServerConfig",
    "TrainDefaults",
    "THINK_END",
    "THINK_START",
    "VOCAB_SIZE",
    "__version__",
    # errors
    "ConfigError",
    "FoundryError",
    "PathTraversalError",
    "PeftNotReadyError",
    # paths
    "ENV_HOME",
    "ENV_PEFT_DIR",
    "PEFT_REQUIRED_FILES",
    "WorkspacePaths",
    "get_paths",
    "repo_root",
    "resolve_paths",
    "safe_join",
    "safe_name",
]
