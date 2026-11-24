"""RWKV 运行环境引导。

**必须在本进程 import torch 之前调用 ``bootstrap()``。**

为什么需要这个模块：RWKV 通过三个环境变量控制内核编译路径，而这些变量只在
首次 import 相关模块时读取一次。

    RWKV_V7_ON   = 1    启用 RWKV-7 算子
    RWKV_JIT_ON  = 1    启用 TorchScript JIT
    RWKV_CUDA_ON = 1    启用 CUDA 加速内核

旧版 llm-finetune-webui 在 ``rwkv_inference.py`` 顶层直接 ``os.environ[...] = '1'``，
依赖「这个模块先被 import」这一隐式顺序。一旦别处先 import 了 torch/rwkv，
设置就已失效且**不会报错**，只会静默降级到慢路径。

这里改为显式函数调用，并在 torch 已加载时给出明确警告，让这类静默故障可见。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RWKVEnv:
    """三个内核开关的期望状态。"""

    v7: bool = True
    jit: bool = True
    cuda: bool = True

    def as_environ(self) -> dict[str, str]:
        return {
            "RWKV_V7_ON": "1" if self.v7 else "0",
            "RWKV_JIT_ON": "1" if self.jit else "0",
            "RWKV_CUDA_ON": "1" if self.cuda else "0",
        }


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def desired_env() -> RWKVEnv:
    """从环境变量读取期望配置，便于无卡机器上关掉 CUDA 内核。"""
    return RWKVEnv(
        v7=_env_flag("FOUNDRY_RWKV_V7", True),
        jit=_env_flag("FOUNDRY_RWKV_JIT", True),
        cuda=_env_flag("FOUNDRY_RWKV_CUDA", True),
    )


def torch_already_loaded() -> bool:
    return "torch" in sys.modules


def bootstrap(*, strict: bool = False) -> RWKVEnv:
    """设置 RWKV 环境变量。

    :param strict: 为 True 时，若 torch 已在 ``sys.modules`` 中则抛错。
                   服务启动应使用 strict=True，脚本与测试可用默认值。
    :returns: 实际生效的配置。

    调用方责任：必须在 import torch 之前调用。本函数无法回滚已加载的 torch，
    只能在检测到顺序错误时报警。
    """
    env = desired_env()
    settings = env.as_environ()

    late = torch_already_loaded()
    if late:
        msg = (
            "RWKV 环境变量设置过晚：torch 已被 import，"
            "RWKV_V7_ON/RWKV_JIT_ON/RWKV_CUDA_ON 可能不会生效。\n"
            "  修复: 在进程入口最顶部（任何 torch/rwkv import 之前）调用 "
            "foundry.env.bootstrap()。"
        )
        if strict:
            raise RuntimeError(msg)
        print(f"[foundry.env] WARNING: {msg}", file=sys.stderr)

    for key, value in settings.items():
        os.environ[key] = value

    return env


def apply_rwkv_paths(tokenizer_tools_dir: str | os.PathLike[str]) -> None:
    """把 RWKV-PEFT 的 tokenizer 工具目录加入 ``sys.path``。

    上游的 ``preprocess_data.py`` 与推理代码都依赖 ``from rwkv_tokenizer import ...``，
    而该模块位于 RWKV-PEFT 的 ``json2binidx_tool/tools/`` 下。旧项目把绝对路径
    直接写死在模块顶层，换机器即失效。
    """
    path = os.fspath(tokenizer_tools_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
