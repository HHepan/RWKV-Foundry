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


#: bootstrap 是否已经成功执行过（模块级一次性标记）
_bootstrapped: bool = False


def is_bootstrapped() -> bool:
    """环境变量是否已经在本进程生效过。"""
    return _bootstrapped


def bootstrap(*, strict: bool = False) -> RWKVEnv:
    """设置 RWKV 环境变量。**幂等，全进程只需成功一次。**

    :param strict: 为 True 时，若本函数**从未**成功执行且 torch 已被 import，
                   则抛错。服务启动应使用 strict=True 以便尽早暴露顺序错误。
    :returns: 期望的配置。

    ## 为什么必须幂等

    这里踩过一个线上 bug：CudaModelLoader 每次 load_model 都调
    ``bootstrap(strict=True)``。第一次加载模型（如 2.9B）时 torch 尚未 import，
    执行成功；**切换到第二个模型（如 13.3B）时 torch 已在 sys.modules 里**，
    strict 模式立即抛错：

        RWKV 环境变量设置过晚：torch 已被 import …

    用户看到的就是「换模型后对话报错」。

    环境变量本就是进程级一次性设置，第二次调用毫无意义。因此：
      - 首次成功执行后置 ``_bootstrapped = True``
      - 后续调用若已置位，直接返回，不再检查 torch 顺序
    这样「切模型」这类重复调用完全无害。

    ## 调用方责任

    真正需要保证的是「第一次调用发生在 import torch 之前」。
    服务入口（create_app）已负责这件事；loader 里的调用只是兜底。
    """
    global _bootstrapped

    env = desired_env()
    settings = env.as_environ()

    # 已成功执行过：环境变量已经在进程里生效，直接返回。
    # 此时 torch 必然已 import（模型都加载过了），不该再报警。
    if _bootstrapped:
        for key, value in settings.items():
            os.environ.setdefault(key, value)
        return env

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

    _bootstrapped = True
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
