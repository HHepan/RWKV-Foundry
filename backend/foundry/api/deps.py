"""API 层依赖注入 —— 应用状态的唯一边界。

## 为什么要有这一层

旧版把 ``current_save_folder`` / ``current_params`` / ``training_process`` 全部
写成**模块级全局变量**（``api/train.py:18-52``）。后果：

- 测试无法隔离（import 一次就有全局状态）
- 多进程部署直接错乱
- 无法回答"这个值是谁改的"

本版把全部可变状态收进 ``AppState``，由 FastAPI 的 ``app.state`` 持有，
通过 ``Depends`` 注入。没有模块级全局变量。

## 模型加载的可插拔

真正加载 RWKV 权重需要 torch + GPU。为了让 API 层能在无 GPU 环境完整测试，
模型加载抽象为 ``ModelLoader`` 协议：

    CudaModelLoader   真实实现，延迟 import torch
    NullModelLoader   明确拒绝，用于只跑 Web 的场景

引擎本身（``InferenceEngine``）不关心怎么加载，只接收已就绪的 pipeline。
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi import Request

from foundry.config import AppConfig
from foundry.errors import ConfigError, FoundryError
from foundry.infer.session import InferenceEngine
from foundry.paths import WorkspacePaths
from foundry.storage.db import Database, open_database
from foundry.train.job import TrainJob


class ModelLoader(Protocol):
    """模型加载策略。"""

    def load(self, model_path: str, *, strategy: str) -> tuple[Any, Any]:
        """加载并返回 ``(pipeline, tokenizer)``。"""
        ...

    def unload(self) -> None: ...


class NullModelLoader:
    """无 GPU 环境的占位实现。明确拒绝而不是静默失败。"""

    def load(self, model_path: str, *, strategy: str) -> tuple[Any, Any]:
        raise ConfigError(
            "当前环境未安装训练/推理依赖（torch），无法加载模型。\n"
            "  安装: pip install -e '.[train]'\n"
            "  或设置 FOUNDRY_NO_MODEL=0 并在有 GPU 的机器上运行。"
        )

    def unload(self) -> None:
        return None


class CudaModelLoader:
    """真实加载器。所有 torch/rwkv 导入都延迟到 ``load()`` 内。"""

    def __init__(self, paths: WorkspacePaths):
        self._paths = paths
        self._loaded: Any = None

    def load(self, model_path: str, *, strategy: str) -> tuple[Any, Any]:
        # RWKV 环境变量必须在 import torch 之前生效。
        #
        # bootstrap() 是幂等的：首次成功后即打标记，后续调用直接返回。
        # 因此这里用 **strict=False** —— 真正的顺序保证由 create_app 负责
        # （它在任何 torch import 之前调用一次 strict=True）。
        #
        # 早前这里写的是 strict=True，导致切换模型时炸：
        # 第一次加载 2.9B 时 torch 尚未 import；换到 13.3B 时 torch 已在
        # sys.modules 里，strict 立刻抛 "环境变量设置过晚"。用户看到的就是
        # 「换模型后对话报错」。
        from foundry.env import apply_rwkv_paths, bootstrap

        bootstrap(strict=False)
        apply_rwkv_paths(self._paths.tokenizer_tools)

        import torch

        from foundry.third_party.rwkv_utils_patch import PIPELINE
        from foundry.third_party.rwkv_x070_dual import RWKV_x070_Dual

        torch.cuda.empty_cache()

        # 词表必须用 RWKV-PEFT 自带的那份，保证与训练时一致。
        # 原版从 rwkv pip 包内部取，版本可能不同 → token id 错位且不报错。
        vocab = self._paths.vocab
        if not vocab.is_file():
            raise ConfigError(
                f"缺少词表文件: {vocab}\n"
                f"  它应随 RWKV-PEFT 一起克隆。请确认 workspace/tools/RWKV-PEFT 完整。"
            )

        model = RWKV_x070_Dual(model=model_path, strategy=strategy)
        pipeline = PIPELINE(model, "rwkv_vocab_v20230424", vocab_path=str(vocab))

        self._loaded = (model, pipeline)
        return pipeline, pipeline.tokenizer

    def unload(self) -> None:
        if self._loaded is None:
            return
        import gc

        import torch

        self._loaded = None
        gc.collect()
        torch.cuda.empty_cache()


@dataclass
class AppState:
    """进程内全部可变状态。

    刻意**不含**任何"当前会话"或"当前参数"——那些是每请求的局部数据。
    """

    config: AppConfig
    paths: WorkspacePaths
    db: Database
    engine: InferenceEngine
    loader: ModelLoader

    #: 训练任务注册表。key = run folder。
    #:
    #: 这是唯一允许存在"全局"的东西——因为一个进程只能跑一个训练。
    #: 用锁保护，且只做增删查，不承载业务逻辑。
    jobs: dict[str, TrainJob] = field(default_factory=dict)
    jobs_lock: threading.Lock = field(default_factory=threading.Lock)

    def active_job(self) -> TrainJob | None:
        """返回正在运行的训练任务（若有）。"""
        with self.jobs_lock:
            for job in self.jobs.values():
                if not job.status.terminal:
                    return job
            return None

    def register_job(self, job: TrainJob) -> None:
        with self.jobs_lock:
            self.jobs[job.folder] = job

    def drop_job(self, folder: str) -> None:
        with self.jobs_lock:
            self.jobs.pop(folder, None)


def build_state(*, paths: WorkspacePaths, config: AppConfig, with_model: bool = True) -> AppState:
    """构造应用状态并完成环境校验。

    校验失败会抛 ``ConfigError``，由调用方决定是退出还是降级——
    **绝不静默创建空目录**（旧版 ``file_service`` 的老毛病）。
    """
    paths.validate()
    paths.ensure_output_dirs()

    db = open_database(paths.db)
    engine = InferenceEngine(max_sessions=config.infer.max_sessions)
    loader: ModelLoader = CudaModelLoader(paths) if with_model else NullModelLoader()

    return AppState(config=config, paths=paths, db=db, engine=engine, loader=loader)


# ── FastAPI 依赖 ─────────────────────────────────────────────────────────
def get_state(request: Request) -> AppState:
    state: AppState | None = getattr(request.app.state, "foundry", None)
    if state is None:  # pragma: no cover - 装配错误
        raise FoundryError("应用状态未初始化，请检查 app_factory")
    return state


def get_paths_dep(request: Request) -> WorkspacePaths:
    return get_state(request).paths


def get_db_dep(request: Request) -> Database:
    return get_state(request).db


def get_engine_dep(request: Request) -> InferenceEngine:
    return get_state(request).engine


def get_config_dep(request: Request) -> AppConfig:
    return get_state(request).config


def iter_ingest(db: Database, paths: WorkspacePaths) -> Iterator[None]:
    """在读取训练数据前先灌一次增量。

    这是单写入者原则的落地：trainer 只写文件，DB 由 ingester 更新。
    读接口调用它，保证"刚训完就能看到曲线"。
    """
    from foundry.storage import ingest

    ingest.ingest_all(db, paths.checkpoints)
    yield None
