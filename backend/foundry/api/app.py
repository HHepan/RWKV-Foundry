"""应用工厂 —— 组装 FastAPI 应用。

## 设计要点

### 1. 无模块级全局状态

旧版在模块顶层放 ``train_state`` / ``current_save_folder`` / ``training_process``
等全局变量。本版全部收进 ``AppState``，挂在 ``app.state.foundry``，
通过 ``Depends`` 注入。测试可并行、可隔离。

### 2. 错误映射集中处理

领域层抛 ``FoundryError`` 及其子类，这里统一翻译成 HTTP 状态码。
路由里不写 ``try/except HTTPException``——旧版每条路由都要重复一遍。

### 3. CORS 与 vite 对齐

旧版把 CORS 写死为 ``http://localhost:5173``，却又让 vite ``--host 0.0.0.0``，
用局域网 IP 打开前端必然被拒。这里从 ``ServerConfig`` 读取，默认已含
127.0.0.1 变体。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from foundry.api import routes_chat, routes_data, routes_train
from foundry.api.deps import AppState, build_state
from foundry.api.schemas import ConfigResponse, HealthResponse
from foundry.config import VOCAB_SIZE, AppConfig, InferDefaults, TrainDefaults, __version__
from foundry.domain.recipe import MODEL_SIZE_MAP
from foundry.errors import (
    ConfigError,
    ConflictError,
    FoundryError,
    NotFoundError,
    PathTraversalError,
)
from foundry.paths import resolve_paths

logger = logging.getLogger("foundry")


def _error_handler(status: int):
    async def handler(_request: Request, exc: FoundryError) -> JSONResponse:
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    return handler


def create_app(
    *,
    config: AppConfig | None = None,
    with_model: bool = True,
    state: AppState | None = None,
) -> FastAPI:
    """创建应用。

    :param with_model: False 时使用 ``NullModelLoader``，可在无 GPU 环境跑
        完整 API 测试（除真实生成外一切可用）。
    :param state: 直接注入已构造的 AppState（测试用）。
    """
    cfg = config or AppConfig.from_env()

    # ★ 必须在任何 torch / rwkv import 之前设置 RWKV 环境变量。
    #
    # 这是全进程唯一的 strict=True 调用点：此刻 torch 一定还没被 import
    # （应用刚创建，未加载任何模型）。若这里就发现 torch 已加载，说明
    # 有模块在 import 期偷偷拉了 torch，属于必须立刻暴露的顺序错误。
    #
    # 之后再调用 bootstrap 都是幂等空操作，不会因 torch 已加载而报错
    # —— 这正是「切换模型」能正常工作的原因。
    from foundry.env import bootstrap

    bootstrap(strict=True)

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    application = FastAPI(
        title="RWKV-Foundry",
        version=__version__,
        description="RWKV 微调与对话测试产线",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.server.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 依赖注入：所有路由通过 get_state(request) 取
    application.state.foundry = state or build_state(
        paths=resolve_paths(), config=cfg, with_model=with_model
    )

    # 错误映射：集中一处，路由不重复
    application.add_exception_handler(NotFoundError, _error_handler(404))
    application.add_exception_handler(ConflictError, _error_handler(409))
    application.add_exception_handler(PathTraversalError, _error_handler(400))
    application.add_exception_handler(ConfigError, _error_handler(503))
    application.add_exception_handler(FoundryError, _error_handler(500))

    application.include_router(routes_data.router)
    application.include_router(routes_train.router)
    application.include_router(routes_chat.router)

    @application.get("/api/health", response_model=HealthResponse, tags=["meta"])
    def health(request: Request) -> dict:
        st: AppState = request.app.state.foundry
        active = st.active_job()
        return {
            "ok": True,
            "model_loaded": st.engine.loaded,
            "model_ref": st.engine.model_ref,
            "active_train": active.folder if active else None,
            "schema_version": st.db.schema_version(),
        }

    @application.get("/api/config", response_model=ConfigResponse, tags=["meta"])
    def get_config(request: Request) -> dict:
        st: AppState = request.app.state.foundry
        return {
            "version": __version__,
            "paths": st.paths.as_dict(),
            "infer_defaults": InferDefaults().as_dict(),
            "train_defaults": TrainDefaults().__dict__,
            "vocab_size": VOCAB_SIZE,
            "model_sizes": sorted(MODEL_SIZE_MAP),
        }

    @application.get("/", tags=["meta"])
    def root() -> dict:
        return {"name": "RWKV-Foundry", "version": __version__, "docs": "/docs"}

    logger.info("RWKV-Foundry %s 已装配，workspace=%s", __version__, cfg.server.port)
    return application


#: uvicorn 入口：``uvicorn foundry.api.app:app_factory --factory``
#:
#: 刻意**不在模块层实例化**。模块级 ``app = create_app()`` 会让任何
#: ``import foundry.api`` 都触发环境校验——包括导入测试模块时。那等于说
#: "没有 GPU、没 clone RWKV-PEFT 的机器连 import 都不行"，这违背了本项目
#: 「无外部依赖也能跑单元测试」的边界。
def app_factory() -> FastAPI:
    """工厂函数的字符串入口，供 uvicorn ``--factory`` 使用。"""
    return create_app()
