"""HTTP 层。

    deps.py         依赖注入、AppState、ModelLoader
    schemas.py      请求/响应模型
    service.py      ★ 业务编排（路由调它，它调领域层）
    routes_data.py  数据集 / 基底模型 / 角色卡 / 对话数据
    routes_train.py 配方 / 启停 / 状态 / loss / 日志
    routes_chat.py  SSE 流式对话
    app.py          create_app 工厂

分层纪律：
    路由   → 只解析参数、调 service、返回响应
    service → 组合领域模块、翻译异常
    领域   → 纯逻辑，零 IO 依赖

**路由里不写 SQL、不拼 shell、不碰 subprocess。**
旧版 api/train.py 537 行把这三样全塞进路由，是不可测试的直接原因。
"""

from __future__ import annotations

from foundry.api.app import app_factory, create_app
from foundry.api.deps import (
    AppState,
    CudaModelLoader,
    ModelLoader,
    NullModelLoader,
    build_state,
    get_state,
)

__all__ = [
    "app_factory",
    "create_app",
    "AppState",
    "CudaModelLoader",
    "ModelLoader",
    "NullModelLoader",
    "build_state",
    "get_state",
]
