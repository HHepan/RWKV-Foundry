"""领域层 —— 纯逻辑，零 web / 零 IO 依赖。

这个包里的模块必须可以在没有 FastAPI、没有 torch 的环境下 import 并单测。
旧版 llm-finetune-webui 把业务逻辑全部焊死在 FastAPI 路由函数里
（3849 行后端，0 测试），根因就是缺了这一层。
"""

from __future__ import annotations

from foundry.domain.recipe import (
    MODEL_SIZE_ALIASES,
    MODEL_SIZE_MAP,
    VALID_MY_TESTING,
    VALID_PRECISIONS,
    Recipe,
    RecipeError,
    normalize_model_size,
)

__all__ = [
    "MODEL_SIZE_ALIASES",
    "MODEL_SIZE_MAP",
    "Recipe",
    "RecipeError",
    "VALID_MY_TESTING",
    "VALID_PRECISIONS",
    "normalize_model_size",
]
