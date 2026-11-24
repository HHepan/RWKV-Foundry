"""数据管理路由 —— 数据集、基底模型、角色卡。

**薄层原则**：这里只做参数解析、调服务、返回响应。任何业务逻辑都应在
``api/service.py`` 里。旧版把 537 行混在路由里，这是不可测试的根源。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from foundry.api import service
from foundry.api.deps import AppState, get_state
from foundry.api.schemas import (
    DatasetPageResponse,
    DatasetRowUpdate,
    DatasetStatsResponse,
    MergeRequest,
    RolePayload,
    RoleResponse,
)

router = APIRouter(prefix="/api/data", tags=["data"])


# ── 数据集 ───────────────────────────────────────────────────────────────
@router.get("/folders", response_model=list[str])
def get_folders(state: AppState = Depends(get_state)) -> list[str]:
    return service.list_dataset_folders(state)


@router.get("/files", response_model=list[str])
def get_files(folder: str = Query(default=""), state: AppState = Depends(get_state)) -> list[str]:
    return service.list_dataset_files(state, folder)


@router.get("/files/stats", response_model=DatasetStatsResponse)
def get_file_stats(
    filename: str = Query(...),
    folder: str = Query(default=""),
    state: AppState = Depends(get_state),
) -> dict:
    return service.dataset_stats(state, folder, filename)


@router.get("/files/{filename}", response_model=DatasetPageResponse)
def get_file_data(
    filename: str,
    folder: str = Query(default=""),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1, le=200),
    rounds_filter: str = Query(default="all", pattern="^(all|single|multi)$"),
    state: AppState = Depends(get_state),
) -> dict:
    return service.page_dataset(
        state, folder, filename, page=page, size=size, rounds_filter=rounds_filter
    )


@router.put("/files/{filename}/{row_id}")
def update_row(
    filename: str,
    row_id: int,
    payload: DatasetRowUpdate,
    folder: str = Query(default=""),
    state: AppState = Depends(get_state),
) -> dict:
    """更新一行。

    旧版直接 ``await request.body()`` 把裸文本写盘，无格式校验、无大小限制。
    这里改用显式 JSON body（``{"text": "..."}``）。
    """
    service.update_dataset_row(state, folder, filename, row_id, payload.text)
    return {"message": "更新成功"}


@router.delete("/files/{filename}/{row_id}")
def delete_row(
    filename: str,
    row_id: int,
    folder: str = Query(default=""),
    state: AppState = Depends(get_state),
) -> dict:
    service.delete_dataset_row(state, folder, filename, row_id)
    return {"message": "删除成功"}


@router.delete("/files/{filename}")
def delete_file(
    filename: str,
    folder: str = Query(default=""),
    state: AppState = Depends(get_state),
) -> dict:
    service.delete_dataset_file(state, folder, filename)
    return {"message": "文件已删除"}


@router.post("/merge")
def merge_files(req: MergeRequest, state: AppState = Depends(get_state)) -> dict:
    return service.merge_datasets(
        state,
        source_files=req.source_files,
        new_name=req.new_name,
        folder=req.folder,
        shuffle=req.shuffle,
        counts=req.counts,
    )


# ── 基底模型 ─────────────────────────────────────────────────────────────
@router.get("/base-models", response_model=list[str])
def get_base_models(state: AppState = Depends(get_state)) -> list[str]:
    return service.list_base_models(state)


@router.get("/base-model-files", response_model=list[str])
def get_base_model_files(state: AppState = Depends(get_state)) -> list[str]:
    """保留旧路径以免前端大改；与 ``/base-models`` 同义。"""
    return service.list_base_models(state)


# ── checkpoint ───────────────────────────────────────────────────────────
@router.get("/checkpoint-folders", response_model=list[str])
def get_checkpoint_folders(state: AppState = Depends(get_state)) -> list[str]:
    return service.list_checkpoint_folders(state)


@router.get("/checkpoint-files", response_model=list[str])
def get_checkpoint_files(
    folder: str = Query(default=""), state: AppState = Depends(get_state)
) -> list[str]:
    return service.list_checkpoint_files(state, folder)


# ── 角色卡 ───────────────────────────────────────────────────────────────
@router.get("/roles", response_model=list[RoleResponse])
def get_roles(state: AppState = Depends(get_state)) -> list[dict]:
    return service.list_roles(state)


@router.post("/roles")
def save_role(req: RolePayload, state: AppState = Depends(get_state)) -> dict:
    return service.save_role(state, role_id=req.id, name=req.name, content=req.content)


@router.delete("/roles/{role_id}")
def delete_role(role_id: str, state: AppState = Depends(get_state)) -> dict:
    service.delete_role(state, role_id)
    return {"message": "角色已删除"}


# ── 对话数据 ─────────────────────────────────────────────────────────────
@router.get("/chat-models")
def get_chat_models(state: AppState = Depends(get_state)) -> list[dict]:
    return service.list_chat_models(state)


@router.get("/chat-data")
def get_chat_data(
    model: str = Query(...),
    session: str = Query(default=""),
    state: AppState = Depends(get_state),
) -> dict:
    return service.get_chat_data(state, model, session)


@router.delete("/chat-data")
def delete_chat_data(
    model: str = Query(...),
    session: str = Query(default=""),
    state: AppState = Depends(get_state),
) -> dict:
    service.delete_chat_data(state, model, session)
    return {"message": "删除成功"}
