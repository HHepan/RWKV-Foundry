"""训练路由 —— 配方校验、脚本预览、启停、状态与曲线。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from foundry.api import service
from foundry.api.deps import AppState, get_state
from foundry.api.schemas import (
    LogLine,
    LossPoint,
    RecipePayload,
    RecipeValidationResponse,
    RunRecord,
    ScriptPreviewResponse,
    TrainStartResponse,
    TrainStatusResponse,
    TrainStopResponse,
)

router = APIRouter(prefix="/api/train", tags=["train"])


@router.post("/recipe/validate", response_model=RecipeValidationResponse)
def validate_recipe(req: RecipePayload) -> dict:
    """校验配方并返回其内容指纹。

    同一份参数永远得到同一个 hash —— 这是横向对比多次训练的前提。
    """
    return service.validate_recipe(req.model_dump())


@router.post("/recipe/preview", response_model=ScriptPreviewResponse)
def preview_script(
    req: RecipePayload,
    save_folder: str = Query(...),
    state: AppState = Depends(get_state),
) -> dict:
    """预览将要执行的训练脚本（不落盘、不启动）。"""
    return service.preview_script(state, req.model_dump(), save_folder)


@router.post("/start", response_model=TrainStartResponse)
def start_training(
    req: RecipePayload,
    save_folder: str = Query(..., description="checkpoints 下的目录名"),
    state: AppState = Depends(get_state),
) -> dict:
    return service.start_training(state, req.model_dump(), save_folder)


@router.post("/stop", response_model=TrainStopResponse)
def stop_training(state: AppState = Depends(get_state)) -> dict:
    return service.stop_training(state)


@router.get("/status", response_model=TrainStatusResponse)
def get_status(
    folder: str | None = Query(default=None), state: AppState = Depends(get_state)
) -> dict:
    return service.training_status(state, folder)


@router.get("/loss", response_model=list[LossPoint])
def get_loss(
    folder: str = Query(...),
    since_step: int = Query(default=0, ge=0),
    state: AppState = Depends(get_state),
) -> list[dict]:
    """loss 曲线。``since_step`` 支持增量拉取。"""
    return service.loss_series(state, folder, since_step=since_step)


@router.get("/loss-file/{folder_name}", response_model=list[LossPoint])
def get_loss_file(
    folder_name: str,
    since_step: int = Query(default=0, ge=0),
    state: AppState = Depends(get_state),
) -> list[dict]:
    """兼容旧路径。与 ``/loss`` 同源——不再有双份数据。"""
    return service.loss_series(state, folder_name, since_step=since_step)


@router.get("/logs", response_model=list[LogLine])
def get_logs(
    folder: str = Query(...),
    since_seq: int = Query(default=0, ge=0),
    limit: int = Query(default=1000, ge=1, le=5000),
    state: AppState = Depends(get_state),
) -> list[dict]:
    """训练日志。``since_seq`` 让前端只拉增量，不再每次全量重拉。"""
    return service.training_logs(state, folder, since_seq=since_seq, limit=limit)


@router.get("/records", response_model=list[RunRecord])
def get_records(state: AppState = Depends(get_state)) -> list[dict]:
    return service.list_runs(state)


@router.get("/records/{folder_name}")
def get_record_detail(folder_name: str, state: AppState = Depends(get_state)) -> dict:
    return service.get_run_detail(state, folder_name)


@router.delete("/records/{folder_name}")
def delete_record(folder_name: str, state: AppState = Depends(get_state)) -> dict:
    """删除训练记录。

    旧版此处 ``os.path.join(CHECKPOINTS_DIR, folder_name)`` 后直接
    ``shutil.rmtree``，传 ``../..`` 可删除任意目录。
    现在 ``folder_name`` 经 ``safe_name`` 校验，且只删 DB 记录不碰磁盘。
    """
    service.delete_run(state, folder_name)
    return {"message": "记录已删除（磁盘权重保留）"}


@router.get("/folders", response_model=list[str])
def get_existing_folders(state: AppState = Depends(get_state)) -> list[str]:
    return service.list_checkpoint_folders(state)
