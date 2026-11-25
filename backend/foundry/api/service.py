"""服务层 —— 编排领域逻辑，供路由调用。

## 为什么不让路由直接调领域层

旧版 ``api/train.py`` 有 537 行，里面混着：路径拼接、subprocess 管理、
JSON 读写、状态机、错误处理。**路由本不该承载业务逻辑。**

本层的职责边界：
    - 组合多个领域模块完成一个用例
    - 把领域异常翻译成服务层语义
    - 不写 SQL、不碰 HTTP、不拼 shell

路由只做：解析请求 → 调服务 → 返回响应。
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from pathlib import Path

from foundry.api.deps import AppState
from foundry.domain.dataset import (
    delete_row as ds_delete_row,
)
from foundry.domain.dataset import (
    read_page,
    scan_stats,
)
from foundry.domain.dataset import (
    update_row as ds_update_row,
)
from foundry.domain.recipe import Recipe, RecipeError
from foundry.errors import ConflictError, FoundryError, NotFoundError
from foundry.infer.params import InferParams
from foundry.infer.prompt import ChatMessage
from foundry.paths import safe_join, safe_name
from foundry.storage import ingest, repo
from foundry.train.job import TrainError, TrainJob
from foundry.train.preprocess import BinidxError, default_python_exe, run_preprocess
from foundry.train.script import build_script, resolve_dataset_path, write_recipe, write_script

logger = logging.getLogger("foundry")


class ServiceError(FoundryError):
    """服务层错误（已包含足够的用户可读信息）。"""


class BusyError(ServiceError):
    """资源正被占用（例如已有训练在跑）。"""


# ── 数据集 ───────────────────────────────────────────────────────────────
def list_dataset_folders(state: AppState) -> list[str]:
    data_dir = state.paths.data
    if not data_dir.is_dir():
        return []
    out = ["./"]
    out += sorted(f"./{p.name}" for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    return out


def list_dataset_files(state: AppState, folder: str) -> list[str]:
    base = _resolve_dataset_dir(state, folder)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.suffix == ".jsonl" and not p.name.startswith("."))


def _resolve_dataset_dir(state: AppState, folder: str) -> Path:
    """解析数据集子目录，拒绝越界。``folder`` 形如 ``"./out"`` 或 ``"out"``。"""
    cleaned = folder.strip()
    if cleaned in ("", ".", "./"):
        return state.paths.data
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    parts = [p for p in Path(cleaned).parts if p not in ("", ".")]
    if not parts:
        return state.paths.data
    return safe_join(state.paths.data, *parts, kind="数据集目录")


def _dataset_file(state: AppState, folder: str, filename: str) -> Path:
    base = _resolve_dataset_dir(state, folder)
    target = safe_join(base, filename, kind="数据集文件名")
    if not target.is_file():
        raise NotFoundError(f"数据集 {filename!r} 不存在")
    return target


def page_dataset(state: AppState, folder: str, filename: str, *, page: int, size: int, rounds_filter: str) -> dict:
    path = _dataset_file(state, folder, filename)

    # total 优先走索引缓存：避免为显示页码而全扫文件（旧版 308ms）
    total = _cached_line_count(state, folder, filename, path)
    return read_page(path, page=page, size=size, rounds_filter=rounds_filter, total_override=total)


def dataset_stats(state: AppState, folder: str, filename: str) -> dict:
    path = _dataset_file(state, folder, filename)
    st = path.stat()
    norm_folder = _norm_folder(folder)

    cached = repo.get_dataset_index(state.db, norm_folder, filename)
    if cached and cached["mtime"] == st.st_mtime and cached["size"] == st.st_size:
        return {
            "filename": filename,
            "folder": norm_folder,
            "line_count": cached["line_count"],
            "rounds_single": cached["rounds_single"],
            "rounds_multi": cached["rounds_multi"],
            "cached": True,
        }

    stats = scan_stats(path)
    repo.upsert_dataset_index(
        state.db,
        folder=norm_folder,
        filename=filename,
        mtime=st.st_mtime,
        size=st.st_size,
        line_count=stats.line_count,
        rounds_single=stats.rounds_single,
        rounds_multi=stats.rounds_multi,
    )
    return {
        "filename": filename,
        "folder": norm_folder,
        "line_count": stats.line_count,
        "rounds_single": stats.rounds_single,
        "rounds_multi": stats.rounds_multi,
        "cached": False,
    }


def _cached_line_count(state: AppState, folder: str, filename: str, path: Path) -> int | None:
    norm = _norm_folder(folder)
    cached = repo.get_dataset_index(state.db, norm, filename)
    if cached:
        st = path.stat()
        if cached["mtime"] == st.st_mtime and cached["size"] == st.st_size:
            return int(cached["line_count"])
    # 缓存失效：后台重建代价高，此处直接返回 None（前端不做精确分页也可用）
    return None


def _norm_folder(folder: str) -> str:
    cleaned = folder.strip().lstrip("./")
    return cleaned if cleaned not in ("", ".") else ""


def update_dataset_row(state: AppState, folder: str, filename: str, row_id: int, text: str) -> None:
    path = _dataset_file(state, folder, filename)
    ds_update_row(path, row_id, text)
    _invalidate_index(state, folder, filename)


def delete_dataset_row(state: AppState, folder: str, filename: str, row_id: int) -> None:
    path = _dataset_file(state, folder, filename)
    ds_delete_row(path, row_id)
    _invalidate_index(state, folder, filename)


def delete_dataset_file(state: AppState, folder: str, filename: str) -> None:
    path = _dataset_file(state, folder, filename)
    path.unlink()
    with state.db.transaction() as tx:
        tx.execute(
            "DELETE FROM dataset WHERE folder = ? AND filename = ?",
            (_norm_folder(folder), filename),
        )


def _invalidate_index(state: AppState, folder: str, filename: str) -> None:
    with state.db.transaction() as tx:
        tx.execute(
            "DELETE FROM dataset WHERE folder = ? AND filename = ?",
            (_norm_folder(folder), filename),
        )


def merge_datasets(
    state: AppState,
    *,
    source_files: list[dict[str, str]],
    new_name: str,
    folder: str,
    shuffle: bool,
    counts: dict[str, int],
) -> dict:
    import random

    if not source_files:
        raise ServiceError("请至少选择一个源文件")
    if not new_name.strip():
        raise ServiceError("新文件名不能为空")

    safe_name(Path(new_name).stem, kind="数据集文件名")
    if not new_name.endswith(".jsonl"):
        new_name += ".jsonl"

    merged: list[str] = []
    original_total = 0

    for item in source_files:
        fn = item.get("filename", "")
        fd = item.get("folder", "")
        path = _dataset_file(state, fd, fn)
        rows = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        original_total += len(rows)

        key = f"{_norm_folder(fd)}/{fn}" if _norm_folder(fd) else fn
        limit = counts.get(key, 0)
        if limit and 0 < limit < len(rows):
            rows = random.sample(rows, limit)
        merged.extend(rows)

    if shuffle:
        random.shuffle(merged)

    out_dir = _resolve_dataset_dir(state, folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = safe_join(out_dir, new_name, kind="数据集文件名")
    out_path.write_text("\n".join(merged) + "\n", encoding="utf-8")

    return {
        "filename": new_name,
        "folder": _norm_folder(folder),
        "total_lines": len(merged),
        "original_lines": original_total,
    }


# ── 基底模型 ─────────────────────────────────────────────────────────────
_MODEL_EXTS = {".pth", ".bin", ".safetensors", ".pt", ".ckpt", ".h5", ".model"}


def list_base_models(state: AppState) -> list[str]:
    d = state.paths.base_models
    if not d.is_dir():
        return []
    return sorted(
        p.name for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in _MODEL_EXTS and not p.name.startswith(".")
    )


def resolve_model_ref(state: AppState, model_ref: str) -> tuple[Path, str]:
    """把 ``"<folder>/<stem>"`` 解析为不含后缀的绝对路径。

    ``folder="base_models"`` 表示基底模型，其余视为 checkpoints 下的 run 目录。
    返回 ``(路径前缀, 是否基底)``。
    """
    parts = [p for p in Path(model_ref.strip()).parts if p not in ("", ".")]
    if len(parts) < 2:
        raise ServiceError(f"模型引用格式应为 '<folder>/<name>'，当前 {model_ref!r}")

    folder, *rest = parts
    if folder == "base_models":
        target = safe_join(state.paths.base_models, *rest, kind="模型名")
    else:
        safe_name(folder, kind="训练目录名")
        target = safe_join(state.paths.checkpoints / folder, *rest, kind="模型名")

    text = str(target)
    stem_text = text[:-4] if text.endswith(".pth") else text

    # 必须用字符串切片裁掉 .pth，不能用 Path.with_suffix("")。
    #
    # Path("rwkv7-g1a-0.1b").with_suffix(".pth") 得到 rwkv7-g1a-0.1.pth ——
    # 它把 ".1b" 当成扩展名替换了。RWKV 权重名普遍含多个点
    # （rwkv7-g1a-0.1b-20250728.pth），with_suffix 会指向不存在的文件。
    if not Path(stem_text + ".pth").is_file() and not Path(stem_text).is_file():
        raise NotFoundError(f"模型文件不存在: {stem_text}.pth")
    return Path(stem_text), folder == "base_models"


# ── 配方 ─────────────────────────────────────────────────────────────────
def validate_recipe(raw: dict) -> dict:
    try:
        recipe = Recipe.from_payload(raw)
        recipe.validate()
    except RecipeError as exc:
        return {
            "hash": "",
            "valid": False,
            "problems": exc.problems,
            "n_layer": None,
            "n_embd": None,
            "peft_config": None,
        }
    return {
        "hash": recipe.hash,
        "valid": True,
        "problems": [],
        "n_layer": recipe.n_layer,
        "n_embd": recipe.n_embd,
        "peft_config": recipe.peft_config,
    }


def preview_script(state: AppState, raw: dict, save_folder: str) -> dict:
    recipe = Recipe.from_payload(raw).validate()
    safe_name(save_folder, kind="保存目录名")

    run_dir = state.paths.checkpoints / save_folder
    base_prefix, _ = resolve_model_ref(state, f"base_models/{recipe.base_model}")
    dataset_path = resolve_dataset_path(state.paths, recipe.dataset)
    if not dataset_path.is_file():
        raise NotFoundError(f"数据集不存在: {dataset_path}")

    binidx_prefix = (
        state.paths.binidx_data / save_folder / Path(recipe.dataset).stem
    )
    script = build_script(
        recipe,
        state.paths,
        run_dir=run_dir,
        base_model_path=base_prefix,
        binidx_prefix=binidx_prefix,
    )
    return {"hash": recipe.hash, "script": script}


# ── 训练 ─────────────────────────────────────────────────────────────────
def start_training(state: AppState, raw: dict, save_folder: str) -> dict:
    """启动训练。严格串行——一个进程只跑一个训练。"""
    if state.active_job() is not None:
        active = state.active_job()
        raise BusyError(
            f"已有训练在运行（{active.folder}，状态 {active.status.value}）。"
            f"请先停止它，再生产新的铸件。"
        )

    recipe = Recipe.from_payload(raw).validate()
    safe_name(save_folder, kind="保存目录名")

    run_dir = state.paths.checkpoints / save_folder
    if (run_dir / "train_inf.json").exists() or repo_has_run(state, save_folder):
        raise ConflictError(f"保存目录 {save_folder!r} 已被占用，请换一个名字")

    base_prefix, _ = resolve_model_ref(state, f"base_models/{recipe.base_model}")
    dataset_path = resolve_dataset_path(state.paths, recipe.dataset)
    if not dataset_path.is_file():
        raise NotFoundError(f"数据集不存在: {dataset_path}")

    # 1. 登记 run（先入库，失败时状态可追溯）
    run_id = repo.create_run(
        state.db,
        folder=save_folder,
        recipe=recipe,
        base_model=recipe.base_model,
        dataset=recipe.dataset,
        status="pending",
    )

    try:
        # 2. binidx 预处理（每 run 独立子目录）
        prefix = run_preprocess(
            peft_dir=state.paths.peft_dir,
            python_exe=default_python_exe(),
            src_jsonl=dataset_path,
            run_folder=save_folder,
            vocab=state.paths.vocab,
        )

        # 3. 生成脚本与配方快照
        script_content = build_script(
            recipe, state.paths, run_dir=run_dir, base_model_path=base_prefix, binidx_prefix=prefix
        )
        script_path = write_script(script_content, run_dir)
        write_recipe(recipe, run_dir)

        # 4. 启动子进程
        job = TrainJob(
            run_id=run_id, folder=save_folder, recipe=recipe, paths=state.paths, run_dir=run_dir
        )
        job.start(script_path)
        state.register_job(job)
        repo.update_run_status(state.db, run_id, "running", pid=job.pid)

        # 5. 后台监控：进程退出时把最终状态写回数据库
        #
        # 没有这一步，训练自然完成后 DB 里 status 会永远停在 running，
        # finished_at / exit_code 永远为 null —— 前端轮询看不到结束。
        # 这个缺陷只在真实训练跑完时暴露，fake 测试发现不了。
        def _on_finished(finished_job: TrainJob) -> None:
            try:
                # 先把最后几步 loss/日志灌进来，再写终态
                with contextlib.suppress(Exception):
                    ingest.ingest_run(state.db, finished_job.run_id, finished_job.run_dir)
                repo.update_run_status(
                    state.db,
                    finished_job.run_id,
                    finished_job.status.value,
                    exit_code=finished_job.exit_code,
                    error=finished_job.error,
                )
            except Exception:  # noqa: BLE001
                pass

        job.monitor(on_finished=_on_finished)

    except (BinidxError, TrainError, OSError) as exc:
        repo.update_run_status(state.db, run_id, "error", error=str(exc))
        raise ServiceError(f"启动训练失败: {exc}") from exc

    return {
        "folder": save_folder,
        "run_id": run_id,
        "recipe_hash": recipe.hash,
        "status": job.status.value,
        "pid": job.pid,
        "pgid": job.pgid,
    }


def repo_has_run(state: AppState, folder: str) -> bool:
    try:
        repo.get_run_by_folder(state.db, folder)
        return True
    except NotFoundError:
        return False


def stop_training(state: AppState) -> dict:
    """三段式停止：SIGTERM 全组 → 宽限 → SIGKILL 全组。"""
    job = state.active_job()
    if job is None:
        raise NotFoundError("当前没有正在运行的训练")

    status, note = job.kill()
    repo.update_run_status(
        state.db, job.run_id, status.value, exit_code=job.exit_code, error=job.error
    )
    # 收尾时再灌一次，确保最后几步 loss 不丢
    with contextlib.suppress(Exception):
        ingest.ingest_run(state.db, job.run_id, job.run_dir)

    return {"status": status.value, "note": note}


def training_status(state: AppState, folder: str | None = None) -> dict:
    job = state.active_job()
    if job is not None and (folder is None or job.folder == folder):
        snap = job.snapshot()
        return {
            "folder": snap["folder"],
            "status": snap["status"],
            "pid": snap["pid"],
            "exit_code": snap["exit_code"],
            "error": snap["error"],
            "elapsed_seconds": snap["elapsed_seconds"],
            "recipe_hash": snap["recipe_hash"],
        }

    if folder is None:
        return {"folder": None, "status": "idle"}

    try:
        row = repo.get_run_by_folder(state.db, folder)
    except NotFoundError:
        raise
    return {
        "folder": row["folder"],
        "status": row["status"],
        "pid": row["pid"],
        "exit_code": row["exit_code"],
        "error": row["error"],
        "elapsed_seconds": None,
        "recipe_hash": row.get("recipe_hash"),
    }


def loss_series(state: AppState, folder: str, *, since_step: int = 0) -> list[dict]:
    run = repo.get_run_by_folder(state.db, folder)
    ingest.ingest_run(state.db, run["id"], state.paths.checkpoints / folder)
    return ingest.get_loss_series(state.db, run["id"], since_step=since_step)


def training_logs(state: AppState, folder: str, *, since_seq: int = 0, limit: int = 1000) -> list[dict]:
    run = repo.get_run_by_folder(state.db, folder)
    ingest.ingest_run(state.db, run["id"], state.paths.checkpoints / folder)
    return ingest.get_logs(state.db, run["id"], since_seq=since_seq, limit=limit)


def list_runs(state: AppState) -> list[dict]:
    return repo.list_runs(state.db)


def get_run_detail(state: AppState, folder: str) -> dict:
    run = repo.get_run_by_folder(state.db, folder)
    ingest.ingest_run(state.db, run["id"], state.paths.checkpoints / folder)
    run["loss_history"] = ingest.get_loss_series(state.db, run["id"])
    run["logs"] = ingest.get_logs(state.db, run["id"], limit=5000)
    return run


def delete_run(state: AppState, folder: str) -> None:
    """删除训练记录。只删 DB 记录，不删磁盘权重——那是用户资产。"""
    safe_name(folder, kind="训练目录名")
    repo.delete_run(state.db, folder)
    state.drop_job(folder)


def list_checkpoint_folders(state: AppState) -> list[str]:
    d = state.paths.checkpoints
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))


def list_checkpoint_files(state: AppState, folder: str) -> list[str]:
    if not folder:
        return []
    safe_name(folder, kind="训练目录名")
    d = state.paths.checkpoints / folder
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.suffix == ".pth" and not p.name.startswith("."))


# ── 角色卡 ───────────────────────────────────────────────────────────────
def list_roles(state: AppState) -> list[dict]:
    return repo.list_roles(state.db)


def save_role(state: AppState, *, role_id: str | None, name: str, content: str) -> dict:
    rid = role_id or f"custom_{int(time.time() * 1000)}"
    safe_name(rid, kind="角色 ID")
    if not name.strip():
        raise ServiceError("角色名不能为空")
    repo.upsert_role(state.db, role_id=rid, name=name.strip(), content=content)
    return {"id": rid, "name": name.strip(), "content": content}


def delete_role(state: AppState, role_id: str) -> None:
    repo.delete_role(state.db, role_id)


# ── 会话与对话 ───────────────────────────────────────────────────────────
def ensure_model_loaded(state: AppState, model_ref: str) -> str:
    """确保模型已加载。返回实际使用的 model_ref。

    ## 切换模型必须先卸载旧的

    这是实际踩过的坑：从 2.9B 换到 13.3B 时，2.9B 仍占着约 6GB 显存。
    13.3B 需要约 25GB，单卡只有 24GB —— 不先释放就直接 OOM，
    而错误信息只会显示 cuda out of memory，很难定位到「上个模型没卸」。

    因此顺序是：卸载 → 清空会话态 → 加载新模型。
    即使加载失败，也不会留下半新半旧的脏状态。
    """
    if state.engine.loaded and state.engine.model_ref == model_ref:
        return model_ref

    stem, _ = resolve_model_ref(state, model_ref)
    existing = state.active_job()
    if existing is not None:
        raise BusyError("训练进行中，暂不能切换推理模型（显存竞争）")

    # ★ 先卸载：释放旧权重与显存，再加载新模型
    if state.engine.loaded:
        logger.info("切换模型：卸载 %s → 加载 %s", state.engine.model_ref, model_ref)
        state.loader.unload()
        state.engine.detach()

    try:
        pipeline, tokenizer = state.loader.load(
            str(stem), strategy=state.config.infer.strategy
        )
    except Exception as exc:  # noqa: BLE001
        # 加载失败时保持「无模型」的干净状态，而不是留着半截状态
        state.engine.detach()
        raise ServiceError(
            f"加载模型失败: {exc}\n"
            f"  模型: {stem}.pth\n"
            f"  显存策略: {state.config.infer.strategy}\n"
            f"  提示: 13.3B 需要约 25GB，单卡 24GB 不够，需用双卡切分，例如\n"
            f"        FOUNDRY_INFER_STRATEGY=\"cuda:0 bf16 -> cuda:1 bf16\""
        ) from exc

    state.engine.attach(pipeline, tokenizer, model_ref=model_ref)
    return model_ref


def chat(
    state: AppState,
    *,
    model_ref: str,
    message: str,
    history: list[dict],
    params: dict,
    thinking_mode: bool,
    session_name: str,
    role_id: str | None,
    on_token=None,
    should_stop=None,
) -> dict:
    ensure_model_loaded(state, model_ref)

    session_key = f"{model_ref}|{session_name}" if session_name else model_ref
    infer_params = InferParams.from_dict(params).validate()

    messages = [ChatMessage.from_dict(m) for m in history]

    result = state.engine.generate(
        session_key=session_key,
        user_message=message,
        history=messages,
        params=infer_params,
        thinking_mode=thinking_mode,
        on_token=on_token,
        should_stop=should_stop,
    )

    # 落库：完整对话 = 历史 + 本轮用户 + 本轮回复
    try:
        sid = _upsert_session(state, model_ref, session_name, role_id, infer_params)
        stored = [m.to_dict() for m in messages]
        stored.append({"role": "user", "content": message})
        stored.append(
            {
                "role": "assistant",
                "content": result.content,
                "thinking": result.thinking,
            }
        )
        repo.replace_messages(state.db, sid, stored)
    except Exception:  # noqa: BLE001
        # 落库失败不应让用户丢失已经生成的回复
        pass

    return result.as_dict()


def _upsert_session(
    state: AppState, model_ref: str, session_name: str, role_id: str | None, params: InferParams
) -> int:
    params_json = json.dumps(params.as_dict(), ensure_ascii=False)
    for row in repo.list_sessions(state.db, model_ref=model_ref):
        if row["name"] == session_name:
            repo.touch_session(state.db, row["id"], params=params_json)
            return int(row["id"])
    return repo.create_session(
        state.db, model_ref=model_ref, name=session_name, role_id=role_id, params=params_json
    )


def regenerate(state: AppState, **kwargs) -> dict:
    """重生成：回滚 state 后重新生成。"""
    kwargs["regenerate"] = True
    on_token = kwargs.pop("on_token", None)
    should_stop = kwargs.pop("should_stop", None)

    ensure_model_loaded(state, kwargs["model_ref"])
    session_key = (
        f"{kwargs['model_ref']}|{kwargs['session_name']}"
        if kwargs.get("session_name")
        else kwargs["model_ref"]
    )
    infer_params = InferParams.from_dict(kwargs.get("params") or {}).validate()
    messages = [ChatMessage.from_dict(m) for m in kwargs.get("history") or []]

    result = state.engine.generate(
        session_key=session_key,
        user_message=kwargs["message"],
        history=messages,
        params=infer_params,
        thinking_mode=bool(kwargs.get("thinking_mode")),
        on_token=on_token,
        should_stop=should_stop,
        rollback=True,
    )
    return result.as_dict()


def reset_state(state: AppState, model_ref: str, session_name: str = "") -> None:
    session_key = f"{model_ref}|{session_name}" if session_name else model_ref
    state.engine.reset_session(session_key)


def list_chat_models(state: AppState) -> list[dict]:
    out: list[dict] = []
    for row in repo.list_sessions(state.db):
        try:
            params = json.loads(row["params"] or "{}")
        except json.JSONDecodeError:
            params = {}
        out.append(
            {
                "model": row["model_ref"],
                "session": row["name"],
                "role_id": row["role_id"] or "none",
                "created_at": row["created_at"],
                "params": params,
                "message_count": row["message_count"],
            }
        )
    return out


def get_chat_data(state: AppState, model_ref: str, session_name: str = "") -> dict:
    for row in repo.list_sessions(state.db, model_ref=model_ref):
        if row["name"] == session_name:
            try:
                params = json.loads(row["params"] or "{}")
            except json.JSONDecodeError:
                params = {}
            return {
                "model": row["model_ref"],
                "session": row["name"],
                "role_id": row["role_id"],
                "params": params,
                "dialogue_content": repo.get_messages(state.db, row["id"]),
            }
    raise NotFoundError(f"对话记录不存在: {model_ref} / {session_name!r}")


def delete_chat_data(state: AppState, model_ref: str, session_name: str = "") -> None:
    """删除某个会话的持久化对话记录，并清掉引擎里的循环态。

    ## 为什么是幂等的

    删除操作天然应当幂等：目标不存在 = 已达到期望状态 = 成功。

    之前这里在找不到记录时抛 NotFoundError，导致「清空一个还没聊过的新对话」
    在 UI 上报错。前端被迫吞掉 404 才能正常工作 —— 那是把语义问题推给了调用方。

    现在无论记录是否存在都返回成功：找到就删，找不到也无妨。
    引擎态同样无条件清理，避免残留脏 state。
    """
    session_key = f"{model_ref}|{session_name}" if session_name else model_ref

    for row in repo.list_sessions(state.db, model_ref=model_ref):
        if row["name"] == session_name:
            repo.delete_session(state.db, row["id"])
            break

    # 无论数据库里有没有记录，内存里的循环态都要清掉
    state.engine.drop_session(session_key)
