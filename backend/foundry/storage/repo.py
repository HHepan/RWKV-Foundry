"""数据访问层（Repository）。

所有 SQL 集中在此，上层（API）不写 SQL、不碰 Connection。
这样换存储后端（例如将来上 PostgreSQL）只需替换本层。

每个函数都只接受**已校验的**标识符。路径类参数由 ``foundry.paths.safe_join``
在进入前处理，本层不再重复校验，但也不信任（SQL 全部参数化，无字符串拼接）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from foundry.domain.recipe import Recipe
from foundry.errors import ConflictError, NotFoundError
from foundry.storage.db import Database


# ── 工具 ─────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ── 配方 ─────────────────────────────────────────────────────────────────
def upsert_recipe(db: Database, recipe: Recipe) -> int:
    """按 hash 去重写入配方，返回 recipe_id。

    同一份参数永远得到同一个 id——这是横向对比多次训练的前提。
    """
    recipe.validate()
    conn = db.conn
    row = conn.execute("SELECT id FROM recipe WHERE hash = ?", (recipe.hash,)).fetchone()
    if row is not None:
        return int(row["id"])

    with db.transaction() as tx:
        cur = tx.execute(
            "INSERT INTO recipe (hash, payload, created_at) VALUES (?, ?, ?)",
            (recipe.hash, _dump(recipe), _now()),
        )
        return int(cur.lastrowid)


def _dump(recipe: Recipe) -> str:
    import json

    return json.dumps(recipe.canonical_payload(), ensure_ascii=False, sort_keys=True)


def get_recipe(db: Database, recipe_id: int) -> Recipe:
    row = db.conn.execute(
        "SELECT payload FROM recipe WHERE id = ?", (recipe_id,)
    ).fetchone()
    if row is None:
        raise NotFoundError(f"配方 {recipe_id} 不存在")
    import json

    return Recipe.from_payload(json.loads(row["payload"]))


# ── 训练运行 ─────────────────────────────────────────────────────────────
def create_run(
    db: Database,
    *,
    folder: str,
    recipe: Recipe,
    base_model: str,
    dataset: str,
    status: str = "pending",
) -> int:
    """登记一次训练。``folder`` 必须唯一（对应 checkpoints 下的目录名）。"""
    recipe_id = upsert_recipe(db, recipe)
    try:
        with db.transaction() as tx:
            cur = tx.execute(
                """
                INSERT INTO train_run
                    (folder, recipe_id, base_model, dataset, status, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (folder, recipe_id, base_model, dataset, status, _now()),
            )
            return int(cur.lastrowid)
    except Exception as exc:
        if "UNIQUE" in str(exc):
            raise ConflictError(f"训练目录 {folder!r} 已存在") from exc
        raise


def update_run_status(
    db: Database,
    run_id: int,
    status: str,
    *,
    exit_code: int | None = None,
    pid: int | None = None,
    error: str | None = None,
) -> None:
    """更新运行状态。完成/失败/停止时补 finished_at。"""
    terminal = status in ("completed", "stopped", "error")
    with db.transaction() as tx:
        tx.execute(
            """
            UPDATE train_run
               SET status = ?,
                   exit_code = COALESCE(?, exit_code),
                   pid = COALESCE(?, pid),
                   error = COALESCE(?, error),
                   finished_at = CASE WHEN ? THEN COALESCE(finished_at, ?) ELSE finished_at END
             WHERE id = ?
            """,
            (status, exit_code, pid, error, 1 if terminal else 0, _now(), run_id),
        )


def get_run_by_folder(db: Database, folder: str) -> dict:
    row = db.conn.execute(
        """
        SELECT r.*, rc.hash AS recipe_hash
          FROM train_run r LEFT JOIN recipe rc ON rc.id = r.recipe_id
         WHERE r.folder = ?
        """,
        (folder,),
    ).fetchone()
    if row is None:
        raise NotFoundError(f"训练记录 {folder!r} 不存在")
    return dict(row)


def list_runs(db: Database, *, limit: int = 100) -> list[dict]:
    rows = db.conn.execute(
        """
        SELECT r.*, rc.hash AS recipe_hash
          FROM train_run r LEFT JOIN recipe rc ON rc.id = r.recipe_id
         ORDER BY COALESCE(r.started_at, '') DESC
         LIMIT ?
        """,
        (limit,),
    )
    return [dict(r) for r in rows]


def delete_run(db: Database, folder: str) -> None:
    """删除运行记录及其 loss/log（CASCADE）。不删除磁盘产物。"""
    with db.transaction() as tx:
        cur = tx.execute("DELETE FROM train_run WHERE folder = ?", (folder,))
        if cur.rowcount == 0:
            raise NotFoundError(f"训练记录 {folder!r} 不存在")


# ── 会话 ─────────────────────────────────────────────────────────────────
def create_session(
    db: Database,
    *,
    model_ref: str,
    name: str = "",
    role_id: str | None = None,
    params: str = "{}",
) -> int:
    try:
        with db.transaction() as tx:
            cur = tx.execute(
                """
                INSERT INTO session (model_ref, name, role_id, params, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (model_ref, name, role_id, params, _now(), _now()),
            )
            return int(cur.lastrowid)
    except Exception as exc:
        if "UNIQUE" in str(exc):
            raise ConflictError(f"会话 {model_ref!r}/{name!r} 已存在") from exc
        raise


def touch_session(db: Database, session_id: int, *, params: str | None = None) -> None:
    with db.transaction() as tx:
        if params is None:
            tx.execute(
                "UPDATE session SET updated_at = ? WHERE id = ?", (_now(), session_id)
            )
        else:
            tx.execute(
                "UPDATE session SET params = ?, updated_at = ? WHERE id = ?",
                (params, _now(), session_id),
            )


def list_sessions(db: Database, *, model_ref: str | None = None) -> list[dict]:
    """列会话。一次 JOIN 取消息条数，避免旧版「遍历目录树解析 JSON」。"""
    sql = """
        SELECT s.*, COUNT(m.id) AS message_count
          FROM session s LEFT JOIN message m ON m.session_id = s.id
    """
    params: list = []
    if model_ref is not None:
        sql += " WHERE s.model_ref = ?"
        params.append(model_ref)
    sql += " GROUP BY s.id ORDER BY s.updated_at DESC"
    return [dict(r) for r in db.conn.execute(sql, params)]


def delete_session(db: Database, session_id: int) -> None:
    with db.transaction() as tx:
        cur = tx.execute("DELETE FROM session WHERE id = ?", (session_id,))
        if cur.rowcount == 0:
            raise NotFoundError(f"会话 {session_id} 不存在")


# ── 消息 ─────────────────────────────────────────────────────────────────
def replace_messages(db: Database, session_id: int, messages: list[dict]) -> None:
    """整体替换一个会话的消息。

    对话是整体演进的（重生成会回滚末尾），逐条 diff 收益不抵复杂度。
    单事务内先删后插，外部不会看到中间状态。
    """
    with db.transaction() as tx:
        tx.execute("DELETE FROM message WHERE session_id = ?", (session_id,))
        for seq, msg in enumerate(messages, 1):
            tx.execute(
                """
                INSERT INTO message (session_id, seq, role, content, thinking)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    seq,
                    msg.get("role", "user"),
                    msg.get("content", ""),
                    msg.get("thinking", "") or "",
                ),
            )
        tx.execute("UPDATE session SET updated_at = ? WHERE id = ?", (_now(), session_id))


def get_messages(db: Database, session_id: int) -> list[dict]:
    rows = db.conn.execute(
        "SELECT seq, role, content, thinking FROM message "
        "WHERE session_id = ? ORDER BY seq",
        (session_id,),
    )
    return [dict(r) for r in rows]


# ── 角色卡 ───────────────────────────────────────────────────────────────
def upsert_role(db: Database, *, role_id: str, name: str, content: str) -> None:
    with db.transaction() as tx:
        tx.execute(
            """
            INSERT INTO role (id, name, content, updated_at) VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                content = excluded.content,
                updated_at = excluded.updated_at
            """,
            (role_id, name, content, _now()),
        )


def list_roles(db: Database) -> list[dict]:
    return [dict(r) for r in db.conn.execute("SELECT * FROM role ORDER BY name")]


def delete_role(db: Database, role_id: str) -> None:
    with db.transaction() as tx:
        cur = tx.execute("DELETE FROM role WHERE id = ?", (role_id,))
        if cur.rowcount == 0:
            raise NotFoundError(f"角色 {role_id!r} 不存在")


# ── 数据集索引 ───────────────────────────────────────────────────────────
def get_dataset_index(db: Database, folder: str, filename: str) -> dict | None:
    row = db.conn.execute(
        "SELECT * FROM dataset WHERE folder = ? AND filename = ?", (folder, filename)
    ).fetchone()
    return dict(row) if row else None


def upsert_dataset_index(
    db: Database,
    *,
    folder: str,
    filename: str,
    mtime: float,
    size: int,
    line_count: int,
    rounds_single: int,
    rounds_multi: int,
) -> None:
    with db.transaction() as tx:
        tx.execute(
            """
            INSERT INTO dataset (folder, filename, mtime, size, line_count,
                                 rounds_single, rounds_multi, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(folder, filename) DO UPDATE SET
                mtime = excluded.mtime,
                size = excluded.size,
                line_count = excluded.line_count,
                rounds_single = excluded.rounds_single,
                rounds_multi = excluded.rounds_multi,
                indexed_at = excluded.indexed_at
            """,
            (folder, filename, mtime, size, line_count, rounds_single, rounds_multi, _now()),
        )


# ── 铸件 ─────────────────────────────────────────────────────────────────
def upsert_checkpoint(
    db: Database,
    *,
    path: str,
    run_id: int | None,
    epoch: int | None = None,
    size_bytes: int | None = None,
) -> None:
    with db.transaction() as tx:
        tx.execute(
            """
            INSERT INTO checkpoint (run_id, path, epoch, size_bytes, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                size_bytes = COALESCE(excluded.size_bytes, checkpoint.size_bytes),
                epoch = COALESCE(excluded.epoch, checkpoint.epoch)
            """,
            (run_id, path, epoch, size_bytes, _now()),
        )


def list_checkpoints(db: Database, *, run_id: int | None = None) -> list[dict]:
    sql = "SELECT * FROM checkpoint"
    params: list = []
    if run_id is not None:
        sql += " WHERE run_id = ?"
        params.append(run_id)
    sql += " ORDER BY created_at DESC"
    return [dict(r) for r in db.conn.execute(sql, params)]


def checkpoint_path_of_run(run_dir: Path) -> list[Path]:
    """列出某 run 目录下的权重文件（磁盘视角）。"""
    if not run_dir.is_dir():
        return []
    return sorted(run_dir.glob("rwkv-*.pth"))
