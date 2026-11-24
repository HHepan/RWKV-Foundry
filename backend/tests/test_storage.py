"""foundry.storage 的测试。

重点验证四件在旧项目里出过问题的事：
  1. 迁移幂等、版本前滚、高版本库被拒绝
  2. ingester 幂等（重复 tail 不产生重复行）
  3. 单写入者原则（loss/log 只有一份真相）
  4. 配方哈希不因输出目录变化而变化
"""

from __future__ import annotations

import json
import threading

import pytest

from foundry.domain.recipe import Recipe
from foundry.errors import ConflictError, MigrationError, NotFoundError
from foundry.storage import ingest, repo
from foundry.storage.db import SCHEMA_VERSION, open_database


@pytest.fixture
def db(tmp_path):
    database = open_database(tmp_path / "foundry.db")
    yield database
    database.close()


@pytest.fixture
def recipe():
    return Recipe(
        base_model="rwkv7-g1a-0.1b.pth",
        dataset="out/train.jsonl",
        model_size="0.1B",
        ctx_len=512,
        epoch_steps=10,
        epoch_count=2,
    )


# ── 迁移 ─────────────────────────────────────────────────────────────────
def test_migrate_creates_schema(tmp_path):
    database = open_database(tmp_path / "a.db")
    try:
        assert database.schema_version() == SCHEMA_VERSION
        tables = {
            r["name"]
            for r in database.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for expected in (
            "recipe", "train_run", "train_step", "train_log",
            "dataset", "session", "message", "checkpoint", "role",
        ):
            assert expected in tables
    finally:
        database.close()


def test_migrate_is_idempotent(tmp_path):
    path = tmp_path / "b.db"
    d1 = open_database(path)
    v1 = d1.schema_version()
    d1.migrate()
    assert d1.schema_version() == v1
    d1.close()

    d2 = open_database(path)
    try:
        assert d2.schema_version() == SCHEMA_VERSION
    finally:
        d2.close()


def test_rejects_newer_schema(tmp_path):
    """被更新版程序创建的库，不能被旧版降级使用。"""
    path = tmp_path / "c.db"
    database = open_database(path)
    database.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    with pytest.raises(MigrationError, match="高于本程序支持"):
        database.migrate()
    database.close()


def test_wal_mode_enabled(db):
    assert db.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_foreign_keys_enabled(db):
    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


# ── 配方 ─────────────────────────────────────────────────────────────────
def test_recipe_dedup_by_hash(db, recipe):
    """同样参数写两次，只应得到一个 recipe 行。"""
    r1 = repo.upsert_recipe(db, recipe)
    r2 = repo.upsert_recipe(db, recipe)
    assert r1 == r2
    count = db.conn.execute("SELECT COUNT(*) FROM recipe").fetchone()[0]
    assert count == 1


def test_recipe_hash_ignores_output_folder(recipe):
    """配方哈希不得因输出目录变化而变——否则无法横向对比。"""
    a = Recipe(**{**recipe.__dict__, "base_model": recipe.base_model})
    # Recipe 里没有 save_folder 字段，本测试锁定这一点
    assert "save_folder" not in Recipe.__dataclass_fields__
    assert a.hash == recipe.hash


def test_recipe_hash_ignores_accelerator_and_devices(recipe):
    """speed 相关参数不进哈希（不影响数学结果）。"""
    payload = recipe.canonical_payload()
    assert "accelerator" not in payload
    assert "devices" not in payload
    assert "strategy" not in payload
    assert "grad_cp" not in payload


def test_recipe_hash_changes_with_math_params(recipe):
    other = Recipe(**{**recipe.__dict__, "lora_r": 64})
    assert other.hash != recipe.hash


def test_recipe_roundtrip(db, recipe):
    rid = repo.upsert_recipe(db, recipe)
    loaded = repo.get_recipe(db, rid)
    assert loaded.hash == recipe.hash
    assert loaded.lora_r == recipe.lora_r


def test_get_missing_recipe_raises(db):
    with pytest.raises(NotFoundError):
        repo.get_recipe(db, 999)


# ── 训练运行 ─────────────────────────────────────────────────────────────
def test_create_and_fetch_run(db, recipe):
    run_id = repo.create_run(
        db, folder="run1", recipe=recipe, base_model=recipe.base_model,
        dataset=recipe.dataset, status="running",
    )
    got = repo.get_run_by_folder(db, "run1")
    assert got["id"] == run_id
    assert got["status"] == "running"
    assert got["recipe_hash"] == recipe.hash


def test_duplicate_folder_conflicts(db, recipe):
    repo.create_run(db, folder="dup", recipe=recipe, base_model="m", dataset="d")
    with pytest.raises(ConflictError):
        repo.create_run(db, folder="dup", recipe=recipe, base_model="m", dataset="d")


def test_update_run_status_sets_finished_at(db, recipe):
    run_id = repo.create_run(db, folder="r2", recipe=recipe, base_model="m", dataset="d")
    repo.update_run_status(db, run_id, "completed")
    got = repo.get_run_by_folder(db, "r2")
    assert got["status"] == "completed"
    assert got["finished_at"] is not None


def test_running_status_does_not_set_finished_at(db, recipe):
    run_id = repo.create_run(db, folder="r3", recipe=recipe, base_model="m", dataset="d")
    repo.update_run_status(db, run_id, "running")
    assert repo.get_run_by_folder(db, "r3")["finished_at"] is None


def test_delete_run_cascades_steps(db, recipe):
    run_id = repo.create_run(db, folder="r4", recipe=recipe, base_model="m", dataset="d")
    with db.transaction() as tx:
        tx.execute(
            "INSERT INTO train_step (run_id, step, loss) VALUES (?, 1, 1.5)", (run_id,)
        )
    repo.delete_run(db, "r4")
    assert db.conn.execute("SELECT COUNT(*) FROM train_step").fetchone()[0] == 0


# ── ingester ─────────────────────────────────────────────────────────────
def _write_run_files(run_dir, losses, logs):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ingest.LOSS_FILE).write_text(
        "\n".join(json.dumps({"loss": v, "kt_s": 1.0}) for v in losses) + "\n",
        encoding="utf-8",
    )
    (run_dir / ingest.LOG_FILE).write_text("\n".join(logs) + "\n", encoding="utf-8")


def test_ingest_reads_upstream_files(db, recipe, tmp_path):
    """读取上游真实文件名 loss_data.jsonl / train_log.txt。"""
    run_dir = tmp_path / "checkpoints" / "run1"
    _write_run_files(run_dir, [1.8, 1.6, 1.4], ["line a", "line b"])
    run_id = repo.create_run(db, folder="run1", recipe=recipe, base_model="m", dataset="d")

    result = ingest.ingest_run(db, run_id, run_dir)
    assert result.steps_added == 3
    assert result.logs_added == 2
    assert result.errors == []

    series = ingest.get_loss_series(db, run_id)
    assert [r["loss"] for r in series] == [1.8, 1.6, 1.4]
    assert [r["step"] for r in series] == [1, 2, 3]


def test_ingest_is_idempotent(db, recipe, tmp_path):
    """重复灌同一个文件不得产生重复行——ingester 的首要要求。"""
    run_dir = tmp_path / "cp" / "run1"
    _write_run_files(run_dir, [1.0, 2.0], ["x"])
    run_id = repo.create_run(db, folder="run1", recipe=recipe, base_model="m", dataset="d")

    first = ingest.ingest_run(db, run_id, run_dir)
    second = ingest.ingest_run(db, run_id, run_dir)

    assert first.steps_added == 2
    assert second.steps_added == 0  # 没有新行
    assert second.logs_added == 0
    assert len(ingest.get_loss_series(db, run_id)) == 2


def test_ingest_is_incremental(db, recipe, tmp_path):
    """训练持续写，ingester 应只消费新增部分。"""
    run_dir = tmp_path / "cp" / "run1"
    _write_run_files(run_dir, [1.0], ["a"])
    run_id = repo.create_run(db, folder="run1", recipe=recipe, base_model="m", dataset="d")
    ingest.ingest_run(db, run_id, run_dir)

    # 训练继续追加
    with open(run_dir / ingest.LOSS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"loss": 0.5}) + "\n")
    with open(run_dir / ingest.LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write("b\n")

    result = ingest.ingest_run(db, run_id, run_dir)
    assert result.steps_added == 1
    assert result.logs_added == 1
    assert len(ingest.get_loss_series(db, run_id)) == 2


def test_ingest_survives_partial_line(db, recipe, tmp_path):
    """训练正在写的半行不应导致整批失败。"""
    run_dir = tmp_path / "cp" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / ingest.LOSS_FILE).write_text(
        json.dumps({"loss": 1.0}) + "\n" + '{"loss": 0.9, "kt',
        encoding="utf-8",
    )
    run_id = repo.create_run(db, folder="run1", recipe=recipe, base_model="m", dataset="d")

    result = ingest.ingest_run(db, run_id, run_dir)
    assert result.steps_added == 1  # 只吃到完整的
    assert len(ingest.get_loss_series(db, run_id)) == 1


def test_ingest_skips_missing_files(db, recipe, tmp_path):
    run_dir = tmp_path / "cp" / "empty"
    run_dir.mkdir(parents=True)
    run_id = repo.create_run(db, folder="empty", recipe=recipe, base_model="m", dataset="d")
    result = ingest.ingest_run(db, run_id, run_dir)
    assert result.steps_added == 0
    assert result.changed is False


def test_ingest_does_not_touch_finished_at(db, recipe, tmp_path):
    """ingester 不得写 finished_at —— 时间戳属于状态机。

    这个 bug 曾真实出现过：灌入时顺手 set finished_at，正在跑的训练
    第一次灌入就带上"结束时间"，且被 COALESCE 永久固定。
    """
    run_dir = tmp_path / "cp" / "run1"
    _write_run_files(run_dir, [1.0], ["a"])
    run_id = repo.create_run(db, folder="run1", recipe=recipe, base_model="m", dataset="d")

    ingest.ingest_run(db, run_id, run_dir)
    got = repo.get_run_by_folder(db, "run1")
    assert got["status"] == "pending"
    assert got["finished_at"] is None, "ingester 不应设置 finished_at"

    # 真正结束后才由状态机写入
    repo.update_run_status(db, run_id, "completed", exit_code=0)
    assert repo.get_run_by_folder(db, "run1")["finished_at"] is not None


def test_ingest_no_legacy_duplicate_files(db, recipe, tmp_path):
    """确认不再产生旧版的 loss_history.jsonl / train_logs.txt 双轨文件。"""
    run_dir = tmp_path / "cp" / "run1"
    _write_run_files(run_dir, [1.0], ["a"])
    run_id = repo.create_run(db, folder="run1", recipe=recipe, base_model="m", dataset="d")
    ingest.ingest_run(db, run_id, run_dir)

    names = {p.name for p in run_dir.iterdir()}
    assert "loss_history.jsonl" not in names
    assert "train_logs.txt" not in names


def test_log_incremental_pull(db, recipe, tmp_path):
    """前端带 since_seq 只拉增量，不再全量重拉。"""
    run_dir = tmp_path / "cp" / "run1"
    _write_run_files(run_dir, [1.0], ["l1", "l2", "l3"])
    run_id = repo.create_run(db, folder="run1", recipe=recipe, base_model="m", dataset="d")
    ingest.ingest_run(db, run_id, run_dir)

    all_logs = ingest.get_logs(db, run_id)
    assert len(all_logs) == 3
    tail = ingest.get_logs(db, run_id, since_seq=2)
    assert [r["line"] for r in tail] == ["l3"]


def test_loss_since_step(db, recipe, tmp_path):
    run_dir = tmp_path / "cp" / "run1"
    _write_run_files(run_dir, [1.0, 2.0, 3.0], [])
    run_id = repo.create_run(db, folder="run1", recipe=recipe, base_model="m", dataset="d")
    ingest.ingest_run(db, run_id, run_dir)

    assert len(ingest.get_loss_series(db, run_id, since_step=1)) == 2


# ── 会话与消息 ───────────────────────────────────────────────────────────
def test_session_and_messages(db):
    sid = repo.create_session(db, model_ref="run1/rwkv-1", name="chat1")
    repo.replace_messages(
        db,
        sid,
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "thinking": "思考中"},
        ],
    )
    msgs = repo.get_messages(db, sid)
    assert len(msgs) == 2
    assert msgs[1]["thinking"] == "思考中"  # thinking 必须独立保存

    listed = repo.list_sessions(db)
    assert listed[0]["message_count"] == 2


def test_replace_messages_is_atomic(db):
    sid = repo.create_session(db, model_ref="m", name="s")
    repo.replace_messages(db, sid, [{"role": "user", "content": "a"}])
    repo.replace_messages(db, sid, [{"role": "user", "content": "b"}])
    msgs = repo.get_messages(db, sid)
    assert len(msgs) == 1
    assert msgs[0]["content"] == "b"


def test_duplicate_session_conflicts(db):
    repo.create_session(db, model_ref="m", name="s")
    with pytest.raises(ConflictError):
        repo.create_session(db, model_ref="m", name="s")


def test_session_delete_cascades_messages(db):
    sid = repo.create_session(db, model_ref="m", name="s")
    repo.replace_messages(db, sid, [{"role": "user", "content": "a"}])
    repo.delete_session(db, sid)
    assert db.conn.execute("SELECT COUNT(*) FROM message").fetchone()[0] == 0


# ── 角色卡 ───────────────────────────────────────────────────────────────
def test_role_upsert_and_list(db):
    repo.upsert_role(db, role_id="r1", name="助手", content="你是助手")
    repo.upsert_role(db, role_id="r1", name="助手2", content="更新")
    roles = repo.list_roles(db)
    assert len(roles) == 1
    assert roles[0]["name"] == "助手2"


def test_delete_missing_role_raises(db):
    with pytest.raises(NotFoundError):
        repo.delete_role(db, "nope")


# ── 并发 ─────────────────────────────────────────────────────────────────
def test_thread_local_connections(tmp_path):
    """多线程各拿各的连接，不互相污染。"""
    database = open_database(tmp_path / "t.db")
    errors: list[Exception] = []

    def worker(n: int):
        try:
            repo.upsert_role(
                database, role_id=f"r{n}", name=f"角色{n}", content="x"
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(repo.list_roles(database)) == 8
    database.close()


def test_transaction_rolls_back(db):
    with pytest.raises(RuntimeError), db.transaction() as tx:
        tx.execute("INSERT INTO role (id, name, content, updated_at) VALUES ('x','x','x','x')")
        raise RuntimeError("boom")
    assert db.conn.execute("SELECT COUNT(*) FROM role").fetchone()[0] == 0
