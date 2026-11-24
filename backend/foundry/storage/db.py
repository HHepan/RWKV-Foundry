"""SQLite 连接管理与 schema 迁移。

设计要点（每一条都是为了避免一类具体故障）：

1. **thread-local 连接**
   FastAPI 的 async 端点运行在事件循环线程，sync 端点在线程池线程。两者若共用
   一个 Connection 会造成 "SQLite objects created in a thread can only be used in
   that same thread" 或更隐蔽的事务互相污染。每线程一条连接最省心。

2. **WAL 模式**
   训练期间 Web 进程要持续写 loss/log（ingester），同时前端要持续读。
   默认的 rollback journal 会让读写互斥，WAL 允许读写并发。

3. **迁移而非 CREATE IF NOT EXISTS**
   直接 ``CREATE TABLE IF NOT EXISTS`` 无法表达「给已有表加一列」。
   用 ``PRAGMA user_version`` 做版本号，逐版本前滚，每个版本可重放。

4. **外键开启**
   SQLite 默认关闭 ``foreign_keys``，必须在每条连接上显式打开，
   否则 REFERENCES 只是装饰。

5. **busy_timeout**
   多进程（Web + 未来可能的 CLI）访问时，写锁冲突会立刻抛
   ``database is locked``。设 5 秒让 SQLite 自行重试。
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from foundry.errors import MigrationError, StorageError

#: 当前 schema 版本。每次新增迁移必须 +1 并在 _MIGRATIONS 中登记。
SCHEMA_VERSION = 1

#: 写锁等待上限（毫秒）
_BUSY_TIMEOUT_MS = 5000


# ── 迁移脚本 ─────────────────────────────────────────────────────────────
#: 索引 = 目标版本号。迁移按版本号升序执行，每个脚本从 (v-1) 推进到 v。
_MIGRATIONS: dict[int, str] = {
    1: """
    -- ===== 配方：一次训练的全部参数，内容寻址 =====
    -- hash 是参数的指纹。同参数训练第二次会自动复用同一 recipe，
    -- 从而可以直接横向对比两次 run 的 loss 曲线。
    CREATE TABLE recipe (
        id          INTEGER PRIMARY KEY,
        hash        TEXT    NOT NULL UNIQUE,
        payload     TEXT    NOT NULL,
        created_at  TEXT    NOT NULL
    );

    -- ===== 训练运行 =====
    -- folder 对应 checkpoints/<folder>，是磁盘产物的唯一标识。
    CREATE TABLE train_run (
        id           INTEGER PRIMARY KEY,
        folder       TEXT    NOT NULL UNIQUE,
        recipe_id    INTEGER REFERENCES recipe(id) ON DELETE SET NULL,
        base_model   TEXT    NOT NULL,
        dataset      TEXT    NOT NULL,
        status       TEXT    NOT NULL DEFAULT 'pending',
        started_at   TEXT,
        finished_at  TEXT,
        exit_code    INTEGER,
        pid          INTEGER,
        error        TEXT,
        CHECK (status IN ('pending','running','completed','stopped','error'))
    );
    CREATE INDEX idx_run_status  ON train_run(status);
    CREATE INDEX idx_run_started ON train_run(started_at DESC);

    -- ===== 每步 loss（ingester 从 loss_data.jsonl 灌入）=====
    -- PK 为 (run_id, step)：重复 tail 同一文件用 INSERT OR REPLACE
    -- 即可幂等，不会产生重复行。这是 ingester 的首要要求。
    CREATE TABLE train_step (
        run_id  INTEGER NOT NULL REFERENCES train_run(id) ON DELETE CASCADE,
        step    INTEGER NOT NULL,
        epoch   INTEGER,
        loss    REAL    NOT NULL,
        lr      REAL,
        kt_s    REAL,
        PRIMARY KEY (run_id, step)
    );

    -- ===== 训练日志（ingester 从 train_log.txt 灌入）=====
    -- seq 单调递增，前端轮询带 ?since=N 只取增量，
    -- 根治旧版「每次拉全量日志」的问题。
    CREATE TABLE train_log (
        run_id  INTEGER NOT NULL REFERENCES train_run(id) ON DELETE CASCADE,
        seq     INTEGER NOT NULL,
        line    TEXT    NOT NULL,
        PRIMARY KEY (run_id, seq)
    );

    -- ===== 数据集索引（派生缓存，mtime 失效）=====
    -- 旧版为了一行行数把整个文件读一遍。这里缓存行数与轮次统计，
    -- mtime/size 变化时才算。
    CREATE TABLE dataset (
        id             INTEGER PRIMARY KEY,
        folder         TEXT    NOT NULL,
        filename       TEXT    NOT NULL,
        mtime          REAL    NOT NULL,
        size           INTEGER NOT NULL,
        line_count     INTEGER NOT NULL,
        rounds_single  INTEGER NOT NULL DEFAULT 0,
        rounds_multi   INTEGER NOT NULL DEFAULT 0,
        indexed_at     TEXT    NOT NULL,
        UNIQUE (folder, filename)
    );

    -- ===== 会话 =====
    -- model_ref 形如 '20260414/checkpoint-1000' 或 'base_models/rwkv7-...'
    -- params 是 JSON 推理参数。旧版把它散在 chat-data/*.json，
    -- 且加载列表要遍历整个 checkpoints 目录树。
    CREATE TABLE session (
        id          INTEGER PRIMARY KEY,
        model_ref   TEXT    NOT NULL,
        name        TEXT    NOT NULL DEFAULT '',
        role_id     TEXT,
        params      TEXT    NOT NULL DEFAULT '{}',
        created_at  TEXT    NOT NULL,
        updated_at  TEXT    NOT NULL,
        UNIQUE (model_ref, name)
    );
    CREATE INDEX idx_session_model ON session(model_ref);

    -- ===== 消息 =====
    -- thinking 单独一列：循环模型的 state 里存着带 think 的 token 序列，
    -- 重建 prompt 时必须还原，因此不能与正文混在一起。
    CREATE TABLE message (
        id          INTEGER PRIMARY KEY,
        session_id  INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
        seq         INTEGER NOT NULL,
        role        TEXT    NOT NULL,
        content     TEXT    NOT NULL,
        thinking    TEXT    NOT NULL DEFAULT '',
        UNIQUE (session_id, seq)
    );
    CREATE INDEX idx_message_session ON message(session_id, seq);

    -- ===== 铸件 =====
    CREATE TABLE checkpoint (
        id          INTEGER PRIMARY KEY,
        run_id      INTEGER REFERENCES train_run(id) ON DELETE SET NULL,
        path        TEXT    NOT NULL UNIQUE,
        epoch       INTEGER,
        size_bytes  INTEGER,
        created_at  TEXT    NOT NULL
    );

    -- ===== 角色卡（旧版散落为 base_models/rwkv_role_*.json）=====
    CREATE TABLE role (
        id          TEXT    PRIMARY KEY,
        name        TEXT    NOT NULL,
        content     TEXT    NOT NULL,
        updated_at  TEXT    NOT NULL
    );
    """,
}


# ── 连接管理 ─────────────────────────────────────────────────────────────
class Database:
    """线程安全的 SQLite 句柄容器。

    ``Database`` 本身可跨线程共享；真正的 ``sqlite3.Connection`` 是
    每线程一份，由 ``_local`` 保管。调用 ``close()`` 只释放当前线程的连接。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._migrated = False

    # -- 连接获取 -------------------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=_BUSY_TIMEOUT_MS / 1000,
            isolation_level=None,  # 自行控制事务，不用隐式 BEGIN
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        """当前线程的连接，惰性创建。"""
        existing = getattr(self._local, "conn", None)
        if existing is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            existing = self._connect()
            self._local.conn = existing
        return existing

    # -- 事务 ------------------------------------------------------------ #
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """显式事务。异常时回滚。

        使用 ``BEGIN IMMEDIATE`` 而非默认的 deferred：立即取写锁，
        避免「读时无锁、写时才发现冲突」的升级失败。
        """
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # -- 迁移 ------------------------------------------------------------ #
    def migrate(self) -> int:
        """把 schema 前滚到 SCHEMA_VERSION，返回最终版本号。

        幂等：已是最新版本时不做任何事。
        """
        with self._lock:
            conn = self.conn
            current = conn.execute("PRAGMA user_version").fetchone()[0]

            if current > SCHEMA_VERSION:
                raise MigrationError(
                    f"数据库 schema 版本 {current} 高于本程序支持的 {SCHEMA_VERSION}。\n"
                    f"  这个库是被更新版本的 Foundry 创建的，请升级程序而不是降级数据。"
                )

            if current == SCHEMA_VERSION:
                self._migrated = True
                return current

            for version in range(current + 1, SCHEMA_VERSION + 1):
                script = _MIGRATIONS.get(version)
                if script is None:
                    raise MigrationError(
                        f"缺少到版本 {version} 的迁移脚本。"
                        f"SCHEMA_VERSION={SCHEMA_VERSION} 但 _MIGRATIONS 中没有对应条目。"
                    )
                try:
                    conn.executescript(script)
                    # executescript 会隐式提交，因此 user_version 需单独设置
                    conn.execute(f"PRAGMA user_version = {version}")
                except sqlite3.Error as exc:
                    raise MigrationError(
                        f"迁移到版本 {version} 失败: {exc}\n  库文件: {self.path}"
                    ) from exc

            self._migrated = True
            return SCHEMA_VERSION

    def schema_version(self) -> int:
        return self.conn.execute("PRAGMA user_version").fetchone()[0]

    # -- 关闭 ------------------------------------------------------------ #
    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


def open_database(path: str | Path, *, migrate: bool = True) -> Database:
    """打开（必要时创建）数据库并完成迁移。"""
    db = Database(path)
    if migrate:
        try:
            db.migrate()
        except (MigrationError, StorageError):
            db.close()
            raise
        except sqlite3.Error as exc:
            db.close()
            raise StorageError(f"无法打开数据库 {path}: {exc}") from exc
    return db
