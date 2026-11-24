"""训练产物 ingester —— 把上游写出的文本文件灌入数据库。

## 为什么需要这一层

上游 RWKV-PEFT 的 ``TrainCallback`` **已经**会 append 写两个文件：

    <proj_dir>/loss_data.jsonl   每步 {"loss", "t_cost", "kt_s"}
    <proj_dir>/train_log.txt     训练日志

旧版 llm-finetune-webui 的病是：**Web 进程又自己写了一份**，而且用的是全量覆盖。

    Web 写:    loss_history.jsonl  (全量重写, {"step","loss","epoch"})
    trainer 写: loss_data.jsonl    (append,      {"loss","t_cost","kt_s"})

结果是同一份数据两个文件名、两套 schema、两个写入者，且前端同时拉两份画同一张图。
更糟的是全量重写在训练期是 O(n²)：实测 4000 行日志放大 **400x**，
外推 2 小时训练（约 5 万行）要浪费 **342 秒**纯 IO。

## 本模块的原则

1. **单写入者**。trainer 只写文件，Web 只在 ingester 里写 DB。
2. **幂等**。``INSERT OR REPLACE`` + 主键 ``(run_id, step)`` / ``(run_id, seq)``，
   重启后重新 tail 不产生重复行。
3. **增量**。记住已消费的行数，下次从那里继续，不重复解析整个文件。
4. **不碰训练进程**。ingester 只读文件，绝不干预子进程。

## 为什么不让训练进程直接写 SQLite

训练是上游代码，我们不能改；而且 SQLite 多进程写要处理锁与 WAL 竞争。
让 trainer 老实 append 文本、Web 来灌，是正确且最小侵入的边界。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from foundry.storage.db import Database

#: 上游 trainer 写出的文件名（固定，来自 RWKV-PEFT 的 TrainCallback）
LOSS_FILE = "loss_data.jsonl"
LOG_FILE = "train_log.txt"


@dataclass
class IngestResult:
    """一次 ingester 的产出统计。"""

    steps_added: int = 0
    logs_added: int = 0
    lines_scanned: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.steps_added > 0 or self.logs_added > 0

    def as_dict(self) -> dict:
        return {
            "steps_added": self.steps_added,
            "logs_added": self.logs_added,
            "lines_scanned": self.lines_scanned,
            "errors": self.errors,
        }


def _read_jsonl_from(path: Path, offset: int) -> tuple[list[dict], int]:
    """从第 ``offset`` 行开始读 JSONL，返回 (记录列表, 新的行数)。

    坏行被记录并跳过，不中断整批导入——训练跑了 3 小时的曲线不该因为
    某一行截断而全部丢失。
    """
    records: list[dict] = []
    seen = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                seen += 1
                if seen <= offset:
                    continue
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    # 训练进程可能正在写这一行，下次再读
                    break
                if isinstance(obj, dict) and "loss" in obj:
                    records.append(obj)
    except FileNotFoundError:
        return [], offset
    return records, seen


def _read_lines_from(path: Path, offset: int) -> tuple[list[str], int]:
    """从第 ``offset`` 行开始读文本，返回 (行列表, 新的行数)。"""
    lines: list[str] = []
    seen = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                seen += 1
                if seen <= offset:
                    continue
                lines.append(raw.rstrip("\n"))
    except FileNotFoundError:
        return [], offset
    return lines, seen


def _ingested_counts(db: Database, run_id: int) -> tuple[int, int]:
    """已入库的 step 数 / log 行数，用作增量 offset。"""
    conn = db.conn
    steps = conn.execute(
        "SELECT COALESCE(MAX(step), 0) FROM train_step WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    logs = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) FROM train_log WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    return int(steps), int(logs)


def ingest_run(db: Database, run_id: int, run_dir: Path) -> IngestResult:
    """把 ``run_dir`` 下的上游产物增量灌入数据库。

    幂等且可重复调用：只消费尚未入库的部分。
    """
    result = IngestResult()

    # ── loss ────────────────────────────────────────────────────────────
    loss_path = run_dir / LOSS_FILE
    if loss_path.exists():
        done_steps, _ = _ingested_counts(db, run_id)
        records, scanned = _read_jsonl_from(loss_path, done_steps)
        result.lines_scanned += scanned

        if records:
            with db.transaction() as conn:
                for offset, rec in enumerate(records, 1):
                    step = done_steps + offset
                    try:
                        loss = float(rec["loss"])
                    except (TypeError, ValueError):
                        result.errors.append(f"step {step}: loss 不是数值，已跳过")
                        continue
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO train_step
                            (run_id, step, epoch, loss, lr, kt_s)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            step,
                            rec.get("epoch"),
                            loss,
                            rec.get("lr"),
                            rec.get("kt_s"),
                        ),
                    )
                    result.steps_added += 1

    # ── 日志 ────────────────────────────────────────────────────────────
    log_path = run_dir / LOG_FILE
    if log_path.exists():
        _, done_logs = _ingested_counts(db, run_id)
        lines, scanned = _read_lines_from(log_path, done_logs)
        result.lines_scanned += scanned

        if lines:
            with db.transaction() as conn:
                for offset, line in enumerate(lines, 1):
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO train_log (run_id, seq, line)
                        VALUES (?, ?, ?)
                        """,
                        (run_id, done_logs + offset, line),
                    )
                    result.logs_added += 1

    # 刻意不修改 finished_at。
    #
    # 时间戳属于 run 的状态机，由 repo.update_run_status() 在状态转入
    # completed/stopped/error 时设置。ingester 只负责数据，如果它顺手写
    # finished_at，正在运行的训练会在第一次灌入时就带上"结束时间"——
    # 而这个字段一旦被 COALESCE 固定，真正结束时也改不回来了。
    return result


def ingest_all(db: Database, checkpoints_dir: Path) -> dict[str, IngestResult]:
    """扫描 checkpoints 下所有 run 并逐个灌入。

    返回 ``{folder: IngestResult}``，失败项记在 errors 里而不中断其他 run。
    """
    out: dict[str, IngestResult] = {}
    if not checkpoints_dir.is_dir():
        return out

    conn = db.conn
    rows = conn.execute("SELECT id, folder FROM train_run").fetchall()
    by_folder = {r["folder"]: r["id"] for r in rows}

    for entry in sorted(checkpoints_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        run_id = by_folder.get(entry.name)
        if run_id is None:
            continue
        try:
            out[entry.name] = ingest_run(db, run_id, entry)
        except (sqlite3.Error, OSError) as exc:
            failed = IngestResult()
            failed.errors.append(f"{entry.name}: {exc}")
            out[entry.name] = failed
    return out


# ── 查询 ─────────────────────────────────────────────────────────────────
def get_loss_series(
    db: Database, run_id: int, *, since_step: int = 0, limit: int | None = None
) -> list[dict]:
    """取 loss 曲线。``since_step`` 支持增量拉取。"""
    sql = (
        "SELECT step, epoch, loss, lr, kt_s FROM train_step "
        "WHERE run_id = ? AND step > ? ORDER BY step"
    )
    params: list = [run_id, since_step]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [dict(r) for r in db.conn.execute(sql, params)]


def get_logs(
    db: Database, run_id: int, *, since_seq: int = 0, limit: int = 1000
) -> list[dict]:
    """取日志。``since_seq`` 让前端只拉增量，不再每次拉全量。"""
    rows = db.conn.execute(
        "SELECT seq, line FROM train_log WHERE run_id = ? AND seq > ? "
        "ORDER BY seq LIMIT ?",
        (run_id, since_seq, limit),
    )
    return [dict(r) for r in rows]
