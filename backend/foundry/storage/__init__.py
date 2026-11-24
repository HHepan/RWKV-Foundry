"""存储层 —— SQLite 元数据与索引。

**本层只存元数据与索引，不存数据本体。**

不存：训练数据 .jsonl、模型权重 .pth（体积大，且 RWKV-PEFT 直接读文件）
只存：训练运行记录、每步 loss、日志、会话、消息、角色卡、数据集行数缓存

导出：
    Database / open_database   连接与迁移
    repo                       数据访问（唯一写 SQL 的地方）
    ingest                     把上游文本产物灌入数据库
"""

from __future__ import annotations

from foundry.storage.db import SCHEMA_VERSION, Database, open_database
from foundry.storage.ingest import (
    IngestResult,
    get_logs,
    get_loss_series,
    ingest_all,
    ingest_run,
)

__all__ = [
    "SCHEMA_VERSION",
    "Database",
    "IngestResult",
    "get_logs",
    "get_loss_series",
    "ingest_all",
    "ingest_run",
    "open_database",
]
