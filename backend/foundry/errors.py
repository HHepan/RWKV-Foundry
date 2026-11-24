"""Foundry 的异常类型。

设计原则：能明确报错的地方，绝不静默失败。

旧版 llm-finetune-webui 的典型病征是「静默创建空目录，然后训练脚本路径全错」。
这里把配置类错误收敛为单一类型，让启动阶段就能暴露问题，而不是等到训练跑到
一半才因为找不到词表而崩。
"""

from __future__ import annotations


class FoundryError(Exception):
    """本项目的所有自定义异常的基类。"""


class ConfigError(FoundryError):
    """配置无效或环境不完整。

    ``problems`` 保存全部问题而非只抛第一个——用户应该一次看到所有要修的东西，
    而不是修一个重启一次。
    """

    def __init__(self, problems: list[str] | str):
        if isinstance(problems, str):
            problems = [problems]
        self.problems = problems
        super().__init__("\n".join(problems))


class PathTraversalError(FoundryError):
    """用户提供的名称试图越出所属目录。"""


class PeftNotReadyError(ConfigError):
    """RWKV-PEFT 克隆不完整（缺文件或未克隆）。"""


# ── 存储层 ───────────────────────────────────────────────────────────────
class StorageError(FoundryError):
    """数据库操作的基类错误。"""


class NotFoundError(StorageError):
    """请求的记录不存在。"""


class ConflictError(StorageError):
    """违反唯一约束（例如同名 run 已存在）。"""


class MigrationError(StorageError):
    """schema 版本不明或迁移失败。"""
