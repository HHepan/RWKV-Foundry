"""训练编排层。

    script.py      把 Recipe 变成可执行 shell 脚本
    preprocess.py  .jsonl → binidx（每 run 独立子目录）
    job.py         子进程生命周期：启动 / 顺出 / 三段式终止 / 状态机

边界原则：**本层不碰数据库，也不碰 HTTP。**
状态由调用方（API）通过 ``repo.update_run_status`` 落库。
"""

from __future__ import annotations

from foundry.train.job import (
    DEFAULT_GRACE_SECONDS,
    JobStatus,
    TrainError,
    TrainJob,
)
from foundry.train.preprocess import BinidxError, default_python_exe, run_preprocess
from foundry.train.script import (
    RECIPE_NAME,
    SCRIPT_NAME,
    build_script,
    resolve_dataset_path,
    write_recipe,
    write_script,
)

__all__ = [
    "DEFAULT_GRACE_SECONDS",
    "JobStatus",
    "TrainError",
    "TrainJob",
    "BinidxError",
    "default_python_exe",
    "run_preprocess",
    "RECIPE_NAME",
    "SCRIPT_NAME",
    "build_script",
    "resolve_dataset_path",
    "write_recipe",
    "write_script",
]
