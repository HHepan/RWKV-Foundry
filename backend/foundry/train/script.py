"""训练脚本生成 —— 把 Recipe 变成可执行的 shell 脚本。

## 为什么写进 run 自己的目录

旧版把 ``lora_temp.sh`` 写进 ``RWKV-PEFT/scripts/``，这是**往第三方仓库里写文件**：

1. 污染上游代码目录，git status 变脏，升级 RWKV-PEFT 时冲突
2. 两个 run 并发会互相覆盖同一个 ``lora_temp.sh``
3. 配方与产出分离，事后无法知道"这次的权重是用什么脚本训的"

现在每个 run 的全部材料自包含在一个目录里::

    checkpoints/<run>/
        recipe.json     配方（只读，可作为复现依据）
        train.sh        本次生成的训练脚本
        loss_data.jsonl trainer 输出（append）
        train_log.txt   trainer 日志（append）
        rwkv-N.pth      权重产物

搬走这个目录即为完整的一份训练档案。

## 路径引用

脚本里一律使用**绝对路径**并用单引号包裹。旧版用双引号，路径含 ``$``、
反引号或空格时会静默展开或断裂。
"""

from __future__ import annotations

import shlex
from pathlib import Path

from foundry.domain.recipe import Recipe
from foundry.paths import WorkspacePaths

SCRIPT_NAME = "train.sh"
RECIPE_NAME = "recipe.json"


def _q(value: str | Path) -> str:
    """安全引用：单引号包裹，内部单引号转义。

    ``shlex.quote`` 处理 ``$``、反引号、空格、引号等全部情况。
    """
    return shlex.quote(str(value))


def resolve_dataset_path(paths: WorkspacePaths, dataset: str) -> Path:
    """把 ``Recipe.dataset``（形如 ``"out/train.jsonl"``）解析为绝对路径。

    ``dataset`` 允许带一层子目录（对应旧版 ``data/out/`` 的习惯）。
    拒绝绝对路径与 ``..``——数据集必须在 workspace/data 之下。
    """
    from foundry.errors import PathTraversalError
    from foundry.paths import safe_join

    if not dataset.strip():
        raise PathTraversalError("dataset 不能为空")
    if Path(dataset).is_absolute():
        raise PathTraversalError(f"dataset 必须是相对 workspace/data 的路径: {dataset!r}")

    parts = [p for p in Path(dataset).parts if p not in ("", ".")]
    if not parts:
        raise PathTraversalError(f"dataset 路径无效: {dataset!r}")

    return safe_join(paths.data, *parts, kind="数据集路径")


def build_script(
    recipe: Recipe,
    paths: WorkspacePaths,
    *,
    run_dir: Path,
    base_model_path: Path,
    binidx_prefix: Path,
) -> str:
    """生成训练脚本内容（不落盘）。

    :param run_dir: 本次 run 的目录（proj_dir）
    :param base_model_path: 基底权重绝对路径（不含 .pth 后缀，RWKV 自行追加）
    :param binidx_prefix: 预处理后 binidx 的路径前缀（不含扩展名）
    """
    recipe.validate()

    peft_dir = paths.peft_dir
    args = [
        "python",
        "train.py",
        "--load_model", _q(base_model_path),
        "--proj_dir", _q(run_dir),
        "--data_file", _q(binidx_prefix),
        "--vocab_size", str(65536),
        "--data_type", "binidx",
        "--n_layer", str(recipe.n_layer),
        "--n_embd", str(recipe.n_embd),
        "--ctx_len", str(recipe.ctx_len),
        "--micro_bsz", str(recipe.micro_bsz),
        "--epoch_steps", str(recipe.epoch_steps),
        "--epoch_count", str(recipe.epoch_count),
        "--epoch_save", str(recipe.epoch_save),
        "--lr_init", repr(recipe.lr_init),
        "--lr_final", repr(recipe.lr_final),
        "--accelerator", "gpu",
        "--precision", recipe.precision,
        "--devices", "1",
        "--strategy", "deepspeed_stage_1",
        "--grad_cp", "1",
        "--my_testing", recipe.my_testing,
        "--peft", "lora",
        "--peft_config", _q(recipe.peft_config_json),
    ]

    lines = [
        "#!/bin/sh",
        "# 由 RWKV-Foundry 生成。请勿手工编辑——改配方后重新生成。",
        f"# 配方 hash: {recipe.hash}",
        f"# 基底模型 : {base_model_path}",
        "",
        "# 切换到 RWKV-PEFT 目录（其内部按相对路径查找模块）",
        f"cd {_q(peft_dir)} || exit 1",
        "",
        "# exec 让 python 取代当前 shell 进程：",
        "#   否则 kill 掉 sh 之后 python 会成为孤儿继续占着 GPU。",
        "#   这一点已在旧版上实测复现过。",
        "exec " + " ".join(args),
        "",
    ]
    return "\n".join(lines)


def write_script(content: str, run_dir: Path) -> Path:
    """把脚本写入 run 目录并赋予执行权限。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / SCRIPT_NAME
    target.write_text(content, encoding="utf-8")
    target.chmod(0o755)
    return target


def write_recipe(recipe: Recipe, run_dir: Path) -> Path:
    """把配方快照写入 run 目录，作为可复现依据。"""
    import json

    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / RECIPE_NAME
    target.write_text(
        json.dumps(recipe.to_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
