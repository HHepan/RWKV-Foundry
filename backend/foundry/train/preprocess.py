"""binidx 预处理 —— 把 .jsonl 转成 RWKV-PEFT 可训练的 .bin/.idx。

## 旧版的问题

``train_service.py:26-30`` 的做法是**清空整个 json2binidx_tool/data 目录**::

    for f in os.listdir(JSON2BINIDX_DIR):
        os.remove(os.path.join(JSON2BINIDX_DIR, f))

后果：任何并发 run 的中间产物会被另一个 run 抹掉；用户放在那里的文件也会消失。

## 本版的做法

每个 run 用**独立子目录** ``<binidx>/<run_folder>/``，互不干扰。
清理只作用于自己的子目录。

## 为什么保留在 RWKV-PEFT 目录下

``preprocess_data.py`` 依赖同目录的 ``rwkv_vocab_v20230424.txt`` 与相对导入，
产物放在其 ``json2binidx_tool/data/`` 下最稳妥（上游脚本默认行为）。
我们只做"子目录隔离"，不去改动上游布局。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from foundry.errors import FoundryError


class BinidxError(FoundryError):
    """binidx 转换失败。"""


def run_preprocess(
    *,
    peft_dir: Path,
    python_exe: str,
    src_jsonl: Path,
    run_folder: str,
    vocab: Path,
    append_eod: bool = True,
    timeout: float = 3600.0,
) -> Path:
    """执行 preprocess_data.py，返回 binidx 路径前缀（不含扩展名）。

    :param run_folder: 用于隔离的 run 标识（会成为子目录名）
    :returns: ``<binidx>/<run_folder>/<stem>`` —— .bin/.idx 的共同前缀
    """
    data_root = peft_dir / "json2binidx_tool" / "data"
    work_dir = data_root / run_folder

    # 只清理自己的工作目录
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 把数据复制进来（上游脚本按 --input 读取，独立副本避免被外部修改影响）
    local_jsonl = work_dir / src_jsonl.name
    shutil.copy2(src_jsonl, local_jsonl)

    stem = Path(src_jsonl.name).stem
    output_prefix = work_dir / stem

    cmd = [
        python_exe,
        str(peft_dir / "json2binidx_tool" / "tools" / "preprocess_data.py"),
        "--input", str(local_jsonl),
        "--output-prefix", str(output_prefix),
        "--vocab", str(vocab),
        "--dataset-impl", "mmap",
        "--tokenizer-type", "RWKVTokenizer",
    ]
    if append_eod:
        cmd.append("--append-eod")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(peft_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise BinidxError(
            f"数据预处理超时（{timeout}s）。数据集过大或 tokenizer 卡住。\n"
            f"  input: {local_jsonl}"
        ) from exc
    except OSError as exc:
        raise BinidxError(f"无法启动预处理进程: {exc}") from exc

    if result.returncode != 0:
        raise BinidxError(
            f"数据预处理失败（退出码 {result.returncode}）\n"
            f"  命令: {' '.join(cmd)}\n"
            f"  stderr: {(result.stderr or '').strip()[:2000]}"
        )

    # 上游产出 <stem>_text_document.bin/.idx，RWKV-PEFT 期望 <stem>.bin/.idx
    produced_bin = work_dir / f"{stem}_text_document.bin"
    produced_idx = work_dir / f"{stem}_text_document.idx"
    target_bin = work_dir / f"{stem}.bin"
    target_idx = work_dir / f"{stem}.idx"

    if not produced_bin.exists():
        raise BinidxError(
            f"预处理声称成功，但未生成 {produced_bin.name}\n"
            f"  目录内容: {[p.name for p in work_dir.iterdir()]}"
        )

    os.replace(produced_bin, target_bin)
    if produced_idx.exists():
        os.replace(produced_idx, target_idx)

    return output_prefix


def default_python_exe() -> str:
    """使用**当前解释器**，而不是硬编码 ``python3``。

    旧版 ``train_service.py`` 写死 ``python3``：在 conda 环境下会调用系统
    Python，导致"明明装了 torch 却 import 不到"。这是最阴的一类故障，
    因为它不报错，只是行为诡异。
    """
    return sys.executable
