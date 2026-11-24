#!/usr/bin/env python
"""环境自检 —— 在新机器上部署前先跑这个。

## 为什么需要它

在当前机器上已经踩过的坑，全都在这个脚本里被显式检查：

1. RWKV-PEFT 未克隆 / 克隆不完整
2. 词表文件缺失（会导致 token id 错位且不报错）
3. torch 装成了 CPU 版（不报错，只是"训练极慢"）
4. 驱动缺失但 torch 以为能用
5. CUDA 算力与 precision 不匹配（fp16 被上游内核拒绝）
6. 磁盘空间不足（装完 CUDA torch 要 5~8GB）
7. 显存不足以容纳目标模型

用法::

    python scripts/doctor.py
    FOUNDRY_HOME=/data/foundry python scripts/doctor.py
    python scripts/doctor.py --model-size 0.4B
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 允许从 backend/ 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: 各尺寸权重显存估算（bf16 = 2 bytes/param）
#: state 是 O(1)，与 ctx_len 无关 —— 这是 RWKV 作为 RNN 的核心优势
MODEL_PARAMS = {
    "0.1B": 0.19e9,
    "0.4B": 0.45e9,
    "1.5B": 1.5e9,
    "2.9B": 2.9e9,
    "7B": 7.2e9,
    "13.3B": 13.3e9,
}

OK = "✅"
WARN = "⚠️ "
FAIL = "❌"

results: list[tuple[str, str, str]] = []


def record(level: str, title: str, detail: str = "") -> None:
    results.append((level, title, detail))
    print(f"{level} {title}")
    if detail:
        for line in detail.splitlines():
            print(f"    {line}")


def check_python() -> None:
    v = sys.version_info
    if v >= (3, 10):
        record(OK, f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        record(FAIL, f"Python 版本过低: {v.major}.{v.minor}", "需要 >= 3.10")


def check_workspace() -> Path | None:
    try:
        from foundry.paths import get_paths
    except ImportError:
        record(FAIL, "无法导入 foundry", "请先 pip install -e backend/")
        return None

    paths = get_paths()
    print(f"\n  workspace = {paths.root}")
    print(f"  RWKV-PEFT = {paths.peft_dir}\n")

    problems = paths.problems()
    if problems:
        for p in problems:
            record(FAIL, p.splitlines()[0], "\n".join(p.splitlines()[1:]))
        return None

    record(OK, "workspace 结构完整")

    # 词表单独确认 —— 它是静默故障的头号来源
    if paths.vocab.is_file():
        size = paths.vocab.stat().st_size
        record(OK, f"词表存在 ({size / 1024:.0f} KB)")
    else:
        record(FAIL, "词表缺失", f"{paths.vocab}\n会导致 token id 错位且不报错，必须修复")

    n_models = len(list(paths.base_models.glob("*.pth")))
    if n_models:
        record(OK, f"base_models 中有 {n_models} 个 .pth")
    else:
        record(WARN, "base_models 中没有 .pth 权重", f"请放入: {paths.base_models}")

    return paths


def check_torch() -> None:
    try:
        import torch
    except ImportError:
        record(WARN, "torch 未安装", '仅 Web/测试无需 torch；要训练请 pip install -e ".[train]"')
        return

    record(OK, f"torch {torch.__version__}")

    if not torch.cuda.is_available():
        record(
            FAIL,
            "torch.cuda.is_available() = False",
            "常见原因：\n"
            "  1. 装成了 CPU 版 wheel —— 用 --index-url .../cu121 重装\n"
            "  2. NVIDIA 驱动未安装 —— sudo ubuntu-drivers install && reboot\n"
            "  3. 驱动版本低于 torch 要求的 CUDA 版本",
        )
        # 检查系统层面是否有驱动，帮助定位是第 1 还是第 2 种
        if shutil.which("nvidia-smi"):
            record(OK, "nvidia-smi 可用 → 驱动正常，问题在 torch wheel")
        else:
            record(FAIL, "nvidia-smi 不存在", "驱动未安装，这是 cuda 不可用的根因")
        return

    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        record(
            OK,
            f"GPU{i}: {p.name}",
            f"显存 {p.total_memory / 1024**3:.2f} GB, 算力 cc{p.major}.{p.minor}",
        )
        if p.major < 7:
            record(WARN, f"GPU{i} 算力 cc{p.major}.{p.minor} 偏低", "bf16 可能不被支持")


def check_vram(model_size: str | None) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        total = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        return

    if model_size is None:
        return

    params = MODEL_PARAMS.get(model_size)
    if params is None:
        record(WARN, f"未知 model_size: {model_size}")
        return

    overhead = 0.7 * 1024**3  # CUDA context + 分配器碎片
    weight = params * 2       # bf16
    need = weight + overhead
    usable = total

    print()
    if need <= usable:
        mem = (need / 1024**3, usable / 1024**3)
        record(OK, f"{model_size} 推理显存 %.2f GB / 可用 %.2f GB (%.0f%%)" % mem)
    else:
        record(
            FAIL,
            f"{model_size} 推理需要 {need / 1024**3:.2f} GB，超出 {usable / 1024**3:.2f} GB",
            "改用更小的模型，或使用 bitsandbytes 做 nf4 量化",
        )


def check_disk() -> None:
    paths_root = Path(os.environ.get("FOUNDRY_HOME", ".")).resolve()
    target = paths_root if paths_root.exists() else Path.cwd()
    usage = shutil.disk_usage(target)
    free_gb = usage.free / 1024**3

    if free_gb >= 20:
        record(OK, f"磁盘剩余 {free_gb:.1f} GB")
    elif free_gb >= 8:
        record(WARN, f"磁盘剩余 {free_gb:.1f} GB", "装 CUDA torch 需 5~8GB，会比较紧")
    else:
        record(
            FAIL,
            f"磁盘剩余仅 {free_gb:.1f} GB",
            "CUDA torch 需 5~8GB，且训练产物还要更多空间",
        )


def check_port(port: int = 8000) -> None:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            record(WARN, f"端口 {port} 已被占用", "启动后端时换 --port 或先停掉占用进程")
        else:
            record(OK, f"端口 {port} 可用")


def check_node() -> None:
    if shutil.which("node") is None:
        record(WARN, "未安装 Node.js", "前端需要 Node 18+")
        return
    try:
        out = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        record(OK, f"Node {out}")
    except (OSError, subprocess.SubprocessError):
        record(WARN, "无法确定 Node 版本")

    if os.environ.get("NODE_ENV") == "production":
        record(
            WARN,
            "NODE_ENV=production",
            "npm install 会跳过 devDependencies，导致 vite 缺失。\n"
            "  用: NODE_ENV=development npm install --include=dev",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="RWKV-Foundry 环境自检")
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_PARAMS),
        help="检查显存能否放下该尺寸模型",
    )
    args = parser.parse_args()

    print("=" * 68)
    print("RWKV-Foundry 环境自检")
    print("=" * 68)

    check_python()
    check_workspace()
    check_torch()
    check_vram(args.model_size)
    check_disk()
    check_port()
    check_node()

    print()
    print("=" * 68)
    n_fail = sum(1 for lv, _, _ in results if lv == FAIL)
    n_warn = sum(1 for lv, _, _ in results if lv == WARN)
    if n_fail:
        print(f"{FAIL} {n_fail} 项必须修复，{n_warn} 项警告")
        print("=" * 68)
        return 1
    if n_warn:
        print(f"{WARN} 无阻塞问题，{n_warn} 项警告")
    else:
        print(f"{OK} 全部通过")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
