"""端到端验证：真实训练进程的进程组终止。

## 为什么必须单独验证这个

`TrainJob.kill()` 用的是 `os.killpg` + 脚本内 `exec` 的组合。
单元测试里验证的是 fake 脚本（`sh -c 'exec python loop.py'`），
但真实场景是：

    sh train.sh  →  exec python train.py  →  DeepSpeed 初始化 → 可能 fork/vfork

DeepSpeed 启动时会有额外的进程与线程，`exec` 之后的进程组形态可能与
fake 场景不同。**旧项目的 bug 正是"界面显示已停止、GPU 仍在满载"**，
所以这一条必须在真进程上确认。

## 验证方法

1. 启动一个真实的 RWKV-PEFT 训练（用足够多的 epoch_steps 保证跑到一半）
2. 记录初始进程组与成员
3. 调 TrainJob.kill()
4. 断言：进程组内无残留 + nvidia-smi 显存已释放

用法::

    FOUNDRY_HOME=... FOUNDRY_PEFT_DIR=... python scripts/verify_kill.py
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foundry.config import AppConfig
from foundry.domain.recipe import Recipe
from foundry.paths import get_paths
from foundry.train.job import TrainJob
from foundry.train.script import build_script, write_recipe, write_script


def gpu_mem_used() -> list[int]:
    """各卡已用显存（MiB）。无 GPU 时返回空列表。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return [int(x.strip()) for x in out.stdout.strip().splitlines() if x.strip()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def group_alive(pgid: int) -> bool:
    """进程组是否仍有成员。信号 0 只探测不发送。"""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def group_members(pgid: int) -> list[str]:
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,pgid=,cmd="], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return []
    members = []
    for line in out.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        if parts[1] == str(pgid):
            members.append(f"  pid={parts[0]} {parts[2][:80]}")
    return members


def main() -> int:
    paths = get_paths()
    cfg = AppConfig.from_env()
    print(f"[ENV] workspace = {paths.root}")
    print(f"[ENV] peft      = {paths.peft_dir}")
    print(f"[ENV] strategy  = {cfg.infer.strategy}")

    # ── 找到可用的基底权重与 binidx 数据 ────────────────────────────────
    models = sorted(paths.base_models.glob("*.pth"))
    if not models:
        print("[FAIL] base_models 里没有 .pth 权重")
        return 1
    base = models[0]
    print(f"[ENV] base      = {base.name}")

    binidx_root = paths.binidx_data
    candidates = [p for p in binidx_root.glob("*/*.bin")]
    if not candidates:
        print(f"[FAIL] {binidx_root} 下没有可用的 .bin 数据")
        print("       请先跑一次 binidx 预处理")
        return 1
    binidx = candidates[0].with_suffix("")
    print(f"[ENV] binidx    = {binidx}")

    # ── 构造一个会跑足够久的配方 ────────────────────────────────────────
    # epoch_steps 给大一些，确保我们在训练途中 kill，而不是等它自然结束
    size = "2.9B" if "2.9b" in base.name.lower() else "0.1B"
    recipe = Recipe(
        base_model=base.name,
        dataset="out/smoke.jsonl",
        model_size=size,
        ctx_len=64,
        micro_bsz=1,
        epoch_steps=5000,   # 足够大，不会被跑完
        epoch_count=1,
        epoch_save=1,
        lr_init=1e-5,
        lr_final=1e-5,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    run_dir = paths.checkpoints / "killtest"
    script = build_script(
        recipe, paths, run_dir=run_dir, base_model_path=base, binidx_prefix=binidx
    )
    script_path = write_script(script, run_dir)
    write_recipe(recipe, run_dir)
    print(f"[PREP] script    = {script_path}")

    mem_before = gpu_mem_used()
    print(f"[GPU ] 启动前    = {mem_before} MiB")

    job = TrainJob(run_id=1, folder="killtest", recipe=recipe, paths=paths, run_dir=run_dir)
    job.start(script_path)
    print(f"[RUN ] pid={job.pid} pgid={job.pgid}")

    # 等训练真正开始吃显存（DeepSpeed 初始化约需数秒）
    deadline = time.time() + 120
    started = False
    while time.time() < deadline:
        time.sleep(3)
        mem_now = gpu_mem_used()
        if mem_now and any(m > 500 for m in mem_now):
            started = True
            print(f"[GPU ] 训练中    = {mem_now} MiB")
            break
        proc = job._process
        if proc is not None and proc.poll() is not None:
            print(f"[FAIL] 训练进程提前退出 rc={proc.returncode}")
            return 1

    if not started:
        print("[WARN] 未观测到显存增长，但进程仍在跑，继续测终止")

    pgid = job.pgid
    assert pgid is not None
    members_before = group_members(pgid)
    print(f"[PROC] 组内 {len(members_before)} 个成员:")
    for m in members_before:
        print(m)

    # ── 执行三段式终止 ─────────────────────────────────────────────────
    print("\n[KILL] 调用 TrainJob.kill(grace=15) ...")
    t0 = time.time()
    status, note = job.kill(grace=15.0)
    elapsed = time.time() - t0
    print(f"[KILL] status={status.value} 耗时 {elapsed:.1f}s")
    print(f"[KILL] note  = {note}")

    # ── 验证无残留 ─────────────────────────────────────────────────────
    time.sleep(3)
    alive = group_alive(pgid)
    members_after = group_members(pgid)
    mem_after = gpu_mem_used()

    print("\n===== 验证结果 =====")
    ok = True

    if alive or members_after:
        print(f"[FAIL] 进程组 {pgid} 仍有残留: {members_after}")
        ok = False
        # 清理，避免污染机器
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
    else:
        print(f"[PASS] 进程组 {pgid} 已完全消失")

    print(f"[GPU ] 终止后    = {mem_after} MiB")
    if mem_before and mem_after:
        # 若启动前本就占着显存，只要终止后不比之前多太多就算释放了
        leaked = [a - b for a, b in zip(mem_after, mem_before, strict=False)]
        if any(x > 500 for x in leaked):
            print(f"[FAIL] 显存未完全释放，增量为 {leaked} MiB")
            ok = False
        else:
            print(f"[PASS] 显存已释放（增量 {leaked} MiB）")

    print(f"\n{'全部通过' if ok else '存在问题'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
