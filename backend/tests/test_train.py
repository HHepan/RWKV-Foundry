"""foundry.train 的测试。

核心验证三件在旧项目里真实出问题的事：

  1. **只杀 sh 会让 python 成为孤儿**（已实测复现）→ 必须整组终止
  2. **stop_event.set() 后立刻 clear() 会丢信号** → 单向状态，永不回退
  3. **save_pid 存错 pid** → 记录真实子进程 pid 与 pgid
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from foundry.domain.recipe import Recipe
from foundry.errors import PathTraversalError
from foundry.paths import resolve_paths
from foundry.train import job as job_mod
from foundry.train.job import JobStatus, TrainError, TrainJob
from foundry.train.preprocess import BinidxError, default_python_exe, run_preprocess
from foundry.train.script import (
    SCRIPT_NAME,
    build_script,
    resolve_dataset_path,
    write_recipe,
    write_script,
)


@pytest.fixture
def paths(tmp_path):
    peft = tmp_path / "workspace" / "tools" / "RWKV-PEFT"
    for rel in (
        "train.py",
        "json2binidx_tool/tools/preprocess_data.py",
        "json2binidx_tool/rwkv_vocab_v20230424.txt",
        "scripts/lora.sh",
    ):
        target = peft / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    (tmp_path / "workspace" / "data").mkdir(parents=True)
    (tmp_path / "workspace" / "checkpoints").mkdir(parents=True)
    (tmp_path / "workspace" / "base_models").mkdir(parents=True)
    os.environ["FOUNDRY_HOME"] = str(tmp_path / "workspace")
    yield resolve_paths()
    os.environ.pop("FOUNDRY_HOME", None)


@pytest.fixture
def recipe():
    return Recipe(
        base_model="rwkv7-g1a-0.1b.pth",
        dataset="out/train.jsonl",
        model_size="0.1B",
        ctx_len=512,
        micro_bsz=1,
        epoch_steps=10,
        epoch_count=2,
        lora_r=32,
    )


# ── 脚本生成 ─────────────────────────────────────────────────────────────
def test_build_script_contains_required_args(recipe, paths, tmp_path):
    content = build_script(
        recipe,
        paths,
        run_dir=tmp_path / "run",
        base_model_path=tmp_path / "base.pth",
        binidx_prefix=tmp_path / "out/train",
    )
    assert "--load_model" in content
    assert "--peft lora" in content
    assert "--my_testing x070" in content
    assert "--vocab_size 65536" in content
    assert f"--n_layer {recipe.n_layer}" in content


def test_build_script_uses_exec(recipe, paths, tmp_path):
    """exec 让 python 取代 sh —— 否则 kill sh 后 python 成孤儿。"""
    content = build_script(
        recipe, paths, run_dir=tmp_path, base_model_path=tmp_path / "b",
        binidx_prefix=tmp_path / "t",
    )
    assert "exec python" in content


def test_build_script_quotes_paths(recipe, paths, tmp_path):
    """含空格与 $ 的路径必须被单引号包裹，避免 shell 展开。"""
    weird = tmp_path / "a b $x"
    content = build_script(
        recipe, paths, run_dir=weird, base_model_path=weird / "m",
        binidx_prefix=weird / "d",
    )
    base_arg = content[content.index("--load_model"):]
    base_arg = base_arg[: base_arg.index("\n")]
    # 路径必须以单引号开始（而非裸写），否则 $x 会被展开成空字符串
    assert "'" in base_arg
    assert f"'{weird}/m'" in content
    assert f"'{weird}'" in content


def test_build_script_rejects_invalid_recipe(recipe, paths, tmp_path):
    bad = Recipe(**{**recipe.__dict__, "lora_dropout": 1.5})
    from foundry.domain.recipe import RecipeError

    with pytest.raises(RecipeError):
        build_script(
            bad, paths, run_dir=tmp_path, base_model_path=tmp_path / "b",
            binidx_prefix=tmp_path / "t",
        )


def test_write_script_self_contained(recipe, paths, tmp_path):
    """脚本写进 run 目录，绝不写进 RWKV-PEFT。

    ## 断言方式说明（这里踩过坑）

    最初写的是 `assert not (paths.peft_scripts / "lora_temp.sh").exists()`，
    在开发机上通过，但换到实际训练机上失败——因为那个文件是旧项目
    2026-05 留下的真实产物。这个断言依赖了外部环境状态，是坏测试。

    正确做法：记录目标目录**前后**的文件集合，断言本次操作没有新增文件。
    这样无论环境里已有什么，都能准确验证「没有再往 PEFT 写东西」。
    """
    run_dir = tmp_path / "run"

    # 记录 PEFT 目录在操作前的快照
    before = set()
    if paths.peft_scripts.is_dir():
        before = {p.name for p in paths.peft_scripts.iterdir()}

    content = build_script(
        recipe, paths, run_dir=run_dir, base_model_path=tmp_path / "b",
        binidx_prefix=tmp_path / "t",
    )
    target = write_script(content, run_dir)

    assert target == run_dir / SCRIPT_NAME
    assert target.exists()
    assert os.access(target, os.X_OK)

    # 操作后不应有任何新增文件
    after = set()
    if paths.peft_scripts.is_dir():
        after = {p.name for p in paths.peft_scripts.iterdir()}
    assert after == before, f"不该往 RWKV-PEFT 写文件，却新增了: {after - before}"

    # 脚本内容里也不应引用 lora_temp.sh
    assert "lora_temp.sh" not in content


def test_write_recipe_snapshot(recipe, tmp_path):
    import json

    p = write_recipe(recipe, tmp_path / "run")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["_hash"] == recipe.hash
    assert data["lora_r"] == recipe.lora_r


# ── 数据集路径解析 ───────────────────────────────────────────────────────
def test_resolve_dataset_path(paths):
    p = resolve_dataset_path(paths, "out/train.jsonl")
    assert p == paths.data / "out" / "train.jsonl"


def test_resolve_dataset_rejects_absolute(paths):
    with pytest.raises(PathTraversalError, match="必须是相对"):
        resolve_dataset_path(paths, "/etc/passwd")


def test_resolve_dataset_rejects_traversal(paths):
    with pytest.raises(PathTraversalError):
        resolve_dataset_path(paths, "../../etc/passwd")


def test_resolve_dataset_rejects_empty(paths):
    with pytest.raises(PathTraversalError, match="不能为空"):
        resolve_dataset_path(paths, "   ")


# ── 预处理 ───────────────────────────────────────────────────────────────
def test_default_python_uses_current_interpreter():
    """旧版硬编码 python3，在 conda 下会调错解释器。"""
    import sys

    assert default_python_exe() == sys.executable


def test_run_preprocess_isolates_run_dir(paths, tmp_path):
    """每 run 独立子目录，不清理整个 data 目录。"""
    src = paths.data / "train.jsonl"
    src.write_text('{"text":"User: a"}\n', encoding="utf-8")

    # 预置一个"别人的"文件，验证不会被删
    other_dir = paths.binidx_data / "other_run"
    other_dir.mkdir(parents=True)
    keeper = other_dir / "keep.bin"
    keeper.write_text("x", encoding="utf-8")

    fake_py = tmp_path / "fake_preprocess.py"
    fake_py.write_text(
        "import argparse, pathlib\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--input'); p.add_argument('--output-prefix')\n"
        "p.add_argument('--vocab'); p.add_argument('--dataset-impl')\n"
        "p.add_argument('--tokenizer-type'); p.add_argument('--append-eod',action='store_true')\n"
        "a=p.parse_args()\n"
        "pre=pathlib.Path(a.output_prefix)\n"
        "pre.with_name(pre.name+'_text_document.bin').write_text('bin')\n"
        "pre.with_name(pre.name+'_text_document.idx').write_text('idx')\n",
        encoding="utf-8",
    )
    # 覆盖上游脚本路径
    target = paths.preprocess_script
    target.write_text(fake_py.read_text(encoding="utf-8"), encoding="utf-8")

    prefix = run_preprocess(
        peft_dir=paths.peft_dir,
        python_exe=default_python_exe(),
        src_jsonl=src,
        run_folder="myrun",
        vocab=paths.vocab,
    )
    assert prefix.name == "train"
    assert (prefix.parent / "train.bin").exists()
    assert (prefix.parent / "train.idx").exists()
    # 别人的文件必须还在
    assert keeper.exists(), "不得清理整个 binidx data 目录"


def test_run_preprocess_reports_failure(paths, tmp_path):
    src = paths.data / "x.jsonl"
    src.write_text('{"text":"a"}\n', encoding="utf-8")
    paths.preprocess_script.write_text("import sys; sys.exit(3)", encoding="utf-8")

    with pytest.raises(BinidxError, match="退出码 3"):
        run_preprocess(
            peft_dir=paths.peft_dir, python_exe=default_python_exe(),
            src_jsonl=src, run_folder="r", vocab=paths.vocab,
        )


# ── 子进程生命周期 ───────────────────────────────────────────────────────
def _write_fake_train(tmp_path: Path, seconds: float = 30.0) -> Path:
    """写一个会一直跑的训练脚本，内含 python 子进程。

    sh 里用 exec，模拟我们生成的真实脚本。
    """
    script = tmp_path / "run.sh"
    py = tmp_path / "loop.py"
    py.write_text(
        "import time, sys\n"
        "print('startup', flush=True)\n"
        "end=time.time()+" + str(seconds) + "\n"
        "while time.time()<end:\n"
        "    print('Epoch 0: 1%|#| 1/100 [00:01<00:01, 3.1it/s, loss=1.7]', flush=True)\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    script.write_text(f'#!/bin/sh\nexec "{default_python_exe()}" "{py}"\n', encoding="utf-8")
    script.chmod(0o755)
    return script


def _pid_alive(pid: int) -> bool:
    return os.path.exists(f"/proc/{pid}")


def test_start_records_real_pid_not_web_pid(paths, recipe, tmp_path):
    """旧版 save_pid 存的是 os.getpid()（uvicorn 自己）。"""
    script = _write_fake_train(tmp_path)
    j = TrainJob(run_id=1, folder="t", recipe=recipe, paths=paths, run_dir=tmp_path)
    j.start(script)
    try:
        assert j.pid is not None
        assert j.pid != os.getpid(), "pid 不能是 Web 服务自己"
        assert j.pgid == j.pid, "start_new_session 后 pgid 应等于 pid"
        assert j.status is JobStatus.RUNNING
    finally:
        j.kill(grace=2.0)


def test_kill_terminates_whole_group(paths, recipe, tmp_path):
    """对进程组发信号，python 子进程必须一起死。

    旧版只 kill(pid) 杀 sh，python 成孤儿继续占 GPU —— 已实测复现。
    """
    script = _write_fake_train(tmp_path)
    j = TrainJob(run_id=1, folder="t", recipe=recipe, paths=paths, run_dir=tmp_path)
    j.start(script)
    child_pid = j.pid
    assert child_pid is not None
    time.sleep(0.4)

    status, note = j.kill(grace=3.0)
    time.sleep(0.3)

    assert status is JobStatus.STOPPED
    assert not _pid_alive(child_pid), f"进程 {child_pid} 仍存活——进程组未被终止"


def test_kill_is_idempotent_on_dead_process(paths, recipe, tmp_path):
    script = _write_fake_train(tmp_path)
    j = TrainJob(run_id=1, folder="t", recipe=recipe, paths=paths, run_dir=tmp_path)
    j.start(script)
    j.kill(grace=2.0)
    status, _ = j.kill(grace=2.0)
    assert status is JobStatus.STOPPED


def test_stop_request_is_sticky(paths, recipe, tmp_path):
    """停止是单向的：request_stop 之后不得被任何逻辑清除。

    旧版 set() 后立刻 clear()，worker 可能读不到。
    """
    script = _write_fake_train(tmp_path)
    j = TrainJob(run_id=1, folder="t", recipe=recipe, paths=paths, run_dir=tmp_path)
    j.start(script)
    try:
        assert j.request_stop() is True
        time.sleep(0.1)
        # 内部标志必须仍为 set
        assert j._stop_requested.is_set()
        # 重复请求返回：已在停止流程中，仍为 True（幂等）
        assert j.request_stop() is True
    finally:
        j.kill(grace=2.0)


def test_pump_output_receives_lines(paths, recipe, tmp_path):
    script = _write_fake_train(tmp_path)
    j = TrainJob(run_id=1, folder="t", recipe=recipe, paths=paths, run_dir=tmp_path)
    j.start(script)

    lines: list[str] = []
    done = threading.Event()

    def pump():
        j.pump_output(lines.append)
        done.set()

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    time.sleep(0.8)
    j.kill(grace=2.0)
    done.wait(timeout=5)
    t.join(timeout=5)

    assert len(lines) > 3
    assert any("Epoch" in ln for ln in lines)


def test_pump_stops_on_stop_request(paths, recipe, tmp_path):
    """request_stop 之后 pump 应尽快退出循环。"""
    script = _write_fake_train(tmp_path, seconds=30)
    j = TrainJob(run_id=1, folder="t", recipe=recipe, paths=paths, run_dir=tmp_path)
    j.start(script)

    done = threading.Event()

    def pump():
        j.pump_output(lambda _l: None)
        done.set()

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    time.sleep(0.5)
    j.request_stop()
    # 进程仍在跑，pump 会读到下一行后退出
    assert done.wait(timeout=6), "pump 未在收到停止请求后退出"
    j.kill(grace=2.0)


def test_wait_sets_completed_on_zero_exit(paths, recipe, tmp_path):
    script = tmp_path / "ok.sh"
    script.write_text(f'#!/bin/sh\nexec "{default_python_exe()}" -c "print(1)"\n', encoding="utf-8")
    script.chmod(0o755)

    j = TrainJob(run_id=1, folder="t", recipe=recipe, paths=paths, run_dir=tmp_path)
    j.start(script)
    rc = j.wait(timeout=10)
    assert rc == 0
    assert j.status is JobStatus.COMPLETED


def test_wait_sets_error_on_nonzero(paths, recipe, tmp_path):
    script = tmp_path / "bad.sh"
    script.write_text(f'#!/bin/sh\nexec "{default_python_exe()}" -c "raise SystemExit(7)"\n', encoding="utf-8")
    script.chmod(0o755)

    j = TrainJob(run_id=1, folder="t", recipe=recipe, paths=paths, run_dir=tmp_path)
    j.start(script)
    rc = j.wait(timeout=10)
    assert rc == 7
    assert j.status is JobStatus.ERROR
    assert "7" in (j.error or "")


def test_double_start_rejected(paths, recipe, tmp_path):
    script = _write_fake_train(tmp_path)
    j = TrainJob(run_id=1, folder="t", recipe=recipe, paths=paths, run_dir=tmp_path)
    j.start(script)
    try:
        with pytest.raises(TrainError, match="已在运行"):
            j.start(script)
    finally:
        j.kill(grace=2.0)


def test_start_missing_script_raises_train_error(paths, recipe, tmp_path):
    """脚本不存在时必须**主动检查**并抛错。

    Popen(["sh", "missing.sh"]) 不会抛 OSError——sh 存在，只是以 127 退出。
    若不主动检查，任务会"成功启动"后立刻失败，只剩 "sh: cannot open" 这种
    难以定位的信息。
    """
    j = TrainJob(run_id=1, folder="t", recipe=recipe, paths=paths, run_dir=tmp_path)
    with pytest.raises(TrainError, match="训练脚本不存在"):
        j.start(tmp_path / "nonexistent.sh")
    assert j.status is JobStatus.ERROR


def test_snapshot_has_no_process_object(paths, recipe, tmp_path):
    """快照绝不应外泄 Popen。"""
    script = _write_fake_train(tmp_path)
    j = TrainJob(run_id=1, folder="t", recipe=recipe, paths=paths, run_dir=tmp_path)
    j.start(script)
    try:
        snap = j.snapshot()
        assert "_process" not in snap
        assert isinstance(snap["status"], str)
        assert snap["recipe_hash"] == recipe.hash
        for v in snap.values():
            assert not isinstance(v, subprocess.Popen)
    finally:
        j.kill(grace=2.0)


def test_status_terminal_flags():
    assert JobStatus.COMPLETED.terminal
    assert JobStatus.STOPPED.terminal
    assert JobStatus.ERROR.terminal
    assert not JobStatus.RUNNING.terminal
    assert not JobStatus.PENDING.terminal


def test_graceful_sigterm_preferred_over_kill(paths, recipe, tmp_path, monkeypatch):
    """宽限期内自行退出时不应升级到 SIGKILL。"""
    script = _write_fake_train(tmp_path, seconds=0.3)
    j = TrainJob(run_id=1, folder="t", recipe=recipe, paths=paths, run_dir=tmp_path)
    j.start(script)
    time.sleep(0.6)  # 让它自己跑完

    called: list[int] = []
    real = job_mod.os.killpg

    def spy(pgid, sig):
        called.append(sig)
        return real(pgid, sig)

    monkeypatch.setattr(job_mod.os, "killpg", spy)
    status, _ = j.kill(grace=3.0)
    assert status is JobStatus.STOPPED
    assert subprocess.signal.SIGKILL not in called, "正常情况下不应 SIGKILL"
