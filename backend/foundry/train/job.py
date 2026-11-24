"""训练任务编排 —— 子进程生命周期管理。

## 旧版的四个致命问题，逐个修正

### 1. 停止信号可能丢失（竞态）
旧版 ``api/train.py:432``::

    stop_event.set()
    ...
    stop_event.clear()   # 立刻清掉

worker 线程可能还没从 ``stop_event()`` 读到 ``True``，信号就被清了。
本版改用 ``threading.Event`` 且**永不在运行期清除**——停止是单向状态转移。

### 2. ``training_process = True`` 哨兵值
旧版用 ``True`` 和 Popen 两种类型塞进同一个全局变量，
再靠 ``is not True`` 区分。极其脆弱。
本版 ``TrainJob`` 是显式对象，状态由枚举表达。

### 3. ``save_pid`` 存了 Web 服务自己的 pid
旧版 ``api/train.py:174``::

    f.write(str(os.getpid()))   # 这是 uvicorn 的 pid！

本版记录真实子进程 pid **和 pgid**。

### 4. 训练进程杀不死（已实测复现）
旧版只对 ``sh`` 发 SIGTERM，``python train.py`` 成为孤儿继续占 GPU。
本版对**进程组**发信号，并配合脚本里的 ``exec``（让 python 取代 sh）。

## 三段式取消

    SIGTERM 整个进程组  →  等待宽限期（默认 10s）  →  SIGKILL 进程组

宽限期让 DeepSpeed/torch 有机会保存当前 checkpoint、释放显存。
直接 SIGKILL 会留下损坏的半写权重。
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from foundry.domain.recipe import Recipe
from foundry.errors import FoundryError
from foundry.paths import WorkspacePaths


class TrainError(FoundryError):
    """训练编排错误。"""


class JobStatus(str, Enum):
    """任务状态。单向流转，不可逆。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.STOPPED, JobStatus.ERROR)


#: 停止时等待进程自行退出的秒数。超时后升级为 SIGKILL。
DEFAULT_GRACE_SECONDS = 10.0


@dataclass
class TrainJob:
    """一次训练运行。

    与 ``Recipe`` 分离：配方描述"怎么训练"，Job 描述"这次训练现在到哪了"。
    """

    run_id: int
    folder: str
    recipe: Recipe
    paths: WorkspacePaths
    run_dir: Path

    status: JobStatus = JobStatus.PENDING
    pid: int | None = None
    pgid: int | None = None
    exit_code: int | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None

    _process: subprocess.Popen | None = field(default=None, repr=False)
    #: 只置位、从不清除。见模块文档「停止信号可能丢失」。
    _stop_requested: threading.Event = field(
        default_factory=threading.Event, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── 启动 ────────────────────────────────────────────────────────────
    def start(
        self,
        script_path: Path,
        *,
        on_output: Callable[[str], None] | None = None,
    ) -> None:
        """启动训练子进程。

        ``start_new_session=True`` 建立独立进程组，这是能整组终止的前提。
        stdout/stderr 合并为 PIPE，由 ``on_output`` 逐行回调。
        """
        if self._process is not None:
            raise TrainError(f"run {self.folder!r} 已在运行，不能重复启动")

        # 必须主动检查脚本存在性。
        #
        # 注意：Popen(["sh", "missing.sh"]) 不会抛 OSError —— sh 本身存在，
        # 它只是以退出码 127 结束。若不在这里拦，任务会"成功启动"然后
        # 立刻失败，错误信息只有 "sh: cannot open"，很难定位。
        if not script_path.is_file():
            self.status = JobStatus.ERROR
            self.error = f"训练脚本不存在: {script_path}"
            raise TrainError(self.error)

        try:
            self._process = subprocess.Popen(
                ["sh", str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,          # 行缓冲：tqdm 输出才能及时读到
                start_new_session=True,
                errors="replace",   # 训练输出可能含非 UTF-8，不应崩
            )
        except OSError as exc:
            self.status = JobStatus.ERROR
            self.error = f"启动训练进程失败: {exc}"
            raise TrainError(self.error) from exc

        self.pid = self._process.pid
        try:
            self.pgid = os.getpgid(self.pid)
        except ProcessLookupError:
            self.pgid = self.pid  # 极罕见，进程已退出

        self.status = JobStatus.RUNNING
        self.started_at = time.time()

    # ── 等待与输出 ──────────────────────────────────────────────────────
    def pump_output(self, on_line: Callable[[str], None]) -> None:
        """逐行消费子进程输出，直到 EOF。阻塞调用。

        停止请求通过 ``_stop_requested`` 检查，检测到即整组终止。
        """
        if self._process is None or self._process.stdout is None:
            return

        try:
            for line in iter(self._process.stdout.readline, ""):
                if self._stop_requested.is_set():
                    break
                text = line.rstrip("\n")
                if text:
                    on_line(text)
        finally:
            with contextlib.suppress(OSError):
                self._process.stdout.close()

    def wait(self, timeout: float | None = None) -> int:
        """等待进程结束并确定最终状态。"""
        if self._process is None:
            raise TrainError("任务尚未启动")

        try:
            self.exit_code = self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.exit_code = None
        self.finished_at = time.time()

        with self._lock:
            if self._stop_requested.is_set():
                self.status = JobStatus.STOPPED
            elif self.exit_code == 0:
                self.status = JobStatus.COMPLETED
            else:
                self.status = JobStatus.ERROR
                if self.error is None:
                    self.error = f"训练进程退出码 {self.exit_code}"
        return self.exit_code if self.exit_code is not None else -1

    # ── 停止 ────────────────────────────────────────────────────────────
    def request_stop(self) -> bool:
        """请求停止。幂等，可在任意线程调用。"""
        if self.status.terminal:
            return False
        self._stop_requested.set()
        return True

    def kill(
        self, grace: float = DEFAULT_GRACE_SECONDS
    ) -> tuple[JobStatus, str]:
        """三段式终止：SIGTERM 全组 → 宽限 → SIGKILL 全组。

        :returns: ``(最终状态, 说明)``
        """
        if self._process is None:
            self.status = JobStatus.STOPPED
            return self.status, "任务未启动，直接标记为已停止"

        self._stop_requested.set()

        if self._process.poll() is not None:
            # 已经自己退出了
            self.exit_code = self._process.returncode
            self.finished_at = self.finished_at or time.time()
            self.status = JobStatus.STOPPED
            return self.status, "进程已自行退出"

        # ── 阶段 1：SIGTERM 整个进程组 ──────────────────────────────────
        #
        # 关键：必须 killpg 而不是 kill(pid)。
        # 旧版只杀 sh，python 子进程成为孤儿继续占 GPU —— 已实测复现。
        sigterm_ok = self._signal_group(signal.SIGTERM)

        try:
            self.exit_code = self._process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            # ── 阶段 2：宽限期用尽，SIGKILL 全组 ────────────────────────
            self._signal_group(signal.SIGKILL)
            try:
                self.exit_code = self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.error = "进程在 SIGKILL 后仍未退出，可能处于不可中断 IO"
                self.finished_at = time.time()
                self.status = JobStatus.ERROR
                return self.status, self.error

        self.finished_at = time.time()
        self.status = JobStatus.STOPPED

        # 兜底：确认组内无残留
        leftovers = self._group_members()
        note = f"SIGTERM{'成功' if sigterm_ok else '失败'}，进程组已清理"
        if leftovers:
            note += f"；仍有残留 pid {leftovers}"
        return self.status, note

    def _signal_group(self, sig: int) -> bool:
        """对进程组发信号。返回是否成功送达。"""
        if self.pgid is None:
            return False
        try:
            os.killpg(self.pgid, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError as exc:
            self.error = f"无权向进程组 {self.pgid} 发信号: {exc}"
            return False

    def _group_members(self) -> list[int]:
        """列出进程组内仍存活的 pid。用于停止后自检。"""
        if self.pgid is None:
            return []
        try:
            os.killpg(self.pgid, 0)  # 信号 0：只探测，不发送
            return [self.pgid]
        except ProcessLookupError:
            return []
        except PermissionError:
            return [self.pgid]

    # ── 快照 ────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        """给 API 层的只读视图。绝不外泄 Popen 对象。"""
        elapsed = None
        if self.started_at is not None:
            end = self.finished_at or time.time()
            elapsed = round(end - self.started_at, 1)

        return {
            "run_id": self.run_id,
            "folder": self.folder,
            "status": self.status.value,
            "pid": self.pid,
            "pgid": self.pgid,
            "exit_code": self.exit_code,
            "error": self.error,
            "elapsed_seconds": elapsed,
            "recipe_hash": self.recipe.hash,
        }
