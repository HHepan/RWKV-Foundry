"""foundry.env 的单元测试。

核心验证：环境变量在 import torch 之前设置是有效的，之后设置会被明确警告，
而不是静默失效。
"""

from __future__ import annotations

import sys

import pytest

from foundry.env import (
    RWKVEnv,
    apply_rwkv_paths,
    bootstrap,
    desired_env,
    torch_already_loaded,
)

RUBRIC_KEYS = ("RWKV_V7_ON", "RWKV_JIT_ON", "RWKV_CUDA_ON")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in RUBRIC_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key in ("FOUNDRY_RWKV_V7", "FOUNDRY_RWKV_JIT", "FOUNDRY_RWKV_CUDA"):
        monkeypatch.delenv(key, raising=False)


def test_bootstrap_sets_all_three():
    bootstrap()
    assert all(__import__("os").environ[k] == "1" for k in RUBRIC_KEYS)


def test_bootstrap_is_idempotent(monkeypatch):
    first = bootstrap()
    second = bootstrap()
    assert first == second


def test_can_disable_cuda_for_cpu_machines(monkeypatch):
    monkeypatch.setenv("FOUNDRY_RWKV_CUDA", "0")
    env = bootstrap()
    assert env.cuda is False
    assert __import__("os").environ["RWKV_CUDA_ON"] == "0"


@pytest.mark.parametrize("value", ["1", "true", "YES", "on", " True "])
def test_env_flag_truthy_forms(monkeypatch, value):
    monkeypatch.setenv("FOUNDRY_RWKV_V7", value)
    assert desired_env().v7 is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_env_flag_falsy_forms(monkeypatch, value):
    monkeypatch.setenv("FOUNDRY_RWKV_V7", value)
    assert desired_env().v7 is False


def test_warns_when_torch_already_loaded(monkeypatch, capsys):
    """模拟「别处先 import 了 torch」这一静默失效场景。"""
    monkeypatch.setitem(sys.modules, "torch", type(sys)("torch"))
    assert torch_already_loaded() is True

    bootstrap()  # 非 strict 只警告
    captured = capsys.readouterr()
    assert "设置过晚" in captured.err


def test_strict_mode_raises_when_torch_already_loaded(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", type(sys)("torch"))
    with pytest.raises(RuntimeError, match="设置过晚"):
        bootstrap(strict=True)


def test_no_warning_when_torch_absent(monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    bootstrap()
    assert "设置过晚" not in capsys.readouterr().err


def test_apply_rwkv_paths(tmp_path):
    apply_rwkv_paths(tmp_path)
    assert str(tmp_path) in sys.path
    # 重复调用不应产生重复项
    apply_rwkv_paths(tmp_path)
    assert sys.path.count(str(tmp_path)) == 1
    sys.path.remove(str(tmp_path))


def test_rwkv_env_as_environ():
    assert RWKVEnv(v7=True, jit=False, cuda=True).as_environ() == {
        "RWKV_V7_ON": "1",
        "RWKV_JIT_ON": "0",
        "RWKV_CUDA_ON": "1",
    }
