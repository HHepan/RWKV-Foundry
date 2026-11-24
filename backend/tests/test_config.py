"""foundry.config 的单元测试。

验证配置优先级、默认值对齐 RWKV-Runner、以及旧项目 CORS 与 vite 矛盾的问题已修正。
"""

from __future__ import annotations

import pytest

from foundry.config import (
    AppConfig,
    InferDefaults,
    InferRuntime,
    LoggingConfig,
    ServerConfig,
    TrainDefaults,
    VOCAB_SIZE,
)

_ENV_KEYS = (
    "FOUNDRY_HOST",
    "FOUNDRY_PORT",
    "FOUNDRY_CORS_ORIGINS",
    "FOUNDRY_INFER_STRATEGY",
    "FOUNDRY_MAX_SESSIONS",
    "FOUNDRY_MAX_QUEUE",
    "FOUNDRY_LOG_LEVEL",
    "FOUNDRY_DUMP_PROMPT",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


# ── 服务配置 ─────────────────────────────────────────────────────────────
def test_server_defaults():
    c = ServerConfig()
    assert c.host == "127.0.0.1"
    assert c.port == 8000
    assert "http://localhost:5173" in c.cors_origins


def test_cors_includes_vite_and_127(monkeypatch):
    """旧项目漏掉 127.0.0.1:5173，用 127 打开前端会被拒。"""
    c = ServerConfig.from_env()
    assert "http://localhost:5173" in c.cors_origins
    assert "http://127.0.0.1:5173" in c.cors_origins


def test_cors_override(monkeypatch):
    monkeypatch.setenv("FOUNDRY_CORS_ORIGINS", "http://192.168.1.10:5173, http://a.b:1")
    c = ServerConfig.from_env()
    assert c.cors_origins == ("http://192.168.1.10:5173", "http://a.b:1")


def test_port_from_env(monkeypatch):
    monkeypatch.setenv("FOUNDRY_PORT", "9001")
    assert ServerConfig.from_env().port == 9001


def test_invalid_port_raises(monkeypatch):
    monkeypatch.setenv("FOUNDRY_PORT", "not-a-number")
    with pytest.raises(ValueError, match="必须是整数"):
        ServerConfig.from_env()


# ── 推理默认值 ───────────────────────────────────────────────────────────
def test_infer_defaults_align_runner():
    """对齐 RWKV-Runner 的 TextRWKV：temp=1, top_p=0.3, top_k=0。"""
    d = InferDefaults()
    assert d.temperature == 1.0
    assert d.top_p == 0.3
    assert d.top_k == 0
    assert d.alpha_presence == 0.0
    assert d.alpha_frequency == 1.0
    assert d.alpha_decay == 0.996


def test_max_rounds_defaults_to_zero():
    """RWKV 是循环模型，默认不裁剪历史，否则增量 prefill 失效。"""
    assert InferDefaults().max_rounds == 0


def test_infer_defaults_as_dict_roundtrip():
    d = InferDefaults()
    payload = d.as_dict()
    assert payload["max_tokens"] == d.max_tokens
    assert payload["presystem"] is True
    assert set(payload) == {
        "max_tokens", "temperature", "top_p", "top_k",
        "alpha_frequency", "alpha_presence", "alpha_decay",
        "newline_adj", "global_penalty", "presystem",
        "max_rounds", "clean_rounds",
    }


# ── 推理运行时 ───────────────────────────────────────────────────────────
def test_infer_strategy_default_is_single_gpu():
    """旧项目把 'cuda:0 fp16 -> cuda:1 fp16' 写死，单卡机直接崩。"""
    assert InferRuntime().strategy == "cuda:0 fp16"


def test_infer_strategy_from_env(monkeypatch):
    monkeypatch.setenv("FOUNDRY_INFER_STRATEGY", "cuda:0 bf16 -> cuda:1 bf16")
    assert InferRuntime.from_env().strategy == "cuda:0 bf16 -> cuda:1 bf16"


def test_cpu_strategy_supported(monkeypatch):
    monkeypatch.setenv("FOUNDRY_INFER_STRATEGY", "cpu fp32")
    assert InferRuntime.from_env().strategy == "cpu fp32"


def test_session_and_queue_limits(monkeypatch):
    monkeypatch.setenv("FOUNDRY_MAX_SESSIONS", "8")
    monkeypatch.setenv("FOUNDRY_MAX_QUEUE", "2")
    r = InferRuntime.from_env()
    assert r.max_sessions == 8
    assert r.max_queue == 2


# ── 训练默认值 ───────────────────────────────────────────────────────────
def test_train_defaults_shape():
    d = TrainDefaults()
    assert d.model_size == "2.9B"
    assert d.lora_r == 32
    assert d.my_testing == "x070"
    assert d.precision == "bf16"


def test_precision_is_not_fp16():
    """上游 CUDA 内核拒绝 fp16，默认必须是 bf16。"""
    assert TrainDefaults().precision != "fp16"


def test_vocab_size():
    assert VOCAB_SIZE == 65536


# ── 日志 ─────────────────────────────────────────────────────────────────
def test_dump_prompt_off_by_default():
    """旧项目无条件打印完整 prompt，默认必须关闭。"""
    assert LoggingConfig().dump_prompt is False


def test_logging_from_env(monkeypatch):
    monkeypatch.setenv("FOUNDRY_LOG_LEVEL", "debug")
    monkeypatch.setenv("FOUNDRY_DUMP_PROMPT", "1")
    c = LoggingConfig.from_env()
    assert c.level == "DEBUG"
    assert c.dump_prompt is True


# ── 总装 ─────────────────────────────────────────────────────────────────
def test_app_config_from_env(monkeypatch):
    monkeypatch.setenv("FOUNDRY_PORT", "8123")
    cfg = AppConfig.from_env()
    assert cfg.server.port == 8123
    assert cfg.train_defaults.model_size == "2.9B"
    assert cfg.infer_defaults.top_p == 0.3


def test_app_config_defaults_are_fresh_instances():
    a, b = AppConfig(), AppConfig()
    assert a.server is not b.server
