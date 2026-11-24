"""foundry.paths 的单元测试。

重点覆盖两件事：
  1. 路径推导在各种 ``FOUNDRY_HOME`` 下都正确（旧项目的核心病灶）
  2. 路径穿越被拦截（旧项目的实际漏洞）
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from foundry.errors import ConfigError, PathTraversalError
from foundry.paths import (
    ENV_HOME,
    ENV_PEFT_DIR,
    PEFT_REQUIRED_FILES,
    resolve_paths,
    safe_join,
    safe_name,
)


# ── 路径推导 ─────────────────────────────────────────────────────────────
def test_default_root_is_repo_workspace(monkeypatch):
    monkeypatch.delenv(ENV_HOME, raising=False)
    monkeypatch.delenv(ENV_PEFT_DIR, raising=False)
    paths = resolve_paths()
    assert paths.root.name == "workspace"
    assert paths.root.parent.name == "RWKV-Foundry"


def test_env_home_overrides_root(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    monkeypatch.delenv(ENV_PEFT_DIR, raising=False)
    paths = resolve_paths()
    assert paths.root == tmp_path.resolve()
    # 未设置 PEFT 覆盖时，应落在 <workspace>/tools/RWKV-PEFT
    assert paths.peft_dir == tmp_path.resolve() / "tools" / "RWKV-PEFT"


def test_explicit_home_beats_env(monkeypatch, tmp_path):
    other = tmp_path / "from_env"
    monkeypatch.setenv(ENV_HOME, str(other))
    explicit = tmp_path / "explicit"
    paths = resolve_paths(home=explicit)
    assert paths.root == explicit.resolve()


def test_peft_dir_override(monkeypatch, tmp_path):
    shared_peft = tmp_path / "shared-rwkv-peft"
    monkeypatch.setenv(ENV_HOME, str(tmp_path / "ws"))
    monkeypatch.setenv(ENV_PEFT_DIR, str(shared_peft))
    paths = resolve_paths()
    assert paths.peft_dir == shared_peft.resolve()


def test_all_derived_paths_under_root(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    monkeypatch.delenv(ENV_PEFT_DIR, raising=False)
    p = resolve_paths()
    assert p.base_models == p.root / "base_models"
    assert p.data == p.root / "data"
    assert p.checkpoints == p.root / "checkpoints"
    assert p.db == p.root / "foundry.db"
    assert p.vocab == p.peft_dir / "json2binidx_tool" / "rwkv_vocab_v20230424.txt"
    assert p.preprocess_script == p.peft_dir / "json2binidx_tool" / "tools" / "preprocess_data.py"
    assert p.binidx_data == p.peft_dir / "json2binidx_tool" / "data"


def test_tilde_expansion(monkeypatch):
    monkeypatch.setenv(ENV_HOME, "~/foundry-test")
    monkeypatch.delenv(ENV_PEFT_DIR, raising=False)
    assert "~" not in str(resolve_paths().root)


# ── 环境校验 ─────────────────────────────────────────────────────────────
def test_validate_reports_missing_root(monkeypatch, tmp_path):
    missing = tmp_path / "nope"
    monkeypatch.setenv(ENV_HOME, str(missing))
    monkeypatch.delenv(ENV_PEFT_DIR, raising=False)
    problems = resolve_paths().problems()
    assert any("workspace 根目录不存在" in p for p in problems)


def test_validate_reports_missing_peft(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    monkeypatch.delenv(ENV_PEFT_DIR, raising=False)
    problems = resolve_paths().problems()
    assert any("RWKV-PEFT 未克隆" in p for p in problems)


def test_validate_reports_incomplete_peft(monkeypatch, tmp_path):
    """克隆中断的场景：目录在但缺关键文件。"""
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    monkeypatch.delenv(ENV_PEFT_DIR, raising=False)
    peft = tmp_path / "tools" / "RWKV-PEFT"
    peft.mkdir(parents=True)
    (peft / "train.py").write_text("", encoding="utf-8")  # 只有一个

    problems = resolve_paths().problems()
    joined = "\n".join(problems)
    assert "RWKV-PEFT 不完整" in joined
    # 应列出所有缺失项，而不是只报第一个
    assert "preprocess_data.py" in joined
    assert "rwkv_vocab_v20230424.txt" in joined


def test_complete_peft_passes(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    monkeypatch.delenv(ENV_PEFT_DIR, raising=False)
    peft = tmp_path / "tools" / "RWKV-PEFT"
    for rel in PEFT_REQUIRED_FILES:
        target = peft / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    (tmp_path / "base_models").mkdir()
    (tmp_path / "data").mkdir()

    paths = resolve_paths()
    assert paths.problems() == []
    paths.validate()  # 不抛异常


def test_validate_raises_with_all_problems(monkeypatch, tmp_path):
    """一次抛出全部问题，而不是修一个重启一次。"""
    monkeypatch.setenv(ENV_HOME, str(tmp_path / "absent"))
    monkeypatch.delenv(ENV_PEFT_DIR, raising=False)
    with pytest.raises(ConfigError) as exc:
        resolve_paths().validate()
    assert len(exc.value.problems) == 2


def test_ensure_output_dirs_only_creates_checkpoints(monkeypatch, tmp_path):
    """绝不自动创建输入目录——静默建空目录正是旧项目的坑。"""
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    monkeypatch.delenv(ENV_PEFT_DIR, raising=False)
    paths = resolve_paths()
    paths.ensure_output_dirs()
    assert paths.checkpoints.is_dir()
    assert not paths.base_models.exists()
    assert not paths.data.exists()


# ── 路径穿越 ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "name",
    ["../..", "..", ".", "a/b", "a\\b", "", "x\x00y", "/etc/passwd", "-rf", " lead"],
)
def test_safe_name_rejects(name):
    with pytest.raises(PathTraversalError):
        safe_name(name)


@pytest.mark.parametrize(
    "name",
    ["20260414", "checkpoint-1000", "run_01", "a.b.c", "default"],
)
def test_safe_name_accepts(name):
    assert safe_name(name) == name


def test_safe_join_normal(tmp_path):
    assert safe_join(tmp_path, "run1", "a.jsonl") == tmp_path / "run1" / "a.jsonl"


def test_safe_join_blocks_traversal(tmp_path):
    with pytest.raises(PathTraversalError):
        safe_join(tmp_path, "..", "evil")


def test_safe_join_blocks_symlink_escape(tmp_path):
    """符号链接绕过：resolve() 之后必须仍在 base 内。"""
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = base / "link"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(PathTraversalError):
        safe_join(base, "link", "file.txt")


def test_rmtree_scenario_is_blocked(tmp_path):
    """复现旧项目 DELETE /api/train/records/{folder_name} 的漏洞路径。"""
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "important.txt").write_text("data", encoding="utf-8")

    with pytest.raises(PathTraversalError):
        safe_join(checkpoints, "../victim", kind="文件夹名")
    assert (victim / "important.txt").exists()
