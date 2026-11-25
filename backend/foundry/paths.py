"""文件系统布局 —— 全项目唯一的路径真相来源。

任何模块都不得自行拼接 workspace 路径。旧版 llm-finetune-webui 把 workspace 根
硬编码在 4 个文件、6 个位置上（全部指向作者本机的 /home/lijiahao/...），换一台
机器整个后端就完全无法启动。这个模块的存在就是为了让那件事不可能再发生。

用法::

    from foundry.paths import get_paths

    paths = get_paths()
    paths.checkpoints        # <workspace>/checkpoints
    paths.vocab              # <peft>/json2binidx_tool/rwkv_vocab_v20230424.txt

覆盖顺序（后者优先）：
    1. 默认值      <repo>/workspace
    2. 环境变量    FOUNDRY_HOME=/data/foundry
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from foundry.errors import ConfigError, PathTraversalError

#: 覆盖 workspace 根目录
ENV_HOME = "FOUNDRY_HOME"

#: 覆盖 RWKV-PEFT 克隆位置（默认 <workspace>/tools/RWKV-PEFT）
ENV_PEFT_DIR = "FOUNDRY_PEFT_DIR"

#: RWKV-PEFT 必须存在的关键文件（相对其根目录）。
#: 这些是上游仓库的固定内部布局，我们不能改，只能校验。
PEFT_REQUIRED_FILES: tuple[str, ...] = (
    "train.py",
    "json2binidx_tool/tools/preprocess_data.py",
    "json2binidx_tool/rwkv_vocab_v20230424.txt",
    "scripts/lora.sh",
)

#: 允许出现在目录名/文件名中的字符。用于拒绝路径穿越。
_SAFE_NAME_RE = re.compile(r"^[\w][\w.\-]*$")


def repo_root() -> Path:
    """仓库根目录（backend/ 的上一级）。

    本文件位于 ``<repo>/backend/foundry/paths.py``：
    parents[0]=foundry, parents[1]=backend, parents[2]=<repo>
    """
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WorkspacePaths:
    """workspace 的目录布局。

    分为三类，语义上不可混：
      - 输入：base_models、data（用户提供，本程序只读）
      - 输出：checkpoints、foundry.db（本程序写）
      - 第三方：peft_dir（clone 来的上游仓库，布局不可改）
    """

    root: Path
    peft_dir: Path

    # ── 输入 ────────────────────────────────────────────────────────────
    @property
    def base_models(self) -> Path:
        """基底权重 .pth 所在目录。"""
        return self.root / "base_models"

    @property
    def data(self) -> Path:
        """训练数据集 .jsonl 所在目录。"""
        return self.root / "data"

    # ── 输出 ────────────────────────────────────────────────────────────
    @property
    def checkpoints(self) -> Path:
        """训练产物：每个 run 一个子目录。"""
        return self.root / "checkpoints"

    @property
    def db(self) -> Path:
        """元数据与索引库（不存权重、不存数据集）。"""
        return self.root / "foundry.db"

    # ── RWKV-PEFT 内部（上游固定布局）───────────────────────────────────
    @property
    def vocab(self) -> Path:
        return self.peft_dir / "json2binidx_tool" / "rwkv_vocab_v20230424.txt"

    @property
    def preprocess_script(self) -> Path:
        return self.peft_dir / "json2binidx_tool" / "tools" / "preprocess_data.py"

    @property
    def binidx_data(self) -> Path:
        """binidx 转换的中间产物目录（训练前会被清空）。"""
        return self.peft_dir / "json2binidx_tool" / "data"

    @property
    def tokenizer_tools(self) -> Path:
        """需要加入 sys.path 才能 ``from rwkv_tokenizer import ...``。"""
        return self.peft_dir / "json2binidx_tool" / "tools"

    @property
    def peft_scripts(self) -> Path:
        return self.peft_dir / "scripts"

    # ── 校验与初始化 ────────────────────────────────────────────────────
    def problems(self) -> list[str]:
        """返回全部环境问题。空列表表示环境可用。"""
        found: list[str] = []

        if not self.root.is_dir():
            found.append(
                f"workspace 根目录不存在: {self.root}\n"
                f"  修复: mkdir -p '{self.root}'，或设置环境变量 {ENV_HOME}"
            )

        if not self.peft_dir.is_dir():
            found.append(
                f"RWKV-PEFT 未克隆到: {self.peft_dir}\n"
                f"  修复: git clone https://github.com/JL-er/RWKV-PEFT.git '{self.peft_dir}'"
            )
        else:
            missing = [rel for rel in PEFT_REQUIRED_FILES if not (self.peft_dir / rel).exists()]
            if missing:
                found.append(
                    f"RWKV-PEFT 不完整，缺少 {len(missing)} 个关键文件（位于 {self.peft_dir}）:\n"
                    + "\n".join(f"  - {rel}" for rel in missing)
                    + "\n  可能是克隆中断或上游目录结构调整，请重新克隆并核对。"
                )
        return found

    def validate(self) -> None:
        """环境不可用时抛 ConfigError，一次性列出全部问题。"""
        found = self.problems()
        if found:
            raise ConfigError(found)

    def ensure_output_dirs(self) -> None:
        """创建输出目录。

        只创建**输出**目录（checkpoints）。输入目录绝不自动创建——
        静默创建空的 base_models/data 正是旧项目让人误以为环境就绪的原因。
        """
        self.checkpoints.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, str]:
        """便于日志与 ``/api/config`` 端点回显。"""
        return {
            "workspace": str(self.root),
            "peft_dir": str(self.peft_dir),
            "base_models": str(self.base_models),
            "data": str(self.data),
            "checkpoints": str(self.checkpoints),
            "db": str(self.db),
            "vocab": str(self.vocab),
        }


def resolve_paths(home: str | os.PathLike[str] | None = None) -> WorkspacePaths:
    """构造路径集合，不做存在性校验。

    :param home: workspace 根目录。为 None 时依次读 ``FOUNDRY_HOME``、
                 ``<repo>/workspace``。
    """
    if home is not None:
        root = Path(home).expanduser().resolve()
    elif env_home := os.environ.get(ENV_HOME):
        root = Path(env_home).expanduser().resolve()
    else:
        root = repo_root() / "workspace"

    if peft_override := os.environ.get(ENV_PEFT_DIR):
        peft_dir = Path(peft_override).expanduser().resolve()
    else:
        peft_dir = root / "tools" / "RWKV-PEFT"

    return WorkspacePaths(root=root, peft_dir=peft_dir)


@lru_cache(maxsize=1)
def get_paths() -> WorkspacePaths:
    """进程级单例。测试中如需重置请调用 ``get_paths.cache_clear()``。"""
    return resolve_paths()


# ── 路径安全 ─────────────────────────────────────────────────────────────
def safe_name(name: str, *, kind: str = "名称") -> str:
    """校验单段名称（目录名/文件名/ID），拒绝路径穿越。

    旧项目有 4 个接口把用户输入直接拼进路径，其中
    ``DELETE /api/train/records/{folder_name}`` 最终执行 ``shutil.rmtree``，
    传 ``../..`` 即可删除任意目录。
    """
    if not name or not isinstance(name, str):
        raise PathTraversalError(f"{kind}不能为空")
    if name in (".", ".."):
        raise PathTraversalError(f"{kind}非法: {name!r}")
    if "/" in name or "\\" in name or "\x00" in name:
        raise PathTraversalError(f"{kind}不得包含路径分隔符: {name!r}")
    if not _SAFE_NAME_RE.match(name):
        raise PathTraversalError(
            f"{kind}含有非法字符，仅允许字母、数字、下划线、连字符、点，且须以字母数字开头: {name!r}"
        )
    return name


def safe_join(base: Path, *parts: str, kind: str = "路径") -> Path:
    """把各段名称安全地拼到 base 下，并确认结果确实位于 base 之内。

    ## 校验放在「逻辑路径」上，而不是 resolve() 之后

    起初这里用 ``base.joinpath(*parts).resolve()`` 比对父目录。这在
    **base 下存在软链接时会把合法路径误判为越界**：

        base_models/rwkv7-2.9b.pth -> /old/project/base_models/rwkv7-2.9b.pth

    ``resolve()`` 跟到旧项目目录，于是 ``base_resolved not in target.parents``
    成立，正常请求被 400 拒绝。实测就是这样挂掉的。

    正确的判据是**逻辑路径**：`base / posix_normalized(parts)` 必须仍在 base
    之下。软链接本身是合法的用户资产（跨盘复用权重是常见需求），
    不该被当作攻击。

    仍然能拦住真正的穿越，因为：
      1. ``safe_name`` 已拒绝 ``..``、``/``、``\\`` 和非法字符
      2. ``os.path.normpath`` 后再比对前缀，``..`` 无法逃逸
    """
    for part in parts:
        safe_name(part, kind=kind)

    # 第一道：逻辑路径校验（不解析软链）。拦住 ../ 这类纯路径穿越。
    logical = Path(os.path.normpath(os.path.join(str(base), *parts)))
    base_logical = Path(os.path.normpath(str(base)))

    if logical != base_logical and base_logical not in logical.parents:
        raise PathTraversalError(f"{kind}越界: {parts} 不在 {base} 之内")

    # 第二道：软链目标校验，必须**逐段**检查。
    #
    # 关键：软链可能出现在中间路径段上（base/link/file.txt 里的 link），
    # 只检查最终路径的 is_symlink() 是查不出来的 —— 最终文件可能根本不存在。
    #
    # 为什么要区分「文件软链」和「目录软链」：
    #
    #   文件软链（允许）：base_models/rwkv7-2.9b.pth -> /other/.../rwkv7-2.9b.pth
    #       跨盘复用权重是正常需求，指向普通文件，只读无风险。
    #
    #   目录软链（拒绝）：base/link -> /etc
    #       后续任何写入都会落到 /etc 下，这是真正的逃逸。
    cursor = base_logical
    for part in parts:
        cursor = cursor / part
        if not cursor.is_symlink():
            continue
        try:
            real = cursor.resolve()
        except OSError:
            continue
        if real == base_logical or base_logical in real.parents:
            continue  # 仍指向 base 内部，无害
        if real.is_dir():
            raise PathTraversalError(
                f"{kind}越界: {cursor} 是指向 base 外目录的软链 -> {real}"
            )

    return logical
