"""Vendored 第三方代码 —— 从上游复制并本地维护的模块。

## 为什么要有这个目录

这两个文件**不是我们自己写的**，但必须改动才能满足本项目需求。放在这里
而不是去改 site-packages，原因：

1. **可追溯**：每个文件的来源、版本、修改点都记录在 ``PROVENANCE.md``
2. **可重装**：``pip install --force-reinstall rwkv`` 不会覆盖我们的修改
3. **可审查**：升级上游时能清楚看到我们改了什么

## 重要约定

**任何对本目录文件的修改，必须同步更新 ``PROVENANCE.md``。**

否则半年后没人知道某段代码为什么和上游不一样——这类"神秘的本地补丁"
是维护灾难的典型来源。旧项目把改动写进 README 的散文里，等于没记录。

## 隔离 torch

本包 ``__init__`` **刻意不 import 任何子模块**。子模块都依赖 torch，
若在此处 import 会让 ``import foundry.third_party`` 在无 GPU 环境失败。
调用方必须显式 import 具体模块。
"""

from __future__ import annotations

__all__ = ["PROVENANCE", "rwkv_utils_patch", "rwkv_x070_dual"]

#: 上游来源说明，便于运行时自检
PROVENANCE = {
    "rwkv_x070_dual": {
        "upstream": "BlinkDL/RWKV-LM (rwkv.model.RWKV_x070)",
        "upstream_version": "rwkv pip 0.8.30",
        "purpose": "支持跨 GPU 层间切分（cuda:0 fp16 -> cuda:1 fp16）",
    },
    "rwkv_utils_patch": {
        "upstream": "BlinkDL/RWKV-LM (rwkv.utils.PIPELINE)",
        "upstream_version": "rwkv pip 0.8.30",
        "purpose": "generate() 返回 state，支持增量 prefill 与跨轮重复惩罚",
    },
}
