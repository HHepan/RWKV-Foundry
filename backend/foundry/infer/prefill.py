"""增量 prefill —— 本项目的技术护城河。

## 为什么 RWKV 能做增量 prefill

RWKV 是**循环**模型：state 累积了已处理过的全部 token 的信息，且显存恒定
（没有 KV cache 增长）。所以第二轮对话时，只要上一轮 prompt 的前缀没变，
那部分 token 的计算结果就已经在 state 里了，**只需喂新增的差量**。

Transformer 做不了这件事——它的 KV cache 是可增的，但每轮重算的代价与
增量不同，且没有"固定 size 的压缩态"这个概念。

效果：多轮对话从 O(n²) 降到 O(n)。

## 为什么必须在 token 层面比较

**绝不能在字符层面切前缀。** 一个中文字符对应多个 UTF-8 字节，按字符数
截断会切进多字节字符中间，导致：

- ``decode()`` 抛 ``UnicodeDecodeError``
- 即使不抛，token 序列也会错位，state 与 prompt 从此不一致

旧项目为此刻意写了 ``_token_prefix_chars()``，用**二分查找**定位第 k 个
token 对应的字符边界。那段代码是吃过亏才写得出来的，这里完整保留。

## 失效条件

以下任一情况必须**全量重建**，不能复用：

1. 没有 state（首轮）
2. 没有 fed_text 基线（不知道 state 里吃了什么）
3. 新 prompt 与旧 prompt 的**第一个 token 就不同**（例如换了角色卡、
   关掉了 presystem）——此时 LCP 长度为 0

特别地：``max_rounds`` 裁剪会让 prompt 开头变化，LCP 立刻塌到接近 0。
这就是为什么默认不裁剪历史。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillPlan:
    """一次增量 prefill 的计算结果。"""

    #: 真正要喂进模型的 token
    feed_tokens: list[int]
    #: 被复用的 token 数（已在 state 中）
    reused_tokens: int
    #: 新 prompt 的总 token 数
    total_tokens: int
    #: 复用的 token 前缀对应的字符数（用于更新 fed_text 基线）
    reused_chars: int
    #: 是否需要全量重建（state 作废）
    full_rebuild: bool

    @property
    def fed_tokens(self) -> int:
        return len(self.feed_tokens)

    @property
    def reuse_ratio(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.reused_tokens / self.total_tokens


def longest_common_prefix(a: list[int], b: list[int]) -> int:
    """两个 token 序列的最长公共前缀长度。"""
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def token_prefix_chars(text: str, n_tokens: int, encode) -> int:
    """返回 ``text`` 中前 ``n_tokens`` 个 token 对应的字符数。

    用二分查找定位 token 边界的字符位置，保证切点落在字符边界上而不是
    多字节字符中间。

    :param encode: ``str -> list[int]`` 的编码函数
    """
    if n_tokens <= 0:
        return 0

    try:
        total = len(encode(text))
    except Exception:
        return 0
    if n_tokens >= total:
        return len(text)

    lo, hi = 0, len(text)
    # 不变式：encode(text[:lo]) <= n_tokens，encode(text[:hi]) > n_tokens
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(encode(text[:mid])) <= n_tokens:
            lo = mid
        else:
            hi = mid - 1
    return lo


def plan_prefill(
    *,
    prompt: str,
    fed_text: str,
    has_state: bool,
    encode,
) -> PrefillPlan:
    """计算增量 prefill 方案。

    :param prompt: 本轮完整 prompt
    :param fed_text: state 中已经"吃进去"的完整文本
    :param has_state: 是否已有可复用的 state
    :param encode: ``str -> list[int]`` 编码函数
    """
    full_tokens = encode(prompt)
    total = len(full_tokens)

    # ── 无法复用：全量重建 ──────────────────────────────────────────────
    if not has_state or not fed_text:
        return PrefillPlan(
            feed_tokens=full_tokens,
            reused_tokens=0,
            total_tokens=total,
            reused_chars=0,
            full_rebuild=True,
        )

    fed_tokens = encode(fed_text)
    common = longest_common_prefix(fed_tokens, full_tokens)

    # 一个 token 都复用不上，说明 prompt 开头变了（换角色卡/关 presystem 等）
    if common == 0:
        return PrefillPlan(
            feed_tokens=full_tokens,
            reused_tokens=0,
            total_tokens=total,
            reused_chars=0,
            full_rebuild=True,
        )

    # 完全一致：没有新内容要喂（重复提交同一 prompt 的边界情况）
    if common == total:
        return PrefillPlan(
            feed_tokens=[],
            reused_tokens=common,
            total_tokens=total,
            reused_chars=len(prompt),
            full_rebuild=False,
        )

    reused_chars = token_prefix_chars(prompt, common, encode)

    return PrefillPlan(
        feed_tokens=full_tokens[common:],
        reused_tokens=common,
        total_tokens=total,
        reused_chars=reused_chars,
        full_rebuild=False,
    )


def rebuild_fed_text(*, prompt: str, raw_output: str) -> str:
    """生成下一轮用的 fed_text 基线。

    fed_text 必须精确反映 state 里吃进去的文本。由于复用的前缀 token 与
    本轮 prompt 的前缀完全一致，state 的 token 序列恒等于::

        encode(prompt) + encode(raw_output)

    因此基线就是 ``prompt + raw_output``，与是否复用了前缀无关。

    边界说明：``encode(prompt) + encode(raw)`` 与 ``encode(prompt + raw)``
    在分词器跨边界合并时可能有极细微差异。这不需要特殊处理——下轮 LCP
    比较会自动少复用几个 token 而不会出错，安全性由 LCP 本身保证。
    """
    return prompt + raw_output
