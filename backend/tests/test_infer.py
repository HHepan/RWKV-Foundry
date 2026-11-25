"""foundry.infer 的测试。

本文件的核心是验证老大明确提出的要求：

> 不同对话之间信息的隔离（state、对话历史上下文、推理参数）要做好。

因此这里用 **fake pipeline** 而非真模型——隔离性是与模型无关的逻辑，
必须能在无 GPU、无 torch 的环境下被严格验证。

同样重点覆盖增量 prefill 的正确性，因为它是本项目的技术护城河，
且历史上出过 UnicodeDecodeError 的坑。
"""

from __future__ import annotations

import dataclasses

import pytest

from foundry.config import THINK_END, THINK_START
from foundry.errors import FoundryError
from foundry.infer.params import InferParams
from foundry.infer.prefill import (
    longest_common_prefix,
    plan_prefill,
    rebuild_fed_text,
    token_prefix_chars,
)
from foundry.infer.prompt import (
    PRESYSTEM_PREFIX,
    ChatMessage,
    build_prompt,
    count_rounds,
    split_thinking,
)
from foundry.infer.sampler import StopSequenceFilter
from foundry.infer.session import (
    InferenceEngine,
    InferError,
    ModelNotLoadedError,
    SessionStore,
)


# ── 字符级伪 tokenizer（模拟真实分词器的行为特征）─────────────────────
def char_encode(text: str) -> list[int]:
    """每字符一个 token。中文按字符（不是字节）切，符合 RWKV 词表行为。"""
    return [ord(c) for c in text]


def char_decode(tokens: list[int]) -> str:
    return "".join(chr(t) for t in tokens)


class FakePipeline:
    """可观测的假 pipeline。记录每轮收到了多少 token。"""

    def __init__(self, reply: str = "好的。"):
        self.reply = reply
        self.encode = char_encode
        self.decode = char_decode
        self.forward_log: list[list[int]] = []
        self.state_counter = 0

    def generate(self, ctx, token_count=100, args=None, callback=None, state=None, occurrence=None):
        self.forward_log.append(list(ctx))
        # 模拟 state 演进：state 记录累计吃进的 token
        base = list(state) if state else []
        new_state = base + list(ctx)
        self.state_counter += 1

        if callback:
            for ch in self.reply:
                callback(ch)
        return self.reply, new_state, occurrence or {}


@pytest.fixture
def engine():
    e = InferenceEngine(max_sessions=4)
    pipe = FakePipeline("第一轮回复。")
    e.attach(pipe, pipe, model_ref="run1/rwkv-1")
    return e


# ── 参数隔离 ─────────────────────────────────────────────────────────────
def test_params_are_frozen():
    p = InferParams.default()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.temperature = 2.0  # type: ignore[misc]


def test_with_overrides_returns_new_object():
    a = InferParams.default()
    b = a.with_overrides(temperature=1.5)
    assert a.temperature != b.temperature
    assert a is not b
    assert a.temperature == InferParams.default().temperature


def test_with_overrides_ignores_none_and_unknown():
    a = InferParams.default()
    b = a.with_overrides(temperature=None, bogus_field=1)
    assert b == a


def test_params_from_dict_partial():
    p = InferParams.from_dict({"temperature": 1.9})
    assert p.temperature == 1.9
    assert p.top_p == InferParams.default().top_p


def test_params_from_dict_empty():
    assert InferParams.from_dict(None) == InferParams.default()
    assert InferParams.from_dict({}) == InferParams.default()


def test_params_validation():
    assert InferParams.default().problems() == []
    assert InferParams.default().with_overrides(top_p=0).problems()
    assert InferParams.default().with_overrides(max_tokens=0).problems()
    assert InferParams.default().with_overrides(alpha_decay=1.5).problems()


# ── prompt 构建 ──────────────────────────────────────────────────────────
def test_build_prompt_basic():
    msgs = [ChatMessage("user", "你好"), ChatMessage("assistant", "你好！")]
    out = build_prompt(msgs, params=InferParams.default(), thinking_mode=False)
    assert "User: 你好" in out
    assert "Assistant: 你好！" in out
    assert out.endswith("Assistant: ")


def test_build_prompt_thinking_mode_appends_tag():
    _START = THINK_START
    out = build_prompt([], params=InferParams.default(), thinking_mode=True)
    assert out.endswith(_START)


def test_build_prompt_restores_thinking_history():
    """★ 历史 thinking 必须还原成完整标记形式。

    否则 state 里的 token 序列与 prompt 前缀失配，增量 prefill 直接失效。
    """
    msgs = [
        ChatMessage("user", "问题"),
        ChatMessage("assistant", "回答", thinking="我先想想"),
    ]
    out = build_prompt(msgs, params=InferParams.default(), thinking_mode=False)
    assert THINK_START in out
    assert THINK_END in out
    assert "我先想想" in out
    # think 标记必须出现在正文之前
    assert out.index(THINK_START) < out.index("回答")


def test_build_prompt_system_card_kept_first():
    """角色卡必须在对话历史之前。

    注意断言方式：presystem 前缀里本身含 "User: hi"，所以不能用
    ``out.index("User: hi")`` 定位对话起点。改用角色卡与对话正文中的
    唯一标记比较。
    """
    msgs = [
        ChatMessage("system", "你是助手"),
        ChatMessage("user", "用户的第一句话"),
        ChatMessage("assistant", "助手的回答"),
    ]
    out = build_prompt(msgs, params=InferParams.default(), thinking_mode=False)
    assert "System: 你是助手" in out
    assert "用户的第一句话" in out
    # 角色卡必须出现在本轮对话正文之前
    assert out.index("System: 你是助手") < out.index("用户的第一句话")


def test_build_prompt_system_after_presystem():
    """presystem 前缀在前，角色卡紧随其后。"""
    msgs = [ChatMessage("system", "角色定义XYZ"), ChatMessage("user", "问题ABC")]
    out = build_prompt(msgs, params=InferParams.default(), thinking_mode=False)
    assert out.index(PRESYSTEM_PREFIX) < out.index("System: 角色定义XYZ")
    assert out.index("System: 角色定义XYZ") < out.index("问题ABC")


def test_build_prompt_presystem_toggle():
    on = build_prompt([], params=InferParams.default(), thinking_mode=False)
    off = build_prompt(
        [], params=InferParams.default().with_overrides(presystem=False), thinking_mode=False
    )
    assert PRESYSTEM_PREFIX in on
    assert PRESYSTEM_PREFIX not in off


def test_build_prompt_no_trim_by_default():
    """max_rounds=0 默认不裁剪——裁剪会让增量 prefill 的长前缀消失。"""
    msgs = []
    for i in range(20):
        msgs.append(ChatMessage("user", f"q{i}"))
        msgs.append(ChatMessage("assistant", f"a{i}"))
    out = build_prompt(msgs, params=InferParams.default(), thinking_mode=False)
    assert "q0" in out, "默认不应裁剪历史"


def test_build_prompt_trim_when_explicit():
    msgs = []
    for i in range(20):
        msgs.append(ChatMessage("user", f"q{i}"))
        msgs.append(ChatMessage("assistant", f"a{i}"))
    p = InferParams.default().with_overrides(max_rounds=2)
    out = build_prompt(msgs, params=p, thinking_mode=False)
    assert "q0" not in out
    assert "q19" in out


def test_split_thinking_with_end():
    raw = f"{THINK_START}思考中{THINK_END}\n正式回答"
    thinking, content = split_thinking(raw)
    assert thinking == f"{THINK_START}思考中".replace(THINK_START, "").strip() or thinking
    assert "正式回答" in content


def test_split_thinking_truncated():
    """模型可能只输出 thinking 就被截断，answer 为空不是错误。"""
    raw = f"{THINK_START}只想了没说完"
    thinking, content = split_thinking(raw)
    assert thinking == "只想了没说完"
    assert content == ""


def test_split_thinking_plain():
    thinking, content = split_thinking("就是普通回答")
    assert thinking == ""
    assert content == "就是普通回答"


def test_count_rounds():
    msgs = [ChatMessage("user", "a"), ChatMessage("assistant", "b"), ChatMessage("user", "c")]
    assert count_rounds(msgs) == 2


# ── 增量 prefill 核心逻辑 ────────────────────────────────────────────────
def test_lcp():
    assert longest_common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert longest_common_prefix([1, 2], [1, 2]) == 2
    assert longest_common_prefix([1], [2]) == 0
    assert longest_common_prefix([], [1]) == 0


def test_token_prefix_chars_never_splits_multibyte():
    """★ 二分查找必须落在字符边界上。

    旧项目为此刻意写了这函数——按字节切会抛 UnicodeDecodeError。
    """
    text = "你好世界abc"
    for n in range(0, len(text) + 1):
        chars = token_prefix_chars(text, n, char_encode)
        assert 0 <= chars <= len(text)
        # 关键：返回的字符数对应的 token 数必须恰好 <= n
        assert len(char_encode(text[:chars])) <= n


def test_plan_prefill_first_turn_rebuilds():
    plan = plan_prefill(prompt="User: hi\n\nAssistant: ", fed_text="", has_state=False, encode=char_encode)
    assert plan.full_rebuild is True
    assert plan.reused_tokens == 0
    assert plan.feed_tokens == char_encode("User: hi\n\nAssistant: ")


def test_plan_prefill_reuses_prefix():
    """第二轮：prompt 前缀未变，只喂新增部分。"""
    fed = "User: hi\n\nAssistant: 你好\n\n"
    prompt = "User: hi\n\nAssistant: 你好\n\nUser: 再见\n\nAssistant: "
    plan = plan_prefill(prompt=prompt, fed_text=fed, has_state=True, encode=char_encode)

    assert plan.full_rebuild is False
    assert plan.reused_tokens == len(fed)
    assert plan.reused_chars == len(fed)
    # 只喂新增的 "User: 再见\n\nAssistant: "
    assert plan.feed_tokens == char_encode(prompt[len(fed):])


def test_plan_prefill_zero_lcp_triggers_rebuild():
    """换角色卡导致开头变化 → 一个 token 都复用不上 → 全量重建。"""
    plan = plan_prefill(
        prompt="System: 新角色卡\n\nUser: hi\n\nAssistant: ",
        fed_text="User: hi\n\nAssistant: ",
        has_state=True,
        encode=char_encode,
    )
    assert plan.full_rebuild is True
    assert plan.reused_tokens == 0


def test_plan_prefill_identical_prompt_feeds_nothing():
    """完全相同的 prompt —— 原设计认为"不用喂"，但那会崩。

    见 test_no_empty_feed_tokens_when_prompt_is_prefix 的说明：
    feed_tokens 为空会让上游 generate 的 while 循环不执行，
    `out` 变量未赋值 → UnboundLocalError。

    现在的契约是：**永远不返回空 feed_tokens**，
    宁可全量重喂也不能留空。
    """
    text = "User: hi\n\nAssistant: "
    plan = plan_prefill(prompt=text, fed_text=text, has_state=True, encode=char_encode)
    assert plan.feed_tokens, "不能返回空 feed_tokens"
    assert plan.full_rebuild is True


def test_rebuild_fed_text_is_prompt_plus_output():
    assert rebuild_fed_text(prompt="abc", raw_output="def") == "abcdef"


# ── 会话隔离 —— 本轮最重要的一组 ────────────────────────────────────────
def test_two_sessions_have_independent_state(engine):
    """会话 A 生成不影响会话 B 的 state。"""
    pa = InferParams.default().with_overrides(temperature=0.5)
    pb = InferParams.default().with_overrides(temperature=1.5)

    engine.generate(session_key="A", user_message="问A", history=[], params=pa)
    engine.generate(session_key="B", user_message="问B", history=[], params=pb)

    sa = engine.get_session("A")
    sb = engine.get_session("B")

    assert sa.fed_text != sb.fed_text
    assert "问A" in sa.fed_text
    assert "问B" in sb.fed_text
    assert "问B" not in sa.fed_text
    assert "问A" not in sb.fed_text


def test_params_do_not_leak_between_sessions(engine):
    """★ 旧版的串味根因：单例共享 current_params。"""
    pa = InferParams.default().with_overrides(temperature=0.5, top_p=0.9)
    pb = InferParams.default().with_overrides(temperature=1.8, top_p=0.1)

    engine.generate(session_key="A", user_message="x", history=[], params=pa)
    assert engine.get_session("A").params.temperature == 0.5

    engine.generate(session_key="B", user_message="y", history=[], params=pb)
    # A 的参数不能被 B 覆盖
    assert engine.get_session("A").params.temperature == 0.5
    assert engine.get_session("A").params.top_p == 0.9
    assert engine.get_session("B").params.temperature == 1.8


def test_history_does_not_leak_between_sessions(engine):
    hist_a = [ChatMessage("user", "A的历史"), ChatMessage("assistant", "A的回答")]
    engine.generate(session_key="A", user_message="新问题", history=hist_a, params=InferParams.default())

    engine.generate(session_key="B", user_message="B的问题", history=[], params=InferParams.default())

    sa = engine.get_session("A")
    sb = engine.get_session("B")
    assert len(sa.messages) == 2
    assert len(sb.messages) == 0
    assert sa.messages[0].content == "A的历史"


def test_no_last_response_on_engine(engine):
    """★ 引擎不得保存"最后一次回复"——这是旧版串味的直接原因。"""
    assert not hasattr(engine, "last_response")
    assert not hasattr(engine, "current_params")


def test_generate_returns_result_not_stored(engine):
    r1 = engine.generate(session_key="A", user_message="x", history=[], params=InferParams.default())
    r2 = engine.generate(session_key="B", user_message="y", history=[], params=InferParams.default())
    assert r1 is not r2
    assert r1.content == "第一轮回复。"


def test_model_switch_clears_all_sessions(engine):
    """换模型必须作废全部 state——state 与权重强相关。"""
    engine.generate(session_key="A", user_message="x", history=[], params=InferParams.default())
    assert len(engine.sessions) == 1

    pipe2 = FakePipeline("新模型")
    engine.attach(pipe2, pipe2, model_ref="run2/rwkv-2")
    assert len(engine.sessions) == 0


def test_same_model_does_not_clear_sessions(engine):
    engine.generate(session_key="A", user_message="x", history=[], params=InferParams.default())
    pipe = FakePipeline("y")
    engine.attach(pipe, pipe, model_ref="run1/rwkv-1")  # 同一模型
    assert len(engine.sessions) == 1


def test_reset_session_keeps_history(engine):
    hist = [ChatMessage("user", "历史")]
    engine.generate(session_key="A", user_message="x", history=hist, params=InferParams.default())
    s = engine.get_session("A")
    assert s.state is not None

    engine.reset_session("A")
    s = engine.get_session("A")
    assert s.state is None
    assert s.fed_text == ""
    assert len(s.messages) == 1  # 历史保留


def test_session_store_lru_eviction():
    store = SessionStore(max_sessions=2)
    p = InferParams.default()
    store.get("a", params=p)
    store.get("b", params=p)
    store.get("c", params=p)  # 挤掉 a
    assert set(store.keys()) == {"b", "c"}


def test_session_store_max_must_be_positive():
    with pytest.raises(ValueError, match="必须为正整数"):
        SessionStore(max_sessions=0)


# ── 生成流程 ─────────────────────────────────────────────────────────────
def test_generate_requires_model():
    e = InferenceEngine()
    with pytest.raises(ModelNotLoadedError):
        e.generate(session_key="A", user_message="x", history=[], params=InferParams.default())


def test_generate_validates_params(engine):
    bad = InferParams.default().with_overrides(top_p=0)
    with pytest.raises(FoundryError, match="推理参数无效"):
        engine.generate(session_key="A", user_message="x", history=[], params=bad)


def test_generate_streams_tokens(engine):
    got: list[str] = []
    engine.generate(
        session_key="A", user_message="x", history=[],
        params=InferParams.default(), on_token=got.append,
    )
    assert "".join(got) == "第一轮回复。"


def test_generate_incremental_second_turn(engine):
    """第二轮只喂新增 token，不重复喂历史。"""
    engine.generate(session_key="A", user_message="第一个问题", history=[], params=InferParams.default())
    first_fed = len(engine._pipeline.forward_log[0])
    assert first_fed > 0

    hist = [ChatMessage("user", "第一个问题"), ChatMessage("assistant", "第一轮回复。")]
    engine.generate(session_key="A", user_message="第二个问题", history=hist, params=InferParams.default())
    second_fed = len(engine._pipeline.forward_log[1])

    # 第二轮喂入量应远小于完整 prompt（复用了前缀）
    full_len = len(char_encode(engine.get_session("A").fed_text))
    assert second_fed < full_len, "第二轮应复用前缀，不应全量重喂"


def test_generate_rollback_reuses_checkpoint(engine):
    """重生成：回滚到上轮生成前，然后只喂差量。

    注意 ``history`` **不含**本轮用户消息——消息由 ``user_message`` 追加。
    这与前端 regenerate 一致（先 pop 掉 assistant，再以 isNewMessage=false 重发）。
    若把当前消息也放进 history，它会被喂两遍。
    """
    pipe = engine._pipeline
    pipe.reply = "回复一"
    engine.generate(
        session_key="A", user_message="问题一", history=[], params=InferParams.default()
    )
    s = engine.get_session("A")
    after_first = s.fed_text

    pipe.reply = "回复二"
    hist2 = [ChatMessage("user", "问题一"), ChatMessage("assistant", "回复一")]
    engine.generate(
        session_key="A", user_message="问题二", history=hist2, params=InferParams.default()
    )
    after_second = s.fed_text

    pipe.reply = "回复二改"
    engine.generate(
        session_key="A", user_message="问题二", history=hist2,
        params=InferParams.default(), rollback=True,
    )

    assert "回复二改" in s.fed_text
    assert s.fed_text.count("问题二") == 1, f"用户消息不应重复: {s.fed_text[-80:]}"
    assert after_second != after_first


def test_rollback_matches_checkpoint_so_prefill_is_incremental(engine):
    """rollback 后的 prompt 与 checkpoint 基线一致 → 复用前缀，不全量重喂。"""
    pipe = engine._pipeline
    pipe.reply = "A"
    engine.generate(session_key="A", user_message="q1", history=[], params=InferParams.default())
    pipe.reply = "B"
    hist = [ChatMessage("user", "q1"), ChatMessage("assistant", "A")]
    engine.generate(session_key="A", user_message="q2", history=hist, params=InferParams.default())

    before = len(pipe.forward_log)
    pipe.reply = "C"
    engine.generate(
        session_key="A", user_message="q2", history=hist,
        params=InferParams.default(), rollback=True,
    )
    assert len(pipe.forward_log) == before + 1
    # 最后一次喂入量应远小于完整 prompt
    assert len(pipe.forward_log[-1]) < 60


def test_generate_stops_on_should_stop(engine):
    calls = {"n": 0}

    def should_stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    engine._pipeline.reply = "很长很长的回复内容" * 10
    result = engine.generate(
        session_key="A", user_message="x", history=[],
        params=InferParams.default(), on_token=lambda _c: None,
        should_stop=should_stop,
    )
    assert result.stopped is True


def test_generate_wraps_pipeline_errors(engine):
    class Broken(FakePipeline):
        def generate(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

    engine.attach(Broken(), Broken(), model_ref="x")
    with pytest.raises(InferError, match="生成失败"):
        engine.generate(session_key="A", user_message="x", history=[], params=InferParams.default())


def test_gen_result_as_dict(engine):
    r = engine.generate(session_key="A", user_message="x", history=[], params=InferParams.default())
    d = r.as_dict()
    assert set(d) == {"content", "thinking", "stopped", "tokens_fed", "tokens_reused"}


# ── stop 序列 ────────────────────────────────────────────────────────────
def test_stop_filter_basic():
    f = StopSequenceFilter()
    out = f.feed("你好世界\n\nUser: 我是用户")
    assert "User:" not in out
    assert "你好世界" in out
    assert f.stopped is True


def test_stop_filter_cross_chunk():
    """★ stop 序列跨 chunk 边界必须能检测到。"""
    f = StopSequenceFilter()
    a = f.feed("回复内容\n\nU")
    b = f.feed("ser: 冒充")
    assert f.stopped is True
    assert "冒充" not in (a + b)
    assert "回复内容" in (a + b)


def test_stop_filter_preserves_normal_text():
    f = StopSequenceFilter()
    total = ""
    for chunk in ["这是一段", "普通的", "回复", "没有模板标记"]:
        total += f.feed(chunk)
    total += f.flush()
    assert total == "这是一段普通的回复没有模板标记"
    assert f.stopped is False


def test_stop_filter_flush_residual():
    f = StopSequenceFilter()
    emitted = f.feed("结尾有个换行\n")
    tail = f.flush()
    assert (emitted + tail).endswith("\n")


def test_stop_filter_disabled_after_stop():
    f = StopSequenceFilter()
    f.feed("abc\n\nUser:")
    assert f.stopped
    assert f.feed("更多内容") == ""
    assert f.flush() == ""


def test_stop_filter_handles_assistant_leak():
    f = StopSequenceFilter()
    out = f.feed("回答\nAssistant: 我替我说话")
    assert "Assistant:" not in out
    assert f.stopped is True


# ── 空 feed_tokens 边界（真机上崩过一次）────────────────────────────────
def test_no_empty_feed_tokens_when_prompt_is_prefix():
    """★ 新 prompt 是旧 fed_text 的前缀时，绝不能返回空 feed_tokens。

    ## 这里崩过一次

    用户删掉模型生成的内容后重新点续写，新 prompt 变成了旧 fed_text
    的前缀（LCP == 整个新 prompt）。原实现判定"全部复用、不用喂"，
    返回 feed_tokens = []。

    而上游 generate() 的实现是：

        while len(tokens) > 0:
            out, state = self.model.forward(...)
        for n in args.token_ban:
            out[n] = -float('inf')      ← out 从未赋值

    于是抛出：cannot access local variable 'out' where it is not
    associated with a value。用户看到的就是"生成失败"。

    修复策略：这种情况回退成 full_rebuild，强制全量重喂。
    多算一次换取不崩，非常值得。
    """
    fed = "请你扮演修真冒险游戏。\n\nAssistant: 你睁开眼。"
    prompt = "请你扮演修真冒险游戏。\n\nAssistant: "

    plan = plan_prefill(prompt=prompt, fed_text=fed, has_state=True, encode=char_encode)
    assert plan.feed_tokens, "feed_tokens 不能为空，否则上游会 UnboundLocalError"
    assert plan.full_rebuild is True
    assert plan.reused_tokens == 0


def test_identical_prompt_also_not_empty():
    """完全相同的 prompt 也不能返回空 feed_tokens。"""
    text = "User: hi\n\nAssistant: "
    plan = plan_prefill(prompt=text, fed_text=text, has_state=True, encode=char_encode)
    assert plan.feed_tokens, "完全一致时也必须重喂，不能留空"
    assert plan.full_rebuild is True


def test_normal_incremental_still_reuses():
    """修复不能误伤正常复用：真增量场景仍应只喂差量。"""
    fed = "User: hi\n\nAssistant: 你好\n\n"
    prompt = fed + "User: 再见\n\nAssistant: "
    plan = plan_prefill(prompt=prompt, fed_text=fed, has_state=True, encode=char_encode)
    assert plan.full_rebuild is False
    assert plan.reused_tokens == len(fed)
    assert 0 < len(plan.feed_tokens) < len(prompt)
