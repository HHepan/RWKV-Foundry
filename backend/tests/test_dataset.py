"""foundry.domain.dataset 的测试。

重中之重是**分页性能契约**：这里曾真实写出过一个比旧版还慢两倍的实现
（308ms vs 旧版 171ms），原因是「为了填 total 而遍历全文件」。
本文件的性能测试就是为了让这个退化不可能悄悄回归。
"""

from __future__ import annotations

import json
import time

import pytest

from foundry.domain.dataset import (
    DatasetError,
    count_rounds,
    delete_row,
    iter_rows,
    parse_conversations,
    read_page,
    scan_stats,
    update_row,
)


def _write(path, texts):
    path.write_text(
        "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in texts) + "\n",
        encoding="utf-8",
    )
    return path


def _multi(n: int) -> str:
    return (
        "User: 问题一\n\nAssistant: 回答一\n\n"
        f"User: 问题二 {n}\n\nAssistant: 回答二 {n}"
    )


# ── 解析 ─────────────────────────────────────────────────────────────────
def test_parse_single_round():
    pairs = parse_conversations("User: 你好\n\nAssistant: 你好！")
    assert pairs == (("user", "你好"), ("assistant", "你好！"))
    assert count_rounds(pairs) == 1


def test_parse_multi_round():
    pairs = parse_conversations(_multi(1))
    assert len(pairs) == 4
    assert count_rounds(pairs) == 2


def test_parse_empty():
    assert parse_conversations("") == ()
    assert count_rounds(()) == 0


# ── 迭代 ─────────────────────────────────────────────────────────────────
def test_iter_rows_skips_blank_lines(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text(
        json.dumps({"text": "User: a\n\nAssistant: b"}) + "\n\n\n"
        + json.dumps({"text": "User: c\n\nAssistant: d"}) + "\n",
        encoding="utf-8",
    )
    rows = list(iter_rows(p))
    assert len(rows) == 2
    assert rows[1].text.startswith("User: c")


def test_iter_rows_survives_bad_json(tmp_path):
    """坏行退化为纯文本，不丢弃——用户需要看到坏数据才能修。"""
    p = tmp_path / "b.jsonl"
    p.write_text('not json at all\n{"text":"User: a"}\n', encoding="utf-8")
    rows = list(iter_rows(p))
    assert len(rows) == 2
    assert rows[0].text == "not json at all"


def test_iter_rows_missing_file(tmp_path):
    with pytest.raises(DatasetError, match="读取数据集失败"):
        list(iter_rows(tmp_path / "nope.jsonl"))


# ── 分页正确性 ───────────────────────────────────────────────────────────
def test_read_page_first(tmp_path):
    p = _write(tmp_path / "c.jsonl", [f"User: q{i}\n\nAssistant: a{i}" for i in range(25)])
    r = read_page(p, page=1, size=10, total_override=25)
    assert len(r["data"]) == 10
    assert r["data"][0]["id"] == 1
    assert r["has_more"] is True
    assert r["total"] == 25


def test_read_page_middle(tmp_path):
    p = _write(tmp_path / "d.jsonl", [f"User: q{i}\n\nAssistant: a{i}" for i in range(25)])
    r = read_page(p, page=2, size=10, total_override=25)
    assert [x["id"] for x in r["data"]] == list(range(11, 21))


def test_read_page_last_partial(tmp_path):
    p = _write(tmp_path / "e.jsonl", [f"User: q{i}\n\nAssistant: a{i}" for i in range(25)])
    r = read_page(p, page=3, size=10, total_override=25)
    assert [x["id"] for x in r["data"]] == list(range(21, 26))
    assert r["has_more"] is False


def test_read_page_out_of_range(tmp_path):
    p = _write(tmp_path / "f.jsonl", ["User: a"])
    r = read_page(p, page=99, size=10, total_override=1)
    assert r["data"] == []
    assert r["has_more"] is False


def test_read_page_exact_boundary(tmp_path):
    """恰好整页时 has_more 应为 False。"""
    p = _write(tmp_path / "g.jsonl", [f"User: q{i}" for i in range(10)])
    r = read_page(p, page=1, size=10, total_override=10)
    assert len(r["data"]) == 10
    assert r["has_more"] is False


def test_read_page_filtered_single(tmp_path):
    texts = [_multi(0), "User: a\n\nAssistant: b", _multi(2), "User: c\n\nAssistant: d"]
    p = _write(tmp_path / "h.jsonl", texts)
    r = read_page(p, page=1, size=10, rounds_filter="single")
    assert r["total"] == 2  # 只有两行是单轮


def test_read_page_filtered_multi(tmp_path):
    texts = [_multi(0), "User: a\n\nAssistant: b", _multi(2)]
    p = _write(tmp_path / "i.jsonl", texts)
    r = read_page(p, page=1, size=10, rounds_filter="multi")
    assert r["total"] == 2


def test_read_page_invalid_args(tmp_path):
    p = _write(tmp_path / "j.jsonl", ["User: a"])
    with pytest.raises(DatasetError, match="page 必须"):
        read_page(p, page=0)
    with pytest.raises(DatasetError, match="size 必须"):
        read_page(p, size=0)
    with pytest.raises(DatasetError, match="rounds_filter 非法"):
        read_page(p, rounds_filter="bogus")


# ── 分页性能契约（防止退化回归）───────────────────────────────────────────
@pytest.fixture(scope="module")
def big_dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("big")
    p = d / "big.jsonl"
    rows = [
        json.dumps(
            {
                "text": f"User: 第{i}个问题，请解释注意力和循环网络在长序列上的差异。\n\n"
                f"Assistant: 注意力是O(n^2)全局加权，RNN是O(n)串行传递隐藏状态，各有利弊。"
            },
            ensure_ascii=False,
        )
        for i in range(50000)
    ]
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def test_first_page_is_fast(big_dataset):
    """第一页必须远快于全扫。

    契约：< 50ms。实测 0.2ms。曾出现的退化实现是 308ms
    （为填 total 遍历全文件，比旧版 171ms 还慢）。
    """
    t = time.perf_counter()
    r = read_page(big_dataset, page=1, size=10, total_override=50000)
    elapsed_ms = (time.perf_counter() - t) * 1000

    assert len(r["data"]) == 10
    assert elapsed_ms < 50, f"第一页耗时 {elapsed_ms:.1f}ms，超出 50ms 契约——可能又退化成全扫了"


def test_deep_page_is_fast(big_dataset):
    """深分页也不该做 JSON 解析。契约 < 100ms，实测 6ms。"""
    t = time.perf_counter()
    r = read_page(big_dataset, page=2000, size=10, total_override=50000)
    elapsed_ms = (time.perf_counter() - t) * 1000

    assert [x["id"] for x in r["data"]] == list(range(19991, 20001))
    assert elapsed_ms < 100, f"深分页耗时 {elapsed_ms:.1f}ms，超出 100ms 契约"


def test_total_none_when_no_index(big_dataset):
    """无索引时不谎报 total。"""
    r = read_page(big_dataset, page=1, size=10)
    assert r["total"] is None
    assert len(r["data"]) == 10


# ── 统计 ─────────────────────────────────────────────────────────────────
def test_scan_stats(tmp_path):
    p = _write(tmp_path / "k.jsonl", [_multi(0), "User: a\n\nAssistant: b", _multi(2)])
    s = scan_stats(p)
    assert s.line_count == 3
    assert s.rounds_multi == 2
    assert s.rounds_single == 1


# ── 编辑 ─────────────────────────────────────────────────────────────────
def test_update_row(tmp_path):
    p = _write(tmp_path / "l.jsonl", ["User: a", "User: b", "User: c"])
    update_row(p, 2, "User: replaced")
    texts = [json.loads(line)["text"] for line in p.read_text(encoding="utf-8").splitlines()]
    assert texts == ["User: a", "User: replaced", "User: c"]


def test_update_row_out_of_range(tmp_path):
    p = _write(tmp_path / "m.jsonl", ["User: a"])
    with pytest.raises(DatasetError, match="超出范围"):
        update_row(p, 5, "x")


def test_delete_row(tmp_path):
    p = _write(tmp_path / "n.jsonl", ["User: a", "User: b", "User: c"])
    delete_row(p, 2)
    texts = [json.loads(line)["text"] for line in p.read_text(encoding="utf-8").splitlines()]
    assert texts == ["User: a", "User: c"]


def test_delete_row_out_of_range(tmp_path):
    p = _write(tmp_path / "o.jsonl", ["User: a"])
    with pytest.raises(DatasetError, match="超出范围"):
        delete_row(p, 0)
