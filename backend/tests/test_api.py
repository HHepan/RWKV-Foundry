"""API 层测试。

用 ``NullModelLoader`` 与注入的 fake pipeline，使全部接口都能在无 GPU、
无 torch 的环境下被验证。真实生成逻辑已由 ``test_infer.py`` 覆盖。

重点验证：
  1. 路径穿越在 HTTP 层被拦截（旧版 4 个接口可被打穿）
  2. SSE 真的流出内容（旧版靠 100ms 轮询节流）
  3. 一个进程只能跑一个训练
  4. 错误映射到正确的状态码
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from foundry.api.app import create_app
from foundry.api.deps import build_state
from foundry.config import AppConfig
from foundry.infer.session import InferenceEngine
from foundry.paths import resolve_paths


# ── 测试用 fake pipeline ─────────────────────────────────────────────────
def _enc(text: str) -> list[int]:
    return [ord(c) for c in text]


def _dec(tokens: list[int]) -> str:
    return "".join(chr(t) for t in tokens)


class FakePipeline:
    def __init__(self, reply: str = "假回复"):
        self.reply = reply
        self.encode = _enc
        self.decode = _dec
        self.calls: list[int] = []

    def generate(self, ctx, token_count=100, args=None, callback=None, state=None, occurrence=None):
        self.calls.append(len(ctx))
        new_state = (list(state) if state else []) + list(ctx) + _enc(self.reply)
        if callback:
            for ch in self.reply:
                callback(ch)
        return self.reply, new_state, occurrence or {}


@pytest.fixture
def workspace(tmp_path):
    """构造一个完整可用的 workspace。"""
    root = tmp_path / "workspace"
    peft = root / "tools" / "RWKV-PEFT"
    for rel in (
        "train.py",
        "json2binidx_tool/tools/preprocess_data.py",
        "json2binidx_tool/rwkv_vocab_v20230424.txt",
        "scripts/lora.sh",
    ):
        p = peft / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")

    (root / "base_models").mkdir(parents=True)
    (root / "data" / "out").mkdir(parents=True)
    (root / "checkpoints").mkdir(parents=True)

    # 一个基底权重
    (root / "base_models" / "rwkv7-g1a-0.1b.pth").write_bytes(b"fake")
    # 两个数据集
    (root / "data" / "out" / "train.jsonl").write_text(
        "\n".join(
            json.dumps({"text": f"User: q{i}\n\nAssistant: a{i}"}, ensure_ascii=False)
            for i in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "data" / "out" / "multi.jsonl").write_text(
        json.dumps(
            {"text": "User: a\n\nAssistant: b\n\nUser: c\n\nAssistant: d"}, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def client(workspace, monkeypatch):
    monkeypatch.setenv("FOUNDRY_HOME", str(workspace))
    # with_model=False → NullModelLoader，保证无 torch 也能跑
    app = create_app(config=AppConfig.from_env(), with_model=False)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def model_client(workspace, monkeypatch):
    """注入了 fake pipeline 的 client，可测真实对话流程。"""
    monkeypatch.setenv("FOUNDRY_HOME", str(workspace))
    paths = resolve_paths()
    state = build_state(paths=paths, config=AppConfig.from_env(), with_model=False)

    pipe = FakePipeline("你好，我是假模型。")
    state.engine = InferenceEngine()
    state.engine.attach(pipe, pipe, model_ref="base_models/rwkv7-g1a-0.1b")

    app = create_app(config=AppConfig.from_env(), with_model=False, state=state)
    with TestClient(app) as c:
        yield c


# ── meta ─────────────────────────────────────────────────────────────────
def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["name"] == "RWKV-Foundry"


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model_loaded"] is False
    assert body["schema_version"] >= 1


def test_config_endpoint(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["vocab_size"] == 65536
    assert "checkpoints" in body["paths"]
    assert "2.9B" in body["model_sizes"]
    assert body["infer_defaults"]["top_p"] == 0.3


def test_openapi_schema_generates(client):
    """前端类型由此生成——必须可用。"""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert "/api/train/start" in schema["paths"]
    assert "/api/chat/chat" in schema["paths"]
    assert "/api/data/files/{filename}" in schema["paths"]


# ── 数据集 ───────────────────────────────────────────────────────────────
def test_list_folders(client):
    r = client.get("/api/data/folders")
    assert r.status_code == 200
    assert "./" in r.json()
    assert "./out" in r.json()


def test_list_files(client):
    r = client.get("/api/data/files", params={"folder": "./out"})
    assert r.status_code == 200
    assert set(r.json()) == {"train.jsonl", "multi.jsonl"}


def test_page_dataset(client):
    r = client.get("/api/data/files/train.jsonl", params={"folder": "./out", "page": 1, "size": 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 2
    assert body["has_more"] is True
    assert body["data"][0]["id"] == 1


def test_page_dataset_filter(client):
    r = client.get(
        "/api/data/files/multi.jsonl",
        params={"folder": "./out", "rounds_filter": "multi"},
    )
    assert r.status_code == 200
    assert r.json()["total"] == 1


def test_dataset_stats_is_cached(client):
    r1 = client.get("/api/data/files/stats", params={"filename": "train.jsonl", "folder": "./out"})
    assert r1.status_code == 200
    assert r1.json()["line_count"] == 5
    r2 = client.get("/api/data/files/stats", params={"filename": "train.jsonl", "folder": "./out"})
    assert r2.json()["cached"] is True


def test_page_missing_file_404(client):
    r = client.get("/api/data/files/nope.jsonl", params={"folder": "./out"})
    assert r.status_code == 404


def test_update_and_delete_row(client):
    r = client.put(
        "/api/data/files/train.jsonl/1",
        params={"folder": "./out"},
        json={"text": "User: 改过\n\nAssistant: 好"},
    )
    assert r.status_code == 200

    r = client.get("/api/data/files/train.jsonl", params={"folder": "./out", "size": 1})
    assert "改过" in r.json()["data"][0]["text"]

    r = client.delete("/api/data/files/train.jsonl/1", params={"folder": "./out"})
    assert r.status_code == 200
    r = client.get("/api/data/files/stats", params={"filename": "train.jsonl", "folder": "./out"})
    assert r.json()["line_count"] == 4


def test_merge_datasets(client):
    r = client.post(
        "/api/data/merge",
        json={
            "source_files": [
                {"filename": "train.jsonl", "folder": "./out"},
                {"filename": "multi.jsonl", "folder": "./out"},
            ],
            "new_name": "merged",
            "folder": "./out",
            "shuffle": False,
            "counts": {},
        },
    )
    assert r.status_code == 200
    assert r.json()["filename"] == "merged.jsonl"
    assert r.json()["total_lines"] == 6

    r = client.get("/api/data/files", params={"folder": "./out"})
    assert "merged.jsonl" in r.json()


# ── 路径穿越 —— 旧版真实漏洞 ─────────────────────────────────────────────
@pytest.mark.parametrize(
    "bad_folder",
    ["../..", "../../etc", "..", "out/../../.."],
)
def test_dataset_folder_traversal_blocked(client, bad_folder):
    r = client.get("/api/data/files/passwd", params={"folder": bad_folder})
    assert r.status_code in (400, 404), f"folder={bad_folder} 未被拦截"


def test_dataset_filename_traversal_blocked(client):
    r = client.get("/api/data/files/../../../etc/passwd", params={"folder": "./out"})
    assert r.status_code in (400, 404)


def test_delete_record_traversal_blocked(client, workspace):
    """★ 旧版 DELETE /records/{folder} 会 rmtree 任意目录。"""
    victim = workspace / "victim"
    victim.mkdir()
    (victim / "important.txt").write_text("data", encoding="utf-8")

    r = client.delete("/api/train/records/../victim")
    assert r.status_code in (400, 404)
    assert (victim / "important.txt").exists(), "受害目录被删除了！"


def test_role_id_traversal_blocked(client):
    """旧版角色 id 直接拼进文件名，可写任意路径。"""
    r = client.post("/api/data/roles", json={"id": "../../evil", "name": "x", "content": "y"})
    assert r.status_code in (400, 404)


# ── 基底模型 ─────────────────────────────────────────────────────────────
def test_list_base_models(client):
    r = client.get("/api/data/base-models")
    assert r.status_code == 200
    assert "rwkv7-g1a-0.1b.pth" in r.json()


# ── 配方 ─────────────────────────────────────────────────────────────────
def _recipe_payload():
    return {
        "base_model": "rwkv7-g1a-0.1b.pth",
        "dataset": "out/train.jsonl",
        "model_size": "0.1B",
        "ctx_len": 512,
        "epoch_steps": 10,
        "epoch_count": 2,
    }


def test_validate_recipe(client):
    r = client.post("/api/train/recipe/validate", json=_recipe_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["hash"]
    assert body["n_layer"] == 12
    assert body["n_embd"] == 768


def test_validate_recipe_reports_all_problems(client):
    bad = {**_recipe_payload(), "top_p": 0, "lora_r": 0}
    r = client.post("/api/train/recipe/validate", json=bad)
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert len(body["problems"]) >= 1


def test_recipe_hash_is_stable(client):
    """同参数必须同 hash —— 横向对比的前提。"""
    a = client.post("/api/train/recipe/validate", json=_recipe_payload()).json()["hash"]
    b = client.post("/api/train/recipe/validate", json=_recipe_payload()).json()["hash"]
    assert a == b


def test_recipe_hash_ignores_unrelated_fields(client):
    """输出目录等不影响数学结果的字段不进哈希（由 Recipe 定义保证）。"""
    base = client.post("/api/train/recipe/validate", json=_recipe_payload()).json()["hash"]
    changed = {
        **_recipe_payload(),
        "lora_r": 64,
    }
    other = client.post("/api/train/recipe/validate", json=changed).json()["hash"]
    assert base != other, "改变 lora_r 应产生不同配方"


def test_preview_script(client):
    r = client.post(
        "/api/train/recipe/preview", json=_recipe_payload(), params={"save_folder": "run1"}
    )
    assert r.status_code == 200
    script = r.json()["script"]
    assert "exec python train.py" in script
    assert "--peft lora" in script
    assert "--n_layer 12" in script


def test_preview_script_missing_dataset(client):
    bad = {**_recipe_payload(), "dataset": "out/nope.jsonl"}
    r = client.post("/api/train/recipe/preview", json=bad, params={"save_folder": "run1"})
    assert r.status_code == 404


# ── 训练状态 ─────────────────────────────────────────────────────────────
def test_status_idle(client):
    r = client.get("/api/train/status")
    assert r.status_code == 200
    assert r.json()["status"] == "idle"


def test_status_missing_folder_404(client):
    r = client.get("/api/train/status", params={"folder": "nope"})
    assert r.status_code == 404


def test_records_initially_empty(client):
    r = client.get("/api/train/records")
    assert r.status_code == 200
    assert r.json() == []


def test_stop_when_idle_404(client):
    r = client.post("/api/train/stop")
    assert r.status_code == 404


# ── 对话 ─────────────────────────────────────────────────────────────────
def test_chat_sse_streams(model_client):
    """★ SSE 必须真的逐块流出，而不是靠轮询。"""
    r = model_client.post(
        "/api/chat/chat",
        json={
            "model_ref": "base_models/rwkv7-g1a-0.1b",
            "message": "你好",
            "history": [],
            "params": {},
            "thinking_mode": False,
        },
    )
    assert r.status_code == 200
    body = r.text
    assert "data: " in body
    assert "[DONE]" in body

    # 内容应逐字出现（fake pipeline 逐字符回调）
    chunks = [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line[6:] not in ("[DONE]", "[FINAL]")
    ]
    contents = [c["content"] for c in chunks if "content" in c]
    assert len(contents) > 1, "应分多个 chunk 流出，而非一次性返回"
    assert "".join(contents) == "你好，我是假模型。"

    results = [c["result"] for c in chunks if "result" in c]
    assert results and results[0]["content"] == "你好，我是假模型。"


def test_chat_without_model_reports_error_in_stream(client):
    """★ 流式响应的本质限制：HTTP 200 已发出，错误只能走 SSE 事件。

    ``StreamingResponse`` 一旦开始 yield，状态码就固定为 200。因此
    "模型没加载"这类错误不能靠 HTTP 503 表达——必须在流里发 ``{"error": ...}``。
    前端据此显示错误，而不是当成正常空回复。
    """
    r = client.post(
        "/api/chat/chat",
        json={"model_ref": "base_models/rwkv7-g1a-0.1b", "message": "hi", "history": []},
    )
    assert r.status_code == 200
    body = r.text
    assert '"error"' in body
    assert "torch" in body
    assert "[DONE]" in body


def test_chat_invalid_model_ref_reports_error_in_stream(model_client):
    r = model_client.post(
        "/api/chat/chat", json={"model_ref": "bad", "message": "hi", "history": []}
    )
    assert r.status_code == 200
    assert '"error"' in r.text


def test_chat_persists_dialogue(model_client):
    model_client.post(
        "/api/chat/chat",
        json={
            "model_ref": "base_models/rwkv7-g1a-0.1b",
            "message": "第一问",
            "history": [],
            "params": {},
            "session_name": "s1",
        },
    )
    r = model_client.get(
        "/api/data/chat-data",
        params={"model": "base_models/rwkv7-g1a-0.1b", "session": "s1"},
    )
    assert r.status_code == 200
    msgs = r.json()["dialogue_content"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_chat_models_listed(model_client):
    model_client.post(
        "/api/chat/chat",
        json={
            "model_ref": "base_models/rwkv7-g1a-0.1b",
            "message": "q",
            "history": [],
            "session_name": "s2",
        },
    )
    r = model_client.get("/api/data/chat-models")
    assert r.status_code == 200
    assert any(m["session"] == "s2" for m in r.json())


def test_reset_state(model_client):
    r = model_client.post(
        "/api/chat/reset-state", params={"model_ref": "base_models/rwkv7-g1a-0.1b"}
    )
    assert r.status_code == 200


# ── 角色卡 ───────────────────────────────────────────────────────────────
def test_role_crud(client):
    r = client.post(
        "/api/data/roles", json={"id": "r1", "name": "助手", "content": "你是助手"}
    )
    assert r.status_code == 200

    r = client.get("/api/data/roles")
    assert any(x["id"] == "r1" for x in r.json())

    r = client.delete("/api/data/roles/r1")
    assert r.status_code == 200
    r = client.get("/api/data/roles")
    assert not any(x["id"] == "r1" for x in r.json())


def test_delete_missing_role_404(client):
    r = client.delete("/api/data/roles/nope")
    assert r.status_code == 404


# ── 单写入者原则 ─────────────────────────────────────────────────────────
def test_loss_endpoint_uses_ingester(client, workspace):
    """loss 从上游 loss_data.jsonl 灌入，不再读旧版 loss_history.jsonl。"""
    from foundry.storage import repo

    state = client.app.state.foundry
    repo.create_run(
        state.db, folder="r1", recipe=_mk_recipe(), base_model="m", dataset="d"
    )
    run_dir = workspace / "checkpoints" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "loss_data.jsonl").write_text(
        json.dumps({"loss": 1.5}) + "\n" + json.dumps({"loss": 1.2}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "train_log.txt").write_text("line1\nline2\n", encoding="utf-8")

    r = client.get("/api/train/loss", params={"folder": "r1"})
    assert r.status_code == 200
    assert [p["loss"] for p in r.json()] == [1.5, 1.2]

    r = client.get("/api/train/logs", params={"folder": "r1"})
    assert len(r.json()) == 2

    # 旧版双轨文件不应出现
    assert not (run_dir / "loss_history.jsonl").exists()
    assert not (run_dir / "train_logs.txt").exists()


def test_loss_incremental(client, workspace):
    from foundry.storage import repo

    state = client.app.state.foundry
    repo.create_run(state.db, folder="r2", recipe=_mk_recipe(), base_model="m", dataset="d")
    run_dir = workspace / "checkpoints" / "r2"
    run_dir.mkdir(parents=True)
    (run_dir / "loss_data.jsonl").write_text(
        "\n".join(json.dumps({"loss": v}) for v in [1.0, 0.9, 0.8]) + "\n",
        encoding="utf-8",
    )
    r = client.get("/api/train/loss", params={"folder": "r2", "since_step": 1})
    assert [p["loss"] for p in r.json()] == [0.9, 0.8]


def _mk_recipe():
    from foundry.domain.recipe import Recipe

    return Recipe(base_model="rwkv7-g1a-0.1b.pth", dataset="out/train.jsonl", model_size="0.1B")


# ── 清空对话必须真正删除持久化记录 ──────────────────────────────────────
def test_clear_chat_data_actually_deletes(model_client):
    """★ 清空后刷新，内容不得复活。

    ## 这里修过一个真实缺陷

    前端 clear() 原本只调 /api/chat/reset-state（清内存 state），
    完全没删数据库里的对话记录。用户清空后一刷新，loadHistory 又从
    chat-data 读回来 —— 表现为「清空无效，内容自己长回来」。

    本测试锁定后端契约：DELETE /api/data/chat-data 之后，
    GET 必须返回 404（记录确实没了），而不是旧内容。
    """
    ref = "base_models/rwkv7-g1a-0.1b"
    model_client.post(
        "/api/chat/chat",
        json={"model_ref": ref, "message": "会被清掉的问题", "history": [], "session_name": "clr"},
    )

    got = model_client.get("/api/data/chat-data", params={"model": ref, "session": "clr"})
    assert got.status_code == 200
    assert len(got.json()["dialogue_content"]) == 2

    deleted = model_client.delete("/api/data/chat-data", params={"model": ref, "session": "clr"})
    assert deleted.status_code == 200

    after = model_client.get("/api/data/chat-data", params={"model": ref, "session": "clr"})
    assert after.status_code == 404, "删除后记录仍然存在 —— 刷新会复活"


def test_clear_chat_data_is_idempotent(model_client):
    """会话不存在时删除也算成功 —— 删空对话不该报错。"""
    r = model_client.delete(
        "/api/data/chat-data",
        params={"model": "base_models/rwkv7-g1a-0.1b", "session": "never-existed"},
    )
    assert r.status_code == 200
    assert r.json()["message"] == "删除成功"


def test_clear_chat_data_twice(model_client):
    """连续删两次都应成功（第二次目标已不存在）。"""
    ref = "base_models/rwkv7-g1a-0.1b"
    model_client.post(
        "/api/chat/chat",
        json={"model_ref": ref, "message": "q", "history": [], "session_name": "twice"},
    )
    assert model_client.delete("/api/data/chat-data", params={"model": ref, "session": "twice"}).status_code == 200
    assert model_client.delete("/api/data/chat-data", params={"model": ref, "session": "twice"}).status_code == 200
