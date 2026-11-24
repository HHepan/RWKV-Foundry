# Vendored 代码来源与修改记录

本目录下的文件**不是本项目原创**，而是从上游 RWKV 代码复制并本地修改。
每次修改都必须在此登记，否则半年后无人知道某段代码为何与上游不同。

---

## `rwkv_x070_dual.py`

| 项目 | 内容 |
|---|---|
| 上游 | [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) — `rwkv.model.RWKV_x070` |
| 上游版本 | rwkv pip **0.8.30** |
| 本地移植自 | `llm-finetune-webui/backend/app/services/rwkv_x070_dual.py` |
| 行数 | 385（未修改） |

### 用途

标准 `RWKV_x070` 只能把整个模型放到单张卡上。本文件实现**层间切分**，
支持形如 `cuda:0 fp16 -> cuda:1 fp16` 的 strategy 语法，让单卡放不下的
大模型（如 13.3B）可以跨两张卡运行。

### 修改内容

**无。** 原样移植。

### 依赖

- `torch`
- `torch.nn.functional`

---

## `rwkv_utils_patch.py`

| 项目 | 内容 |
|---|---|
| 上游 | [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) — `rwkv.utils.PIPELINE` |
| 上游版本 | rwkv pip **0.8.30** |
| 本地移植自 | `llm-finetune-webui/backend/app/services/rwkv_utils_patch.py` |
| 原始行数 | 228 |

### 用途

官方 `rwkv.utils.PIPELINE.generate()` 只返回文本，不返回 `state`。
没有 state 就无法做**增量 prefill**（本项目核心能力），因此需要补丁版本。

### 上游原版 vs 本文件

| # | 修改点 | 原因 |
|---|---|---|
| 1 | `generate()` 返回 `(out_str, state, occurrence)` | 官方只返回 `out_str`，无法保存循环态 |
| 2 | `generate()` 接受 `occurrence` 参数 | 支持跨轮次重复惩罚（frequency/presence penalty） |
| 3 | callback 抛 `StopIteration` 时仍返回当前 state | 支持中途停止且不丢失已有推理成果 |
| 4 | **`PIPELINE.__init__` 新增 `vocab_path` 参数** | 见下方说明 |

### ★ 修改点 4 的详细说明（foundry 移植时新增）

**原版做法**（`llm-finetune-webui`）：

```python
import rwkv
_RWKV_DIR = os.path.dirname(rwkv.__file__)
from rwkv_tokenizer import TRIE_TOKENIZER
_VOCAB_PATH = os.path.join(_RWKV_DIR, 'rwkv_vocab_v20230424.txt')

class PIPELINE:
    def __init__(self, model, WORD_NAME):
        self.tokenizer = TRIE_TOKENIZER(_VOCAB_PATH)   # 从 rwkv 包内部取
```

**问题**：

1. `rwkv` pip 包**不保证**附带 `rwkv_vocab_v20230424.txt`，不同版本可能没有，
   或者路径不同 → 运行时 `FileNotFoundError`
2. 更严重的是：**即使找到了，也不保证与训练用的词表一致**。RWKV-PEFT 在
   `json2binidx_tool/` 下自带一份词表用于 binidx 转换。若推理用的是 rwkv 包
   里另一份版本稍有不同的词表，**token id 会静默错位**：
   - 不报错
   - 模型输出人类读不懂的乱码
   - 极难定位到"词表版本不一致"这个根因

**现在做法**：

```python
class PIPELINE:
    def __init__(self, model, WORD_NAME, vocab_path=None):
        if WORD_NAME == 'rwkv_vocab_v20230424':
            if not vocab_path:
                raise ValueError("必须显式传入 vocab_path ...")
            self.tokenizer = TRIE_TOKENIZER(vocab_path)
```

调用方（`foundry/api/deps.py` 的 `CudaModelLoader.load`）传入
`<RWKV-PEFT>/json2binidx_tool/rwkv_vocab_v20230424.txt`，
**保证训练与推理用同一份词表**。

缺参数时立即报错而非静默降级——这是本项目的一贯原则。

### 依赖

- `torch`
- `numpy`
- `rwkv_tokenizer`（由 `apply_rwkv_paths()` 加入 `sys.path`）
- `rwkv`（可选：仅用于定位 tokenizer 模块，缺失时不影响词表加载）

---

## 升级流程

上游 RWKV-LM 更新时，按以下步骤操作：

```bash
# 1. 下载新版
pip download rwkv==<新版本> --no-deps -d /tmp/rwkv-pkg
cd /tmp/rwkv-pkg && unzip -o rwkv-*.whl -d extracted

# 2. 与新版本对比
diff extracted/rwkv/model.py         backend/foundry/third_party/rwkv_x070_dual.py
diff extracted/rwkv/utils.py         backend/foundry/third_party/rwkv_utils_patch.py

# 3. 逐条确认本文件表格中的 4 处修改点是否仍然必要、是否仍然适用

# 4. 更新本文件的"上游版本"字段与修改记录
```

**不要跳过第 3 步。** 上游可能已经自行修好了某个问题——
旧项目 README 里那个「改 trainer.py 167 行」的补丁就这样过时了：
上游重构后行号已变，而补丁说明还留着，成了误导性文档。
