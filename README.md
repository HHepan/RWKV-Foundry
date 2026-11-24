# RWKV-Foundry

RWKV 模型的微调与对话测试产线。

给定一个基底模型和一份数据集，产出一个可复现的 LoRA 铸件，并立刻检验它。

## 名字的含义

Foundry 是铸造厂。这个名字定义了项目的数据流，而不是界面结构：

| 铸造厂 | 本项目 |
|---|---|
| 原材料 | `workspace/base_models/*.pth` |
| 熔炉 | RWKV-PEFT 训练进程 |
| **模具 / 配方** | `Recipe`：LoRA 超参 + 数据 + 训练参数 |
| 铸件 | `workspace/checkpoints/<run>/rwkv-N.pth` |
| 质检 | 对话测试 |
| 出厂记录 | `foundry.db` |

核心资产是**配方**，不是熔炉。每一炉独立可复现——同样的参数产出同一个
`recipe.hash`，因此可以直接对比两次训练的 loss 曲线。

## 与 RWKV-PEFT / RWKV-Runner 的关系

| 项目 | 定位 | 本项目的定位 |
|---|---|---|
| `RWKV-PEFT` | 微调引擎（LoRA/PiSSA/MiSS/State/MTP） | **调用它**，不重写 |
| `RWKV-Runner` | 桌面 GUI + 推理 API | 独立实现，采样参数对齐 |
| **RWKV-Foundry** | **把两端接成一条可复现的产线** | 数据 → 配方 → 铸件 → 质检 |

差异化能力：**token 级增量 prefill**。RWKV 是循环模型，state 累积完整历史，
因此后续轮次只需喂差量（O(n) 而非 O(n²)）。这是本项目相对 RWKV-Runner 的核心
技术点，实现位于 `foundry/infer/session.py`。

## 目录结构

```
RWKV-Foundry/
├── backend/
│   ├── foundry/
│   │   ├── errors.py        # 异常类型
│   │   ├── paths.py         # ★ 唯一路径真相来源
│   │   ├── env.py           # RWKV 环境变量引导
│   │   ├── config.py        # 运行配置
│   │   ├── domain/          # 纯逻辑，零 web 依赖
│   │   ├── train/           # 训练编排
│   │   ├── infer/           # 推理与增量 prefill
│   │   ├── api/             # FastAPI 路由（薄层）
│   │   └── third_party/     # vendored 上游代码 + PROVENANCE
│   ├── tests/
│   └── pyproject.toml
├── frontend/                # 仅桌面端
├── workspace/               # 运行期数据，不入库
└── docs/
```

## 快速开始

```bash
# ── 后端 ────────────────────────────────────────────────────────────────
conda create -n foundry python=3.11 && conda activate foundry
cd backend

# 只跑 Web 与单元测试（无需 GPU / torch）
pip install -e ".[dev]"

# 要跑训练与推理，额外装 train 组
pip install -e ".[dev,train]"
```

### 关于 torch 与 CUDA（重要）

`.[train]` 里的 `torch>=2.1` 装到的是 **PyPI 默认 wheel**，在部分机器上是 CPU 版
或与你的驱动不匹配。**必须按本机 CUDA 版本显式指定索引**：

```bash
# 先确认驱动支持的 CUDA 版本
nvidia-smi          # 右上角 "CUDA Version: 12.x"

# 再装对应 wheel（示例：CUDA 12.1）
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

装反了不会报错，只会 `torch.cuda.is_available()` 返回 False，
然后发现"训练极慢"——因为它悄悄跑在 CPU 上。

### 准备 workspace

```bash
# 数据目录可以放在仓库外，用 FOUNDRY_HOME 指向（推荐，避免仓库变胖）
mkdir -p /data/foundry/{base_models,data,checkpoints,tools}

# 克隆上游微调框架
git clone https://github.com/JL-er/RWKV-PEFT.git /data/foundry/tools/RWKV-PEFT
cd /data/foundry/tools/RWKV-PEFT && pip install -r requirements.txt && cd -

# 放入基底权重
cp rwkv7-g1a-0.1b-*.pth /data/foundry/base_models/
```

### 前端

```bash
cd frontend
npm install
npm run dev        # http://127.0.0.1:5173
```

**注意**：若环境设了 `NODE_ENV=production`，`npm install` 会**跳过
devDependencies**，导致 vite 缺失、`npm run dev` 报 `vite: not found`。
遇到时用：

```bash
NODE_ENV=development npm install --include=dev
```

### 启动

```bash
# 后端（在 backend/ 下）
export FOUNDRY_HOME=/data/foundry
uvicorn foundry.api.app:app_factory --factory --host 127.0.0.1 --port 8000

# 前端会通过 vite proxy 自动转发 /api 到 8000，无需额外配置 CORS
```

### 验证

```bash
cd backend && pytest              # 238 项，无需 GPU
curl http://127.0.0.1:8000/api/health
```

预期输出：

```json
{"ok":true,"model_loaded":false,"model_ref":null,"active_train":null,"schema_version":1}
```

`model_loaded: false` 是正常的——模型在你第一次发起对话时才加载。

## 配置

所有路径通过 `FOUNDRY_HOME` 覆盖，不存在任何硬编码绝对路径：

```bash
export FOUNDRY_HOME=/data/foundry
export FOUNDRY_PEFT_DIR=/opt/RWKV-PEFT   # 可选，多项目共享时用
```

启动时校验环境完整性，缺失项会**一次性全部列出**并给出修复命令，而不是
静默创建空目录后让训练跑到一半才崩溃。

## 关键约束

1. **必须在 import torch 之前调用 `foundry.env.bootstrap()`**。RWKV 的三个内核
   开关只在首次 import 时读取一次，晚了会静默降级。
2. **训练数据与权重保持文件形态，不入库**。RWKV-PEFT 直接读文件，
   入库即产生两套真相。
3. **训练子进程不碰 SQLite**。它只 append 文本文件，由 Web 进程作为
   ingester 灌入数据库——单写入者原则。

## 许可

待定。
