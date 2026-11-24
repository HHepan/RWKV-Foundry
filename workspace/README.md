# workspace —— 运行期数据目录

**本目录不进入版本库。** 仓库里只有骨架（各子目录的 `.gitkeep`），
实际的模型权重、数据集、训练产物都在本地。

## 目录职责

| 目录 | 内容 | 写入方 | 入版本库 |
|---|---|---|---|
| `base_models/` | 基底权重 `.pth`、角色卡 | 用户手动放置 | 否（数十 GB） |
| `data/` | 训练数据集 `.jsonl` | 用户上传/管理 | 否（体积大） |
| `checkpoints/<run>/` | 训练产物：LoRA 权重 + 元数据 | 训练进程 | 否 |
| `tools/RWKV-PEFT/` | 上游微调框架（clone） | git clone | 否 |
| `foundry.db` | 元数据与索引（SQLite） | Web 服务 | 否 |

## 初始化

```bash
# 1. 克隆上游微调框架（必须，否则无法训练）
git clone https://github.com/JL-er/RWKV-PEFT.git workspace/tools/RWKV-PEFT

# 2. 安装其依赖
cd workspace/tools/RWKV-PEFT && pip install -r requirements.txt && cd -

# 3. 放入基底权重
cp rwkv7-g1a-0.1b-*.pth workspace/base_models/
```

## 换位置

整个 workspace 可以搬到任意路径，通过环境变量指向：

```bash
export FOUNDRY_HOME=/data/foundry
```

如果 RWKV-PEFT 想多项目共享，单独指定：

```bash
export FOUNDRY_PEFT_DIR=/opt/RWKV-PEFT
```

数据库里记录的是**相对 workspace 的路径**，因此 DB 与 data/ 必须一起搬。

## 为什么数据本体不进数据库

训练数据 `.jsonl` 和权重 `.pth` 保持文件形态，理由：

1. **RWKV-PEFT 直接读文件**。入库后必须先导出再训练 = 两套真相 = 一次不同步就训练跑错数据。
2. 权重是 2~40GB 二进制，存 DB 是灾难。

数据库中只存**元数据与索引**（训练运行记录、loss 曲线、会话、数据集行数缓存）。
