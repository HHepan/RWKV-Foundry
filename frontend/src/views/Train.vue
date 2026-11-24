<script setup>
/**
 * 微调训练页。
 *
 * ## 配方（Recipe）是这一页的核心概念
 *
 * 页面顶部始终显示配方指纹（hash）。同一份参数永远得到同一 hash，
 * 因此训练完两次可以直接对比 loss 曲线，不必靠记忆区分「哪次是 lr=2e-5」。
 *
 * 这是 Foundry 隐喻落到 UI 的地方：**模具比熔炉重要**。
 *
 * ## 为什么启动前要先 validate
 *
 * 后端 `Recipe.validate()` 会一次性列出全部问题。旧版没有这层校验，
 * 参数错了要等训练跑到一半才崩，浪费的是 GPU 时间。
 */

import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import * as echarts from 'echarts'

import { trainApi, MODEL_SIZES, defaultRecipe } from '@/api/train'
import { dataApi } from '@/api/data'
import { useTrainPolling } from '@/composables/useTrainPolling'

const recipe = ref(defaultRecipe())
const saveFolder = ref('')
const recipeHash = ref('')
const validation = ref({ valid: true, problems: [], n_layer: null, n_embd: null })
const scriptPreview = ref('')
const previewVisible = ref(false)

const baseModels = ref([])
const datasetFiles = ref([])
const datasetFolder = ref('./out')
const folders = ref([])

const records = ref([])
const activeTab = ref('config')

const poll = useTrainPolling({ intervalMs: 1500 })

const chartEl = ref(null)
let chart = null

const canStart = computed(
  () =>
    validation.value.valid &&
    !!saveFolder.value.trim() &&
    !poll.isRunning() &&
    !!recipe.value.base_model &&
    !!recipe.value.dataset,
)

const statusType = computed(() => {
  switch (poll.status.value) {
    case 'running':
      return 'warning'
    case 'completed':
      return 'success'
    case 'error':
      return 'danger'
    case 'stopped':
      return 'info'
    default:
      return 'info'
  }
})

const statusText = computed(() => {
  const map = {
    idle: '空闲',
    pending: '排队中',
    running: '训练中',
    completed: '已完成',
    stopped: '已停止',
    error: '出错',
  }
  return map[poll.status.value] || poll.status.value
})

const lastLoss = computed(() => {
  const pts = poll.lossPoints.value
  return pts.length ? pts[pts.length - 1].loss.toFixed(4) : '—'
})

const currentStep = computed(() => {
  const pts = poll.lossPoints.value
  return pts.length ? pts[pts.length - 1].step : 0
})

// ── 配方校验（防抖，避免每敲一个字就发请求）────────────────────────────
let hashTimer = null
function scheduleValidate() {
  if (hashTimer) clearTimeout(hashTimer)
  hashTimer = setTimeout(async () => {
    if (!recipe.value.base_model || !recipe.value.dataset) return
    try {
      const res = await trainApi.validateRecipe(recipe.value)
      validation.value = res
      recipeHash.value = res.hash
    } catch (err) {
      validation.value = { valid: false, problems: [err.message], n_layer: null, n_embd: null }
      recipeHash.value = ''
    }
  }, 400)
}

watch(recipe, scheduleValidate, { deep: true })

// ── 初始化 ───────────────────────────────────────────────────────────────
async function loadOptions() {
  try {
    const [models, files, recs] = await Promise.all([
      dataApi.listBaseModels(),
      dataApi.listFiles(datasetFolder.value),
      trainApi.records(),
    ])
    baseModels.value = models
    datasetFiles.value = files
    records.value = recs

    if (models.length && !recipe.value.base_model) recipe.value.base_model = models[0]
    if (files.length && !recipe.value.dataset) {
      recipe.value.dataset = `${datasetFolder.value.replace(/^\.\//, '')}/${files[0]}`
    }
    scheduleValidate()
  } catch (err) {
    ElMessage.error(`加载选项失败: ${err.message}`)
  }
}

async function loadRecords() {
  try {
    records.value = await trainApi.records()
    folders.value = await trainApi.folders()
  } catch (err) {
    ElMessage.error(`加载记录失败: ${err.message}`)
  }
}

onMounted(async () => {
  await loadOptions()
  initChart()
})

// ── 图表 ─────────────────────────────────────────────────────────────────
function initChart() {
  if (!chartEl.value) return
  chart = echarts.init(chartEl.value)
  chart.setOption({
    grid: { left: 52, right: 20, top: 24, bottom: 32 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', name: 'step', axisLine: { lineStyle: { color: '#c9cfda' } } },
    yAxis: { type: 'value', name: 'loss', scale: true, splitLine: { lineStyle: { color: '#eef0f3' } } },
    series: [
      {
        type: 'line',
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 1.8, color: '#e8734a' },
        areaStyle: { opacity: 0.08, color: '#e8734a' },
        data: [],
      },
    ],
  })
  window.addEventListener('resize', resizeChart)
}

function resizeChart() {
  chart?.resize()
}

watch(
  () => poll.lossPoints.value,
  (pts) => {
    if (!chart) return
    chart.setOption({
      xAxis: { data: pts.map((p) => p.step) },
      series: [{ data: pts.map((p) => p.loss) }],
    })
  },
)

// ── 操作 ─────────────────────────────────────────────────────────────────
async function startTraining() {
  const name = saveFolder.value.trim()
  if (!name) {
    ElMessage.warning('请填写保存目录名')
    return
  }
  if (!/^[\w][\w.\-]*$/.test(name)) {
    ElMessage.warning('目录名只能包含字母、数字、下划线、连字符、点，且须以字母数字开头')
    return
  }
  try {
    poll.resetSeries()
    const res = await trainApi.start(recipe.value, name)
    ElMessage.success(`训练已启动 (pid ${res.pid ?? '—'})`)
    activeTab.value = 'monitor'
    await poll.watch(name)
    await loadRecords()
  } catch (err) {
    ElMessage.error(err.message)
  }
}

async function stopTraining() {
  try {
    await ElMessageBox.confirm(
      '停止会先发 SIGTERM 并等待 10 秒，让 DeepSpeed 保存当前进度；超时才强制终止。',
      '停止训练',
      { type: 'warning', confirmButtonText: '停止' },
    )
    await poll.stop()
    ElMessage.info('已发送停止信号')
    await loadRecords()
  } catch (err) {
    if (err !== 'cancel') ElMessage.error(err.message)
  }
}

async function previewScript() {
  const name = saveFolder.value.trim() || 'preview'
  try {
    const res = await trainApi.previewScript(recipe.value, name)
    scriptPreview.value = res.script
    previewVisible.value = true
  } catch (err) {
    ElMessage.error(err.message)
  }
}

async function viewRecord(row) {
  try {
    const detail = await trainApi.recordDetail(row.folder)
    poll.resetSeries()
    await poll.watch(row.folder)
    activeTab.value = 'monitor'
    ElMessage.success(`已载入 ${row.folder} 的曲线（${detail.loss_history?.length || 0} 个点）`)
  } catch (err) {
    ElMessage.error(err.message)
  }
}

async function deleteRecord(row) {
  try {
    await ElMessageBox.confirm(
      `删除训练记录 ${row.folder}？\n\n注意：只删除数据库记录，磁盘上的权重文件会保留。`,
      '删除记录',
      { type: 'warning', confirmButtonText: '删除' },
    )
    await trainApi.deleteRecord(row.folder)
    ElMessage.success('记录已删除')
    await loadRecords()
  } catch (err) {
    if (err !== 'cancel') ElMessage.error(err.message)
  }
}

function fmtTime(s) {
  return s ? s.replace('T', ' ').replace('+00:00', ' UTC') : '—'
}

function statusTagType(s) {
  const map = { completed: 'success', running: 'warning', error: 'danger', stopped: 'info', pending: 'info' }
  return map[s] || 'info'
}
</script>

<template>
  <div class="train-layout">
    <el-tabs v-model="activeTab" class="full-tabs">
      <!-- ═══ 配置 ═══ -->
      <el-tab-pane label="配方配置" name="config">
        <div class="config-grid">
          <div class="panel">
            <div class="panel-title">
              <span>训练配方</span>
              <el-tooltip content="配方指纹：同参数永远同 hash，可用于横向对比实验">
                <el-tag :type="validation.valid ? 'success' : 'danger'" size="small" effect="dark">
                  {{ recipeHash || '待校验' }}
                </el-tag>
              </el-tooltip>
            </div>

            <div class="panel-body">
              <el-form label-width="112px" size="small">
                <el-form-item label="基底模型">
                  <el-select v-model="recipe.base_model" class="full" placeholder="选择基底权重">
                    <el-option v-for="m in baseModels" :key="m" :label="m" :value="m" />
                  </el-select>
                </el-form-item>

                <el-form-item label="数据集目录">
                  <el-select v-model="datasetFolder" class="full" @change="loadOptions">
                    <el-option label="./out" value="./out" />
                    <el-option label="./" value="./" />
                  </el-select>
                </el-form-item>

                <el-form-item label="数据集文件">
                  <el-select v-model="recipe.dataset" class="full" placeholder="选择 .jsonl">
                    <el-option
                      v-for="f in datasetFiles"
                      :key="f"
                      :label="f"
                      :value="`${datasetFolder.replace(/^\.\//, '')}/${f}`"
                    />
                  </el-select>
                </el-form-item>

                <el-form-item label="模型尺寸">
                  <el-select v-model="recipe.model_size" class="full">
                    <el-option v-for="s in MODEL_SIZES" :key="s" :label="s" :value="s" />
                  </el-select>
                  <div v-if="validation.n_layer" class="hint">
                    n_layer={{ validation.n_layer }}, n_embd={{ validation.n_embd }}
                  </div>
                </el-form-item>

                <el-divider content-position="left">序列与批量</el-divider>

                <el-form-item label="ctx_len">
                  <el-input-number v-model="recipe.ctx_len" :min="64" :step="64" class="full" />
                </el-form-item>
                <el-form-item label="micro_bsz">
                  <el-input-number v-model="recipe.micro_bsz" :min="1" class="full" />
                </el-form-item>
                <el-form-item label="epoch_steps">
                  <el-input-number v-model="recipe.epoch_steps" :min="1" :step="100" class="full" />
                </el-form-item>
                <el-form-item label="epoch_count">
                  <el-input-number v-model="recipe.epoch_count" :min="1" class="full" />
                </el-form-item>
                <el-form-item label="epoch_save">
                  <el-input-number v-model="recipe.epoch_save" :min="1" class="full" />
                </el-form-item>

                <el-divider content-position="left">学习率</el-divider>

                <el-form-item label="lr_init">
                  <el-input-number v-model="recipe.lr_init" :min="0" :step="1e-5" :precision="7" class="full" />
                </el-form-item>
                <el-form-item label="lr_final">
                  <el-input-number v-model="recipe.lr_final" :min="0" :step="1e-5" :precision="7" class="full" />
                </el-form-item>

                <el-divider content-position="left">LoRA</el-divider>

                <el-form-item label="lora_r">
                  <el-input-number v-model="recipe.lora_r" :min="1" :step="8" class="full" />
                </el-form-item>
                <el-form-item label="lora_alpha">
                  <el-input-number v-model="recipe.lora_alpha" :min="1" :step="8" class="full" />
                </el-form-item>
                <el-form-item label="lora_dropout">
                  <el-input-number v-model="recipe.lora_dropout" :min="0" :max="0.99" :step="0.01" :precision="2" class="full" />
                </el-form-item>

                <el-divider content-position="left">引擎</el-divider>

                <el-form-item label="precision">
                  <el-radio-group v-model="recipe.precision">
                    <el-radio-button value="bf16">bf16</el-radio-button>
                    <el-radio-button value="fp32">fp32</el-radio-button>
                  </el-radio-group>
                  <div class="hint">上游 CUDA 内核拒绝 fp16</div>
                </el-form-item>

                <el-form-item label="my_testing">
                  <el-select v-model="recipe.my_testing" class="full">
                    <el-option label="x070 (RWKV-7)" value="x070" />
                    <el-option label="x060 (RWKV-6)" value="x060" />
                    <el-option label="x050 (RWKV-5)" value="x050" />
                  </el-select>
                </el-form-item>
              </el-form>
            </div>
          </div>

          <div class="right-col">
            <div class="panel">
              <div class="panel-title"><span>保存与校验</span></div>
              <div class="panel-body">
                <el-form label-width="92px" size="small">
                  <el-form-item label="保存目录名">
                    <el-input v-model="saveFolder" placeholder="例如 lora-run-01" />
                  </el-form-item>
                </el-form>

                <div v-if="validation.problems.length" class="problem-box">
                  <p class="problem-title">配方存在问题（{{ validation.problems.length }} 项）</p>
                  <p v-for="(p, i) in validation.problems" :key="i" class="problem-item">• {{ p }}</p>
                </div>

                <div class="actions">
                  <el-button size="small" @click="previewScript">预览脚本</el-button>
                  <el-button
                    v-if="!poll.isRunning()"
                    type="primary"
                    size="small"
                    :disabled="!canStart"
                    @click="startTraining"
                  >
                    开始训练
                  </el-button>
                  <el-button v-else type="danger" size="small" @click="stopTraining">停止训练</el-button>
                </div>
              </div>
            </div>

            <div class="panel">
              <div class="panel-title">
                <span>训练状态</span>
                <el-tag :type="statusType" size="small">{{ statusText }}</el-tag>
              </div>
              <div class="panel-body">
                <el-descriptions :column="2" size="small" border>
                  <el-descriptions-item label="目录">{{ poll.folder.value || '—' }}</el-descriptions-item>
                  <el-descriptions-item label="PID">{{ poll.pid.value ?? '—' }}</el-descriptions-item>
                  <el-descriptions-item label="当前步">{{ currentStep }}</el-descriptions-item>
                  <el-descriptions-item label="最新 loss">{{ lastLoss }}</el-descriptions-item>
                  <el-descriptions-item label="耗时">{{ poll.elapsed.value ? `${poll.elapsed.value}s` : '—' }}</el-descriptions-item>
                  <el-descriptions-item label="退出码">{{ poll.exitCode.value ?? '—' }}</el-descriptions-item>
                </el-descriptions>
                <p v-if="poll.error.value" class="problem-box problem-title">{{ poll.error.value }}</p>
              </div>
            </div>
          </div>
        </div>
      </el-tab-pane>

      <!-- ═══ 监控 ═══ -->
      <el-tab-pane label="训练监控" name="monitor">
        <div class="monitor-grid">
          <div class="panel chart-panel">
            <div class="panel-title">
              <span>Loss 曲线</span>
              <span class="mono dim">{{ poll.lossPoints.value.length }} 点</span>
            </div>
            <div ref="chartEl" class="chart" />
          </div>

          <div class="panel log-panel">
            <div class="panel-title">
              <span>训练日志</span>
              <span class="mono dim">{{ poll.logs.value.length }} 行</span>
            </div>
            <div class="log-body mono">
              <p v-if="!poll.logs.value.length" class="empty-hint">暂无日志</p>
              <div v-for="(line, i) in poll.logs.value" :key="i" class="log-line">{{ line }}</div>
            </div>
          </div>
        </div>
      </el-tab-pane>

      <!-- ═══ 记录 ═══ -->
      <el-tab-pane label="训练记录" name="records">
        <div class="panel">
          <div class="panel-title">
            <span>历史训练</span>
            <el-button size="small" text @click="loadRecords">刷新</el-button>
          </div>
          <div class="panel-body">
            <p v-if="!records.length" class="empty-hint">还没有训练记录</p>
            <el-table v-else :data="records" size="small">
              <el-table-column prop="folder" label="目录" min-width="140" />
              <el-table-column prop="base_model" label="基底模型" min-width="180" />
              <el-table-column prop="status" label="状态" width="90">
                <template #default="{ row }">
                  <el-tag :type="statusTagType(row.status)" size="small">{{ row.status }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column label="配方" width="120">
                <template #default="{ row }">
                  <span class="mono dim">{{ row.recipe_hash || '—' }}</span>
                </template>
              </el-table-column>
              <el-table-column label="开始时间" min-width="160">
                <template #default="{ row }">{{ fmtTime(row.started_at) }}</template>
              </el-table-column>
              <el-table-column label="操作" width="140" fixed="right">
                <template #default="{ row }">
                  <el-button size="small" text @click="viewRecord(row)">曲线</el-button>
                  <el-button size="small" text type="danger" @click="deleteRecord(row)">删除</el-button>
                </template>
              </el-table-column>
            </el-table>
          </div>
        </div>
      </el-tab-pane>
    </el-tabs>

    <el-dialog v-model="previewVisible" title="训练脚本预览" width="820px">
      <pre class="mono script-preview">{{ scriptPreview }}</pre>
    </el-dialog>
  </div>
</template>

<style scoped>
.train-layout {
  height: 100%;
}

.full-tabs {
  height: 100%;
}

:deep(.el-tabs__content) {
  height: calc(100% - 40px);
  overflow: auto;
}

.config-grid {
  display: grid;
  grid-template-columns: minmax(420px, 1fr) minmax(320px, 420px);
  gap: 12px;
  align-items: start;
}

.right-col {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.full {
  width: 100%;
}

.hint {
  font-size: 11px;
  color: var(--foundry-text-dim);
  margin-top: 2px;
  line-height: 1.4;
}

.problem-box {
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 8px;
  padding: 10px 12px;
  margin: 10px 0;
}

.problem-title {
  margin: 0 0 6px;
  font-size: 12px;
  font-weight: 600;
  color: #b91c1c;
}

.problem-item {
  margin: 0 0 4px;
  font-size: 12px;
  color: #7f1d1d;
  line-height: 1.5;
}

.actions {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}

.monitor-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
}

.chart-panel .chart {
  height: 300px;
  padding: 8px;
}

.log-panel {
  display: flex;
  flex-direction: column;
  min-height: 240px;
}

.log-body {
  flex: 1;
  overflow: auto;
  padding: 10px 14px;
  background: #1b1f2a;
  border-radius: 0 0 10px 10px;
  max-height: 360px;
}

.log-line {
  color: #c8d0e0;
  white-space: pre-wrap;
  word-break: break-all;
}

.dim {
  color: var(--foundry-text-dim);
}

.script-preview {
  margin: 0;
  padding: 14px;
  background: #1b1f2a;
  color: #c8d0e0;
  border-radius: 8px;
  max-height: 60vh;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
