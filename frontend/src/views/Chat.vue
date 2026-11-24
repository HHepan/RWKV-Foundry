<script setup>
/**
 * 对话测试页。
 *
 * ## 这一页的复杂度都收敛到 useChatStream 里了
 *
 * 旧版 Chat.vue 有 1718 行，把 SSE 解析、thinking 切分、参数面板、图表
 * 全塞在一个组件里。这里页面只负责渲染与用户交互，流式状态由 composable 管。
 *
 * ## 会话隔离的 UI 体现
 *
 * 每个「模型 + 会话名」是一条独立会话。切换会话会：
 *   1. 从后端读该会话的历史消息
 *   2. 清空当前展示的流式状态
 * 后端 state 也按同一 key 隔离，两侧一致。
 */

import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import * as echarts from 'echarts'

import { chatApi, defaultInferParams, roleplayParams } from '@/api/chat'
import { dataApi } from '@/api/data'
import { useChatStream } from '@/composables/useChatStream'

// ── 模型与会话 ───────────────────────────────────────────────────────────
const modelOptions = ref([])
const selectedModel = ref('')
const sessionName = ref('default')
const models = ref([]) // 可选模型（基底 + checkpoints）

const roleList = ref([])
const selectedRole = ref('none')

const params = ref(defaultInferParams())
const roleId = computed(() => (selectedRole.value === 'none' ? null : selectedRole.value))

const { messages, isStreaming, isThinking, error, stats, canSend, send, regenerate, abort, clear, loadMessages } =
  useChatStream({
    getModelRef: () => selectedModel.value,
    getParams: () => params.value,
    getRoleId: () => roleId.value,
    getSessionName: () => sessionName.value,
  })

const userInput = ref('')
const messagesEl = ref(null)
const modelLoading = ref(false)

const thinkingMode = computed({
  get: () => !!params.value.thinking_mode,
  set: (v) => {
    params.value = { ...params.value, thinking_mode: v }
  },
})

const showParams = ref(false)
const showPpl = ref(true)
const pplEl = ref(null)
let pplChart = null
const pplPoints = ref([])

// ── 初始化 ───────────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const [baseModels, folders, chatModels, roles] = await Promise.all([
      dataApi.listBaseModels(),
      dataApi.listCheckpointFolders(),
      dataApi.listChatModels(),
      dataApi.listRoles(),
    ])

    roleList.value = [{ id: 'none', name: '无（普通对话）', content: '' }, ...roles]

    const opts = []
    for (const m of baseModels) {
      opts.push({ ref: `base_models/${m}`, label: `[基底] ${m}` })
    }
    for (const f of folders) {
      const files = await dataApi.listCheckpointFiles(f)
      for (const file of files) {
        opts.push({ ref: `${f}/${file}`, label: `[${f}] ${file}` })
      }
    }
    models.value = opts
    modelOptions.value = opts

    // 已有对话记录优先
    if (chatModels.length) {
      const latest = chatModels[0]
      selectedModel.value = latest.model
      sessionName.value = latest.session || 'default'
      if (latest.role_id && latest.role_id !== 'none') selectedRole.value = latest.role_id
      if (latest.params) params.value = { ...params.value, ...latest.params }
      await loadHistory()
    } else if (opts.length) {
      selectedModel.value = opts[0].ref
    }
  } catch (err) {
    ElMessage.error(`加载模型列表失败: ${err.message}`)
  }
}

async function loadHistory() {
  if (!selectedModel.value) return
  try {
    const res = await dataApi.getChatData(selectedModel.value, sessionName.value)
    loadMessages(res.dialogue_content || [])
    if (res.params) params.value = { ...params.value, ...res.params }
  } catch (err) {
    // 新会话没有历史是正常的
    if (err.status !== 404) ElMessage.warning(`读取历史失败: ${err.message}`)
    loadMessages([])
  }
}

async function preloadModel() {
  if (!selectedModel.value) return
  modelLoading.value = true
  try {
    await chatApi.preloadModel(selectedModel.value)
    ElMessage.success('模型已加载')
  } catch (err) {
    ElMessage.error(err.message)
  } finally {
    modelLoading.value = false
  }
}

onMounted(async () => {
  await loadModels()
  initPplChart()
})

// ── 交互 ─────────────────────────────────────────────────────────────────
function scrollToBottom() {
  requestAnimationFrame(() => {
    if (messagesEl.value) messagesEl.value.scrollTop = messagesEl.value.scrollHeight
  })
}

watch(messages, scrollToBottom, { deep: true })

async function onSend() {
  const text = userInput.value.trim()
  if (!text) return
  userInput.value = ''
  const ok = await send(text)
  if (!ok && error.value) ElMessage.error(error.value)
}

function onKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault()
    onSend()
  }
}

async function onRegenerate() {
  const ok = await regenerate()
  if (!ok && error.value) ElMessage.error(error.value)
}

async function onClear() {
  try {
    await ElMessageBox.confirm(
      '清空会调用 /reset-state，让模型彻底忘记本次对话的 state。确定继续？',
      '清空对话',
      { type: 'warning', confirmButtonText: '清空' },
    )
    await clear()
    pplPoints.value = []
    updatePplChart()
    ElMessage.success('已清空，模型状态已重置')
  } catch (err) {
    if (err !== 'cancel') ElMessage.error(err.message)
  }
}

async function onSwitchModel() {
  await loadHistory()
  ElMessage.info(`已切换到 ${selectedModel.value} / ${sessionName.value}`)
}

function onRoleChange(val) {
  if (val === 'none') {
    params.value = { ...defaultInferParams(), thinking_mode: params.value.thinking_mode }
    ElMessage.info('已恢复默认推理参数')
  } else {
    const role = roleList.value.find((r) => r.id === val)
    if (role) {
      params.value = { ...params.value, ...roleplayParams() }
      ElMessage.success(`已切换角色：${role.name}，参数已调整为角色扮演模式`)
    }
  }
}

async function saveParams() {
  if (!selectedModel.value) return
  try {
    await dataApi.saveChatParams?.({
      folder: selectedModel.value.split('/')[0],
      model: selectedModel.value.split('/')[1],
      session: sessionName.value,
      params: params.value,
      roleId: roleId.value,
    })
  } catch {
    /* 端点可选，失败不阻断 */
  }
  ElMessage.success('参数已应用到当前会话')
}

// ── 困惑度图表 ───────────────────────────────────────────────────────────
function initPplChart() {
  if (!pplEl.value) return
  pplChart = echarts.init(pplEl.value)
  pplChart.setOption({
    grid: { left: 44, right: 12, top: 16, bottom: 24 },
    xAxis: { type: 'category', show: false },
    yAxis: { type: 'value', scale: true, splitLine: { lineStyle: { color: '#eef0f3' } } },
    series: [{ type: 'line', smooth: true, symbol: 'none', lineStyle: { width: 1.5, color: '#4f7cff' }, data: [] }],
  })
  window.addEventListener('resize', () => pplChart?.resize())
}

function updatePplChart() {
  pplChart?.setOption({
    xAxis: { data: pplPoints.value.map((_, i) => i) },
    series: [{ data: pplPoints.value }],
  })
}

watch(
  () => stats.value,
  (s) => {
    if (!s) return
    // 用「复用比例」直观体现增量 prefill 的效果
    const total = s.tokensFed + s.tokensReused
    if (total > 0) {
      pplPoints.value = [...pplPoints.value.slice(-59), s.tokensReused]
      updatePplChart()
    }
  },
)

const reuseRatio = computed(() => {
  const s = stats.value
  if (!s) return null
  const total = s.tokensFed + s.tokensReused
  if (!total) return null
  return Math.round((s.tokensReused / total) * 100)
})
</script>

<template>
  <div class="chat-layout">
    <!-- 左：会话设置 -->
    <aside class="panel side">
      <div class="panel-title"><span>会话</span></div>
      <div class="panel-body side-body">
        <el-select v-model="selectedModel" size="small" class="full" filterable @change="onSwitchModel">
          <el-option v-for="m in modelOptions" :key="m.ref" :label="m.label" :value="m.ref" />
        </el-select>

        <el-input v-model="sessionName" size="small" placeholder="会话名" @change="onSwitchModel" />

        <el-button size="small" class="full" :loading="modelLoading" @click="preloadModel">
          预加载模型
        </el-button>

        <el-divider />

        <el-form label-position="top" size="small">
          <el-form-item label="角色卡">
            <el-select v-model="selectedRole" class="full" @change="onRoleChange">
              <el-option v-for="r in roleList" :key="r.id" :label="r.name" :value="r.id" />
            </el-select>
          </el-form-item>
        </el-form>

        <el-switch v-model="thinkingMode" active-text="思考模式" class="full" />

        <el-button size="small" text class="full" @click="showParams = !showParams">
          {{ showParams ? '收起参数' : '展开推理参数' }}
        </el-button>

        <div v-if="showParams" class="param-box">
          <label class="param">
            <span>temperature</span>
            <el-input-number v-model="params.temperature" :min="0" :max="3" :step="0.05" :precision="2" size="small" controls-position="right" />
          </label>
          <label class="param">
            <span>top_p</span>
            <el-input-number v-model="params.top_p" :min="0.01" :max="1" :step="0.05" :precision="2" size="small" controls-position="right" />
          </label>
          <label class="param">
            <span>top_k</span>
            <el-input-number v-model="params.top_k" :min="0" :step="10" size="small" controls-position="right" />
          </label>
          <label class="param">
            <span>max_tokens</span>
            <el-input-number v-model="params.max_tokens" :min="1" :step="128" size="small" controls-position="right" />
          </label>
          <label class="param">
            <span>presence</span>
            <el-input-number v-model="params.alpha_presence" :min="0" :max="2" :step="0.05" :precision="2" size="small" controls-position="right" />
          </label>
          <label class="param">
            <span>frequency</span>
            <el-input-number v-model="params.alpha_frequency" :min="0" :max="2" :step="0.05" :precision="2" size="small" controls-position="right" />
          </label>
          <label class="param">
            <span>decay</span>
            <el-input-number v-model="params.alpha_decay" :min="0" :max="1" :step="0.001" :precision="3" size="small" controls-position="right" />
          </label>
          <label class="param">
            <span>newline_adj</span>
            <el-input-number v-model="params.newline_adj" :min="-10" :max="10" size="small" controls-position="right" />
          </label>
          <label class="param">
            <span>max_rounds</span>
            <el-input-number v-model="params.max_rounds" :min="0" size="small" controls-position="right" />
            <small class="hint">0 = 不裁剪（保留增量 prefill 前缀）</small>
          </label>
          <el-checkbox v-model="params.presystem">注入示范前缀</el-checkbox>
          <el-checkbox v-model="params.global_penalty">全局重复惩罚</el-checkbox>
          <el-button size="small" class="full" type="primary" plain @click="saveParams">
            应用到本会话
          </el-button>
        </div>

        <el-divider />

        <div v-if="stats" class="stats-mini">
          <div class="mini-row">
            <span>复用</span>
            <strong>{{ reuseRatio ?? 0 }}%</strong>
          </div>
          <div class="mini-row">
            <span>喂入 / 复用</span>
            <strong class="mono">{{ stats.tokensFed }} / {{ stats.tokensReused }}</strong>
          </div>
          <p class="hint">复用率高说明增量 prefill 生效</p>
        </div>
      </div>
    </aside>

    <!-- 中：对话 -->
    <section class="panel chat-panel">
      <div class="panel-title">
        <span>{{ selectedModel || '未选择模型' }} <em class="dim">/ {{ sessionName }}</em></span>
        <div class="head-actions">
          <el-tooltip content="整段历史 + 本轮消息重新生成，后端会回滚 state">
            <el-button size="small" text :disabled="isStreaming || messages.length < 2" @click="onRegenerate">
              重新生成
            </el-button>
          </el-tooltip>
          <el-button size="small" text type="danger" :disabled="!messages.length" @click="onClear">
            清空
          </el-button>
        </div>
      </div>

      <div ref="messagesEl" class="messages">
        <p v-if="!messages.length" class="empty-hint">开始一段新对话</p>

        <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
          <div class="msg-role">{{ m.role === 'user' ? '你' : '模型' }}</div>

          <div class="msg-body">
            <!-- thinking 折叠区 -->
            <div v-if="m.role === 'assistant' && (m.thinking || (isThinking && i === messages.length - 1))" class="think-block">
              <button class="think-toggle" @click="m.showThinking = !m.showThinking">
                <el-icon><component :is="m.showThinking ? 'ArrowDown' : 'ArrowRight'" /></el-icon>
                <span>{{ isThinking && i === messages.length - 1 ? '思考中…' : '思考过程' }}</span>
              </button>
              <pre v-if="m.showThinking" class="think-text">{{ m.thinking || m.visibleThinking }}</pre>
            </div>

            <div class="msg-text" :class="{ streaming: m.streaming }">{{ m.content }}</div>
            <span v-if="m.streaming && !m.content" class="cursor">▍</span>
          </div>
        </div>
      </div>

      <div v-if="error" class="error-bar">{{ error }}</div>

      <div class="composer">
        <el-input
          v-model="userInput"
          type="textarea"
          :rows="3"
          resize="none"
          placeholder="输入消息，Enter 发送，Shift+Enter 换行"
          :disabled="!selectedModel"
          @keydown="onKeydown"
        />
        <div class="composer-actions">
          <span class="hint">
            {{ thinkingMode ? '思考模式开启' : '普通模式' }}
          </span>
          <el-button v-if="isStreaming" size="small" @click="abort">中止</el-button>
          <el-button type="primary" size="small" :disabled="!canSend || !userInput.trim()" @click="onSend">
            发送
          </el-button>
        </div>
      </div>
    </section>

    <!-- 右：困惑度／复用趋势 -->
    <aside v-if="showPpl" class="panel ppl-panel">
      <div class="panel-title">
        <span>增量复用趋势</span>
        <el-button size="small" text @click="showPpl = false">×</el-button>
      </div>
      <div ref="pplEl" class="ppl-chart" />
      <p class="hint ppl-hint">纵轴为每轮复用的 token 数</p>
    </aside>
    <el-button v-else class="ppl-toggle" size="small" @click="showPpl = true">趋势</el-button>
  </div>
</template>

<style scoped>
.chat-layout {
  display: grid;
  grid-template-columns: 260px 1fr auto;
  gap: 12px;
  height: 100%;
}

.side {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.side-body {
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow: auto;
  flex: 1;
}

.full {
  width: 100%;
}

.param-box {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px;
  background: #f8f9fb;
  border-radius: 8px;
}

.param {
  display: flex;
  flex-direction: column;
  gap: 3px;
  font-size: 12px;
}

.param span {
  color: var(--foundry-text-dim);
}

.param :deep(.el-input-number) {
  width: 100%;
}

.stats-mini {
  background: #eef4ff;
  border-radius: 8px;
  padding: 10px;
}

.mini-row {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  margin-bottom: 4px;
}

.hint {
  font-size: 11px;
  color: var(--foundry-text-dim);
  line-height: 1.4;
}

.chat-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.head-actions {
  display: flex;
  gap: 2px;
}

.messages {
  flex: 1;
  overflow: auto;
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-height: 0;
}

.msg {
  display: flex;
  gap: 10px;
}

.msg-role {
  flex: 0 0 36px;
  font-size: 11px;
  color: var(--foundry-text-dim);
  padding-top: 3px;
}

.msg-body {
  flex: 1;
  min-width: 0;
}

.msg.user .msg-text {
  background: #eef4ff;
  border-radius: 8px;
  padding: 8px 12px;
}

.msg-text {
  font-size: 13.5px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}

.cursor {
  color: var(--foundry-accent);
  animation: blink 1s steps(2) infinite;
}

@keyframes blink {
  50% {
    opacity: 0;
  }
}

.think-block {
  margin-bottom: 6px;
  border-left: 2px solid #d7dce6;
  padding-left: 10px;
}

.think-toggle {
  display: flex;
  align-items: center;
  gap: 4px;
  border: none;
  background: transparent;
  color: var(--foundry-text-dim);
  font-size: 12px;
  cursor: pointer;
  padding: 2px 0;
}

.think-text {
  margin: 6px 0 0;
  font-size: 12.5px;
  line-height: 1.65;
  color: #6b7280;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: inherit;
}

.error-bar {
  background: #fef2f2;
  border-top: 1px solid #fecaca;
  color: #b91c1c;
  font-size: 12px;
  padding: 8px 16px;
}

.composer {
  border-top: 1px solid var(--foundry-border);
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.composer-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.composer-actions .hint {
  margin-right: auto;
}

.ppl-panel {
  width: 200px;
  display: flex;
  flex-direction: column;
}

.ppl-chart {
  height: 160px;
  padding: 6px;
}

.ppl-hint {
  padding: 0 10px 8px;
  margin: 0;
}

.ppl-toggle {
  align-self: start;
}

.dim {
  color: var(--foundry-text-dim);
  font-style: normal;
  font-weight: 400;
}
</style>
