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

import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Setting } from '@element-plus/icons-vue'
import * as echarts from 'echarts'

import { chatApi, defaultInferParams, roleplayParams } from '@/api/chat'
import { useBreakpoint } from '@/composables/useBreakpoint'
import { dataApi } from '@/api/data'
import { useChatStream } from '@/composables/useChatStream'

// ── 模型与会话 ───────────────────────────────────────────────────────────
const modelOptions = ref([])
const selectedModel = ref('')
const sessionName = ref('default')
const models = ref([]) // 可选模型（基底 + checkpoints）

const roleList = ref([])
const selectedRole = ref('none')

// ── 角色卡管理 ───────────────────────────────────────────────────────────
const roleDialog = ref(false)
const roleEditing = ref(null)   // null = 新建模式
const roleSaving = ref(false)
const roleForm = ref({ id: '', name: '', content: '' })

const currentRoleContent = computed(() => {
  if (selectedRole.value === 'none') return ''
  return roleList.value.find((r) => r.id === selectedRole.value)?.content || ''
})

const params = ref(defaultInferParams())
const roleId = computed(() => (selectedRole.value === 'none' ? null : selectedRole.value))

const { messages, isStreaming, isThinking, error, stats, canSend, send, complete, regenerate, abort, clear, loadMessages } =
  useChatStream({
    getModelRef: () => selectedModel.value,
    getParams: () => params.value,
    getRoleId: () => roleId.value,
    getSessionName: () => sessionName.value,
  })

// ── 模式：对话 / 续写 ────────────────────────────────────────────────────
const mode = ref('chat')
const completionText = ref('')
const completionPreset = ref('novel')
const completionRef = ref(null)

const completionPlaceholder = computed(() =>
  currentPreset.value?.prompt
    ? '已填入示例原文，可直接续写；生成结果会追加在下方，接着往下写即可'
    : '贴入需要续写的原文，点「续写」后结果会直接追加在这里',
)

/**
 * 续写场景预设 —— 从后端 /api/chat/completion-presets 拉取。
 *
 * 不在前端写死：后端才是唯一真相。这些预设迁移自 RWKV-Runner 的
 * defaultPresets，每个都带 prompt 示例、stop 序列、inject 模板，
 * 前端只需要 label 和 hint 用于展示。
 */
const COMPLETION_PRESETS = ref([])

const currentPreset = computed(
  () => COMPLETION_PRESETS.value.find((p) => p.key === completionPreset.value),
)
const currentPresetHint = computed(() => currentPreset.value?.hint || '')

async function loadCompletionPresets() {
  try {
    const res = await chatApi.completionPresets()
    const list = res?.presets || {}
    COMPLETION_PRESETS.value = Object.entries(list).map(([key, v]) => ({
      key,
      label: v.label || key,
      hint: v.hint || '',
      prompt: v.prompt || '',
      stop: v.stop || '',
      // inject_start 决定提交时是否要补 "\n\nAssistant: "
      // 这是让文本域与模型实际输入对齐的关键
      injectStart: v.inject_start || '',
      injectEnd: v.inject_end || '',
    }))
  } catch (err) {
    ElMessage.warning(`加载续写预设失败: ${err.message}`)
  }
}

/** 记录上次预设填入的内容，用于判断能否安全覆盖 */
const lastPresetPrompt = ref('')

/** 选中预设时把示例 prompt 填进输入框（若当前为空或用户确认覆盖） */
/**
 * 切换预设。
 *
 * ## 新交互下的语义变化
 *
 * 由于文本域现在是一块**持续生长的文档**（模型输出直接追加进去），
 * 切换预设时文本域里往往已有上一轮的完整内容。
 *
 * 这里不再静默判断"能否覆盖"，而是明确询问用户：
 *   - 文本域为空 → 直接填入示例
 *   - 文本域有内容 → 提示是否用新预设的示例替换
 *
 * 静默覆盖会丢用户正在写的东西，静默不覆盖又会让用户以为切换没生效。
 * 两种都不对，只能问。
 */
async function onPresetChange(key) {
  const p = COMPLETION_PRESETS.value.find((x) => x.key === key)
  if (!p) return

  const hasContent = completionText.value.trim().length > 0
  const sameAsPreset = completionText.value === p.prompt

  if (!hasContent || sameAsPreset) {
    if (p.prompt) completionText.value = p.prompt
    lastPresetPrompt.value = p.prompt
    return
  }

  // 已有其他内容：问一次
  try {
    await ElMessageBox.confirm(
      `文本域里已有内容。要用「${p.label}」的示例原文替换吗？\n\n` +
        '选「替换」会清空现有文本；选「保留」则沿用当前内容，只切换场景参数。',
      '切换续写场景',
      { confirmButtonText: '替换', cancelButtonText: '保留', type: 'warning' },
    )
    completionText.value = p.prompt || ''
    lastPresetPrompt.value = completionText.value
  } catch {
    /* 用户选择保留，什么都不做 */
  }
}

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

// ── 响应式 ───────────────────────────────────────────────────────────────
const { isMobile, needDrawer } = useBreakpoint()
const sideDrawer = ref(false)      // 左侧会话设置抽屉
const paramDrawer = ref(false)     // 右侧推理参数抽屉
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
  await loadCompletionPresets()
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

/**
 * 续写：对齐 RWKV-Runner 的交互。
 *
 * ## 与对话模式的根本区别
 *
 * 对话：输入框写问题 → 发出去 → 消息列表多一条问答
 * 续写：**只有一个文本域**，模型输出流式追加到它末尾，用户在那之后接着写
 *
 * 实现要点（对应 Runner 的 Completion.tsx）：
 *
 *    Runner:  setPrompt(prompt + answer + injectEnd)
 *    我们:    completionText += chunk          （逐 token 追加）
 *            completionText += inject_end     （结束时补下一轮入口）
 *
 * 因为有了 inject_end，用户续写完看到的是：
 *
 *      <原有文本><模型续写>
 *      User: |          ← 光标天然落在这里，直接往下写
 *
 * 不需要另开输入框，也不需要手动敲 "User: "。
 */
async function onComplete() {
  const presetCfg = COMPLETION_PRESETS.value.find((x) => x.key === completionPreset.value)
  const injStart = presetCfg?.injectStart || ''
  const injEnd = presetCfg?.injectEnd || ''

  // 对话式预设的结束标记（如 "\n\nUser: "）直接复用 inject_end；
  // 纯续写预设没有 inject_end，用普通换行分隔。
  const turnSep = injEnd || '\n'

  let base = completionText.value

  if (!base.trim()) {
    ElMessage.warning('请先贴入需要续写的原文')
    return
  }

  // ★ 方案 A 的核心：提交前把文本规范化成「模型将要消费的序列」。
  //
  // 后端 build_completion_prompt 收到 text 后执行：
  //     text.rstrip() + "\n\nAssistant: "     （对话式）
  //     text.rstrip() + "\n"                    （纯续写）
  //
  // 因此文本域必须在提交时也变成同样的形态，否则下一轮的 fed_text
  // 与前端文本前缀对不上，LCP 会在分隔符处断掉、复用率归零。
  //
  // 实测断点：
  //     fed  = '原文\n\nAssistant: 输出'
  //     front= '原文\n输出\n\nUser: ...'
  //            ↑ '\n\nAssistant:' vs '\n' → LCP 只剩 12
  base = base.replace(/\s+$/, '')
  const normalized = injStart ? `${base}\n\nAssistant: ` : `${base}\n`

  // 把规范化后的文本写回文本域 —— 用户看到的即模型看到的
  completionText.value = normalized
  scrollCompletionToEnd()

  // 流式阶段只追加模型输出；结束后追加下一轮入口标记
  let appended = ''

  const render = () => {
    completionText.value = normalized + appended
    scrollCompletionToEnd()
  }

  const ok = await complete(normalized, {
    preset: completionPreset.value,
    onChunk: (chunk) => {
      appended += chunk
      render()
    },
    onFinished: (r) => {
      // 先 rstrip 掉模型输出的尾随空白，再追加分隔符。
      // 不清理的话，模型每轮吐的尾随换行会逐轮累积（第N轮出现2N个空行）。
      appended = appended.replace(/\s+$/, '')
      appended += turnSep
      render()
    },
  })

  if (!ok && error.value) {
    ElMessage.error(error.value)
    // 失败回滚到规范化之前的内容，不让文本域留下半截
    completionText.value = base
    scrollCompletionToEnd()
  }
}

/** 让文本域滚动到末尾，用户能立刻看到新生成的内容 */
function scrollCompletionToEnd() {
  nextTick(() => {
    const el = completionRef.value?.textarea
    if (el) el.scrollTop = el.scrollHeight
  })
}

/** 切换模式时清空提示，避免把对话报错带到续写界面 */
function switchMode(m) {
  if (isStreaming.value) {
    ElMessage.warning('生成中，请先中止或等待完成')
    return
  }
  mode.value = m
  error.value = ''
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
      '清空会删除该会话的持久化对话记录，并重置模型的循环态。刷新后不会恢复。确定继续？',
      '清空对话',
      { type: 'warning', confirmButtonText: '清空' },
    )
    await clear()
    pplPoints.value = []
    updatePplChart()
    ElMessage.success('对话记录已删除，模型状态已重置')
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

// ── 角色卡管理 ───────────────────────────────────────────────────────────
async function loadRoleList() {
  try {
    const roles = await dataApi.listRoles()
    roleList.value = [{ id: 'none', name: '无（普通对话）', content: '' }, ...(roles || [])]
  } catch (err) {
    ElMessage.error(`加载角色卡失败: ${err.message}`)
  }
}

function openRoleManager() {
  roleEditing.value = null
  roleForm.value = { id: '', name: '', content: '' }
  roleDialog.value = true
}

function startNewRole() {
  roleEditing.value = null
  roleForm.value = { id: '', name: '', content: '' }
}

function editRole(role) {
  if (role.id === 'none') {
    ElMessage.info('「无」是内置占位项，不可编辑')
    return
  }
  roleEditing.value = role
  roleForm.value = { id: role.id, name: role.name, content: role.content }
  roleDialog.value = true
}

async function submitRole() {
  const name = roleForm.value.name.trim()
  const content = roleForm.value.content.trim()
  if (!name) {
    ElMessage.warning('请填写角色名称')
    return
  }
  if (!content) {
    ElMessage.warning('请填写角色设定')
    return
  }

  roleSaving.value = true
  try {
    // 新建时 id 留空，由后端生成
    const saved = await dataApi.saveRole({
      id: roleEditing.value ? roleForm.value.id : undefined,
      name,
      content,
    })
    ElMessage.success(roleEditing.value ? `角色「${name}」已更新` : `角色「${name}」已创建`)
    await loadRoleList()
    // 新建/编辑后自动选中该角色
    if (saved?.id) {
      selectedRole.value = saved.id
      onRoleChange(saved.id)
    }
    roleDialog.value = false
    roleEditing.value = null
  } catch (err) {
    ElMessage.error(`保存失败: ${err.message}`)
  } finally {
    roleSaving.value = false
  }
}

async function removeRole(role) {
  if (role.id === 'none') return
  try {
    await ElMessageBox.confirm(`确定删除角色「${role.name}」？此操作不可撤销。`, '删除角色', {
      type: 'warning',
      confirmButtonText: '删除',
    })
    await dataApi.deleteRole(role.id)
    ElMessage.success('角色已删除')
    if (selectedRole.value === role.id) {
      selectedRole.value = 'none'
      onRoleChange('none')
    }
    await loadRoleList()
  } catch (err) {
    if (err !== 'cancel') ElMessage.error(`删除失败: ${err.message}`)
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

/**
 * 收起/展开趋势图。
 *
 * 必须手动 resize：v-show 隐藏时容器宽高为 0，echarts 内部会记下 0 尺寸，
 * 重新展开后画布仍是 0x0（表现为图表"消失"）。等 DOM 显示完成再 resize 一次。
 */
function togglePpl() {
  showPpl.value = !showPpl.value
  if (showPpl.value) {
    nextTick(() => pplChart?.resize())
  }
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
    <!-- 左：会话设置（窄屏时 CSS 转为抽屉浮层，DOM 不复制） -->
    <div v-if="needDrawer && sideDrawer" class="drawer-mask" @click="sideDrawer = false" />
    <aside class="panel side" :class="{ 'as-drawer': needDrawer, open: sideDrawer }">
      <div class="panel-title">
        <span>会话</span>
        <el-button v-if="needDrawer" size="small" text @click="sideDrawer = false">×</el-button>
      </div>
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
            <div class="role-row">
              <el-select v-model="selectedRole" class="role-select" @change="onRoleChange">
                <el-option v-for="r in roleList" :key="r.id" :label="r.name" :value="r.id" />
              </el-select>
              <el-button size="small" :icon="Setting" @click="openRoleManager" />
            </div>
            <small v-if="currentRoleContent" class="hint role-preview">
              {{ currentRoleContent.slice(0, 60) }}{{ currentRoleContent.length > 60 ? '…' : '' }}
            </small>
          </el-form-item>
        </el-form>

        <el-switch v-model="thinkingMode" active-text="思考模式" class="full" />

        <el-divider />

        <div class="ppl-inline">
          <div class="ppl-head">
            <span>增量复用趋势</span>
            <el-button size="small" text @click="togglePpl">
              {{ showPpl ? '×' : '展开' }}
            </el-button>
          </div>
          <!-- 用 v-show 而非 v-if：图表 DOM 必须常驻，
               否则重新展开时 echarts 容器丢失、initPplChart 不会再跑 -->
          <div v-show="showPpl" ref="pplEl" class="ppl-chart" />
          <p v-show="showPpl" class="hint">纵轴为每轮复用的 token 数</p>
        </div>

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
        <div class="title-left">
          <!-- 窄屏：两个侧栏都收进抽屉，用这两个按钮唤出 -->
          <el-button v-if="needDrawer" size="small" text class="drawer-btn" @click="sideDrawer = true">
            <el-icon><Setting /></el-icon>
          </el-button>
          <span class="model-label" :title="selectedModel">
            {{ selectedModel || '未选择模型' }} <em class="dim">/ {{ sessionName }}</em>
          </span>
          <el-radio-group :model-value="mode" size="small" class="mode-switch" @update:model-value="switchMode">
            <el-radio-button value="chat">对话</el-radio-button>
            <el-radio-button value="completion">续写</el-radio-button>
          </el-radio-group>
        </div>
        <el-button v-if="needDrawer" size="small" text class="drawer-btn" @click="paramDrawer = true">
          <el-icon><Odometer /></el-icon>
        </el-button>
        <div v-if="mode === 'chat'" class="head-actions">
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

      <!--
        续写模式**不显示消息列表** —— 文本域本身就是完整内容。
        若两者同时出现，用户会看到文本域和消息列表两份重复的东西。
      -->
      <div v-if="mode === 'chat'" ref="messagesEl" class="messages">
        <p v-if="!messages.length" class="empty-hint">开始一段新对话</p>

        <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
          <div class="msg-role">{{ m.role === 'user' ? '你' : '模型' }}</div>

          <div class="msg-body">
            <!-- 续写：原文折叠区 -->
            <div v-if="m.kind === 'completion' && m.source" class="source-block">
              <button class="source-toggle" @click="m.showSource = !m.showSource">
                <el-icon><component :is="m.showSource ? 'ArrowDown' : 'ArrowRight'" /></el-icon>
                <span>续写原文（{{ m.source.length }} 字）</span>
              </button>
              <pre v-if="m.showSource" class="source-text">{{ m.source }}</pre>
            </div>

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
        <!-- 对话模式 -->
        <template v-if="mode === 'chat'">
          <el-input
            v-model="userInput"
            type="textarea"
            :rows="3"
            resize="none"
            placeholder="输入消息，Enter 发送，Shift+Enter 换行"
            :disabled="!selectedModel"
            @keydown="onKeydown"
          />
        </template>

        <!-- 续写模式：单一大文本域，对齐 RWKV-Runner -->
        <template v-else>
          <div class="completion-bar">
            <el-select v-model="completionPreset" size="small" class="preset-select" @change="onPresetChange">
              <el-option
                v-for="p in COMPLETION_PRESETS"
                :key="p.key"
                :label="p.label"
                :value="p.key"
              />
            </el-select>
            <span class="hint preset-hint">{{ currentPresetHint }}</span>
            <el-button size="small" text @click="completionText = ''" :disabled="!completionText">
              清空
            </el-button>
          </div>

          <!--
            ★ 这是对齐 RWKV-Runner 的核心改动。

            整段文本域就是"对话本身"：
              1. 预设把示例 prompt 填进来
              2. 点"续写"后，模型输出**流式追加到这个文本域**
              3. 生成结束后自动补 inject_end（如 "\n\nUser: "）
              4. 用户直接在末尾接着写，再点"续写" —— 不需要另开输入框

            这样交互是"一条不断生长的文档"，而不是"提问-回答"。
          -->
          <el-input
            ref="completionRef"
            v-model="completionText"
            type="textarea"
            :rows="14"
            resize="none"
            class="completion-area"
            :placeholder="completionPlaceholder"
            :disabled="!selectedModel"
          />
        </template>

        <div class="composer-actions">
          <span class="hint">
            {{ thinkingMode ? '思考模式开启' : '普通模式' }}
          </span>
          <el-button v-if="isStreaming" size="small" @click="abort">中止</el-button>
          <el-button
            v-if="mode === 'chat'"
            type="primary"
            size="small"
            :disabled="!canSend || !userInput.trim()"
            @click="onSend"
          >
            发送
          </el-button>
          <el-button
            v-else
            type="primary"
            size="small"
            :disabled="!canSend || !completionText.trim()"
            @click="onComplete"
          >
            续写
          </el-button>
        </div>
      </div>
    </section>

    <!-- 右：推理参数（窄屏同样转抽屉浮层） -->
    <div v-if="needDrawer && paramDrawer" class="drawer-mask" @click="paramDrawer = false" />
    <aside class="panel param-panel" :class="{ 'as-drawer': needDrawer, open: paramDrawer }">
      <div class="panel-title">
        <span>推理参数</span>
        <div class="title-actions">
          <el-button size="small" text @click="saveParams">应用</el-button>
          <el-button v-if="needDrawer" size="small" text @click="paramDrawer = false">×</el-button>
        </div>
      </div>

      <div class="param-body">
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
      </div>
    </aside>

    <!-- 角色卡管理 -->
    <el-dialog
      v-model="roleDialog"
      :title="roleEditing ? `编辑角色：${roleEditing.name}` : '管理角色卡'"
      width="760px"
      top="6vh"
    >
      <div class="role-mgr">
        <!-- 左：角色列表 -->
        <div class="role-list-pane">
          <div class="role-list-head">
            <span>已有角色（{{ roleList.length }}）</span>
            <el-button size="small" type="primary" plain @click="startNewRole">新建</el-button>
          </div>
          <div class="role-list">
            <div
              v-for="r in roleList"
              :key="r.id"
              class="role-item"
              :class="{ active: roleEditing && roleEditing.id === r.id, builtin: r.id === 'none' }"
              @click="r.id !== 'none' && editRole(r)"
            >
              <div class="role-item-name">{{ r.name }}</div>
              <div class="role-item-ops" v-if="r.id !== 'none'">
                <el-button size="small" text @click.stop="editRole(r)">改</el-button>
                <el-button size="small" text type="danger" @click.stop="removeRole(r)">删</el-button>
              </div>
            </div>
          </div>
        </div>

        <!-- 右：编辑区 -->
        <div class="role-edit-pane">
          <el-form label-position="top" size="small">
            <el-form-item label="角色名称">
              <el-input v-model="roleForm.name" placeholder="例如：助手 / 猫娘 / 翻译官" maxlength="40" show-word-limit />
            </el-form-item>
            <el-form-item label="角色设定（会作为 System 提示注入）">
              <el-input
                v-model="roleForm.content"
                type="textarea"
                :rows="14"
                placeholder="描述这个角色的身份、语气、行为准则…"
              />
            </el-form-item>
          </el-form>
          <div class="role-actions">
            <el-button size="small" @click="startNewRole">清空表单</el-button>
            <el-button size="small" type="primary" :loading="roleSaving" @click="submitRole">
              {{ roleEditing ? '保存修改' : '新建角色' }}
            </el-button>
          </div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<style scoped>
.chat-layout {
  display: grid;
  /* 左：会话+趋势；中：对话；右：推理参数 */
  grid-template-columns: 260px 1fr 280px;
  gap: 12px;
  height: 100%;
}

@media (max-width: 1280px) {
  .chat-layout {
    grid-template-columns: 232px 1fr 248px;
  }
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
  flex: 0 0 auto;
}

/*
 * 头部兜底：任何子项溢出都不得撑开父容器。
 *
 * .panel-title 在 index.css 里是全局样式（justify-content: space-between，
 * 无 overflow 约束）。窄屏时「齿轮按钮 + 长模型名 + 模式切换 + 仪表按钮」
 * 一排摆不下，会把整个卡片顶宽，进而让 .app-main 出现横向滚动条。
 */
.chat-panel .panel-title {
  overflow: hidden;
  gap: 4px;
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

/*
 * 续写模式：composer 必须撑满整个卡片剩余空间。
 *
 * 之前文本域只占下半部分，上半留白 —— 因为 .chat-panel 是 flex 列布局，
 * .messages 隐藏后 composer 仍按内容高度排布，不会自动伸展。
 * 这里用 :has() 精确选中"含续写文本域的 composer"，让它 flex:1 吃掉所有空间。
 */
.chat-panel:has(.completion-area) .composer {
  flex: 1;
  min-height: 0;
}

/* 续写模式下文本域本身也要能被 flex 拉伸 */
.chat-panel:has(.completion-area) .completion-area {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.chat-panel:has(.completion-area) .completion-area :deep(.el-textarea__inner) {
  flex: 1;
  height: auto;
}

.composer-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.composer-actions .hint {
  margin-right: auto;
}

/* ── 左侧内嵌趋势图 ───────────────────────────────────────────────── */
.ppl-inline {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.ppl-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  font-weight: 600;
  color: var(--foundry-text-dim);
}

.ppl-chart {
  height: 150px;
  margin: 0 -6px;
}

/* ── 右侧推理参数面板 ─────────────────────────────────────────────── */
.param-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.param-body {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.param-toggle {
  align-self: start;
}

.dim {
  color: var(--foundry-text-dim);
  font-style: normal;
  font-weight: 400;
}

/* ── 角色卡选择行 ─────────────────────────────────────────────────────── */
.role-row {
  display: flex;
  gap: 6px;
  align-items: center;
  width: 100%;
}

.role-select {
  flex: 1;
  min-width: 0;
}

.role-preview {
  display: block;
  margin-top: 4px;
  color: var(--foundry-text-dim);
  line-height: 1.4;
  word-break: break-all;
}

/* ── 角色卡管理对话框 ─────────────────────────────────────────────────── */
.role-mgr {
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 14px;
  min-height: 380px;
}

.role-list-pane {
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--foundry-border);
  padding-right: 12px;
  min-height: 0;
}

.role-list-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  color: var(--foundry-text-dim);
  margin-bottom: 8px;
}

.role-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.role-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 7px 10px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
}

.role-item:hover {
  background: #f2f4f7;
}

.role-item.active {
  background: #fdeee8;
  color: #b8432a;
  font-weight: 600;
}

.role-item.builtin {
  cursor: default;
  color: var(--foundry-text-dim);
}

.role-item-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.role-item-ops {
  display: flex;
  gap: 0;
  flex: 0 0 auto;
}

.role-edit-pane {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.role-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: auto;
  padding-top: 10px;
}

/* ── 模式切换 ─────────────────────────────────────────────────────────── */
/*
 * flex:1 + min-width:0 是让内部文本能省略的关键组合。
 *
 * 只写 min-width:0 不够 —— flex 项默认 flex:0 1 auto，
 * 基准宽度仍取内容宽度，模型名一长就把父容器撑开。
 * 显式 flex:1 后才能让 1fr 生效，配合子项的 ellipsis 才不会溢出。
 */
.title-left {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
  flex: 1 1 auto;
  overflow: hidden;
}

.mode-switch {
  flex: 0 0 auto;
}

/* ── 续写输入区 ───────────────────────────────────────────────────────── */
.completion-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 2px;
}

.preset-select {
  width: 132px;
  flex: 0 0 auto;
}

.preset-hint {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 续写大文本域：独占剩余空间，像一块不断生长的文档 */
.completion-area {
  flex: 1;
  min-height: 0;
}

.completion-area :deep(.el-textarea__inner) {
  height: 100%;
  font-family: var(--foundry-mono);
  font-size: 13px;
  line-height: 1.7;
}

/* 续写原文折叠区 */
.source-block {
  margin-bottom: 6px;
  border-left: 2px solid #c7d2e0;
  padding-left: 10px;
}

.source-toggle {
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

.source-text {
  margin: 6px 0 0;
  font-size: 12.5px;
  line-height: 1.65;
  color: #6b7280;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 200px;
  overflow-y: auto;
  font-family: inherit;
}

/* ═══════════════════════════════════════════════════════════════════
   Chat 移动端适配
   ═══════════════════════════════════════════════════════════════════ */

/* 模型名在窄屏可能很长，允许收缩 */
.model-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  /* flex:1 让它在 title-left 内部吃剩余空间；min-width:0 允许缩到 0 */
  flex: 1 1 auto;
  min-width: 0;
}

.drawer-btn {
  flex: 0 0 auto;
  padding: 4px 6px;
}

.title-actions {
  display: flex;
  align-items: center;
  gap: 2px;
}

/* 抽屉遮罩：点击空白处关闭 */
.drawer-mask {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  z-index: 3000;
}

/*
 * 窄屏：侧栏从「栅格列」变成「浮层抽屉」。
 *
 * 关键在 position:fixed + z-index，配合 .drawer-mask 实现遮罩点击关闭。
 * 这样 DOM 完全不用复制 —— 桌面和移动共用同一份模板，
 * 只是定位方式不同。旧项目那套 mobile/ 目录就是复制出来的，维护成本翻倍。
 */
.as-drawer {
  position: fixed;
  top: 0;
  bottom: 0;
  z-index: 3001;
  width: min(86vw, 340px);
  border-radius: 0;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.32);
  transition: transform 0.24s ease;
  overflow: hidden;
}

.side.as-drawer {
  left: 0;
  transform: translateX(-102%);
}

.side.as-drawer.open {
  transform: translateX(0);
}

.param-panel.as-drawer {
  right: 0;
  transform: translateX(102%);
}

.param-panel.as-drawer.open {
  transform: translateX(0);
}

@media (max-width: 1099px) {
  .chat-layout {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 767px) {
  .chat-layout {
    gap: 8px;
    /* grid 项同样要 min-width:0，否则 1fr 被内容撑开 */
    min-width: 0;
  }

  .chat-layout > * {
    min-width: 0;
  }

  /*
   * 手机上头部空间只有 390px 左右，
   * 「齿轮 + 模型名 + 模式切换 + 仪表」一排摆不下。
   * 让 title-left 换行：模型名占一行，模式切换落到第二行。
   */
  .chat-panel .panel-title {
    flex-wrap: wrap;
    row-gap: 6px;
    padding: 10px 12px;
  }

  .title-left {
    flex-wrap: wrap;
    gap: 8px;
    flex: 1 1 100%;
  }

  .model-label {
    flex: 1 1 auto;
    max-width: calc(100% - 90px);
  }

  .mode-switch {
    flex: 1 1 100%;
    display: flex;
  }

  .mode-switch :deep(.el-radio-button) {
    flex: 1;
  }

  .mode-switch :deep(.el-radio-button__inner) {
    width: 100%;
    padding: 5px 10px;
    font-size: 12px;
  }

  .messages {
    padding: 10px;
    gap: 10px;
  }

  .msg-role {
    flex: 0 0 28px;
    font-size: 10px;
  }

  .msg-text {
    font-size: 14px;
    line-height: 1.65;
  }

  .completion-area :deep(.el-textarea__inner) {
    font-size: 12.5px;
  }

  .param-body {
    padding: 10px;
  }

}

</style>
