/**
 * 流式对话 composable。
 *
 * 从旧版 Chat.vue（1718 行）里剥离出来的核心逻辑。旧版把 SSE 解析、
 * thinking/answer 实时切分、参数面板、图表全塞在一个组件里，无法复用也无法测试。
 *
 * ## thinking 实时切分的关键
 *
 * 模型的原始输出是连续 token 流，形如：
 *     <think>推理过程</think>正式回答
 *
 * 逐 token 到达时，我们**无法预知** `</think>` 何时出现。因此策略是：
 *   1. 累积全部原文 rawText
 *   2. 每次增量都从 rawText 重新解析（而不是增量拼接已解析结果）
 *   3. 未见到 `</think>` 前，thinking 是 growing 的，answer 为 ''
 *
 * 第 2 点是关键。若增量拼接已切好的 thinking/answer，一旦 `</think>` 跨 chunk
 * 到达，已经错位的分界无法回退 —— 这正是旧版「思考内容突然跳到正文」的根因。
 *
 * ## 为什么解析用字符串拼接而不是正则
 *
 * `indexOf` 是 O(n)，但每轮只调两次且 n 是本轮输出长度（通常 < 4K 字符）。
 * 正则更慢且容易在跨 chunk 边界出错。
 */

import { computed, ref, shallowRef } from 'vue'
import { chatApi } from '../api/chat'
import { dataApi } from '../api/data'

const THINK_START = '<' + 'think' + '>'
const THINK_END = '<' + '/think' + '>'

/**
 * 从累积原文中切分 thinking 与 answer。
 * @param {string} raw
 * @returns {{thinking: string, answer: string, thinkingEnded: boolean}}
 */
export function splitStream(raw) {
  const endIdx = raw.indexOf(THINK_END)

  if (endIdx !== -1) {
    let thinking = raw.slice(0, endIdx)
    if (thinking.startsWith(THINK_START)) thinking = thinking.slice(THINK_START.length)
    return {
      thinking: trimEdges(thinking),
      answer: trimEdges(raw.slice(endIdx + THINK_END.length)),
      thinkingEnded: true,
    }
  }

  const startIdx = raw.indexOf(THINK_START)
  if (startIdx !== -1) {
    return {
      thinking: trimEdges(raw.slice(startIdx + THINK_START.length)),
      answer: '',
      thinkingEnded: false,
    }
  }

  // 完全没有 think 标记：整段都是正文
  return { thinking: '', answer: raw, thinkingEnded: false }
}

function trimEdges(s) {
  return s.replace(/^\n+/, '').replace(/\n+$/, '')
}

/**
 * @param {object} opts
 * @param {() => string} opts.getModelRef 当前模型引用
 */
export function useChatStream({ getModelRef, getParams, getRoleId, getSessionName }) {
  /** @type {import('vue').Ref<Array>} */
  const messages = ref([])
  const isStreaming = ref(false)
  const isThinking = ref(false)
  const error = ref('')
  const stats = shallowRef(null)

  let controller = null

  const canSend = computed(() => !!getModelRef() && !isStreaming.value)

  function lastAssistant() {
    const last = messages.value[messages.value.length - 1]
    return last && last.role === 'assistant' ? last : null
  }

  /** 把消息数组转成后端 history 格式（不含本轮用户消息） */
  function historyPayload(exclude) {
    const src = exclude === undefined ? messages.value : messages.value.slice(0, exclude)
    return src
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => ({
        role: m.role,
        content: m.content || '',
        thinking: m.thinking || '',
      }))
  }

  function beginAssistantMessage() {
    messages.value.push({
      role: 'assistant',
      content: '',
      thinking: '',
      visibleThinking: '',
      showThinking: true,
      streaming: true,
      raw: '',
    })
    isStreaming.value = true
    isThinking.value = true
    error.value = ''
    stats.value = null
  }

  function applyChunk(chunk) {
    const msg = lastAssistant()
    if (!msg) return

    msg.raw += chunk
    const { thinking, answer, thinkingEnded } = splitStream(msg.raw)

    msg.thinking = thinking
    msg.content = answer
    // 只在首次解析出内容时自动展开，之后尊重用户手动的折叠状态
    if (thinking && msg.visibleThinking === '') msg.showThinking = true
    msg.visibleThinking = thinking

    if (thinkingEnded) isThinking.value = false
  }

  function finalize() {
    const msg = lastAssistant()
    if (msg) {
      msg.streaming = false
      delete msg.raw
    }
    isStreaming.value = false
    isThinking.value = false
    controller = null
  }

  /**
   * 发送消息。
   * @param {string} content
   * @returns {Promise<boolean>} 是否成功
   */
  async function send(content) {
    const text = (content || '').trim()
    if (!text || !getModelRef() || isStreaming.value) return false

    // history 必须在 push 本轮用户消息之前取
    const history = historyPayload()
    messages.value.push({ role: 'user', content: text })
    beginAssistantMessage()

    controller = new AbortController()
    try {
      await chatApi.stream(
        {
          modelRef: getModelRef(),
          message: text,
          history,
          params: getParams(),
          thinkingMode: !!getParams()?.thinking_mode,
          sessionName: getSessionName?.() || '',
          roleId: getRoleId?.() || null,
        },
        {
          onContent: applyChunk,
          onResult: (r) => {
            stats.value = {
              tokensFed: r.tokens_fed,
              tokensReused: r.tokens_reused,
              stopped: r.stopped,
            }
            // 后端返回的是权威切分结果，覆盖实时解析（可能因截断略有误差）
            const msg = lastAssistant()
            if (msg) {
              msg.thinking = r.thinking || ''
              msg.content = r.content || ''
              msg.visibleThinking = msg.thinking
            }
          },
          onError: (msg) => {
            error.value = msg
          },
          onDone: finalize,
          signal: controller.signal,
        },
      )
    } catch (err) {
      error.value = err.message || '发送失败'
      finalize()
      return false
    }
    return !error.value
  }

  /**
   * 续写：把给定原文交给模型接着写。
   *
   * ## 与 send() 的差别
   *
   * send() 是"提问—回答"：原文被包成 ``User:``，模型以 ``Assistant:`` 作答。
   * complete() 是"文档接续"：原文直接作为 prompt，模型在其后natural继续。
   *
   * 因此续写**不往 messages 里塞 user 消息**（那会让界面显示成你在提问），
   * 而是塞一条带 ``kind: 'completion'`` 标记的 assistant 消息，
   * 并把原文放在 ``source`` 里供界面折叠展示。
   *
   * @param {string} text  待续写的原文
   * @param {object} opts
   * @param {'novel'|'scene'|'story'} [opts.preset]
   * @returns {Promise<boolean>}
   */
  async function complete(text, { preset = 'novel' } = {}) {
    const src = (text || '').trim()
    if (!src || !getModelRef() || isStreaming.value) return false

    // ★ 收集之前的续写轮次作为历史。
    //
    // 这里修过一个真实缺陷：原本每轮只发当前这一句，后端又重置 state，
    // 导致多轮剧情完全断裂。用户第二轮说"我对那本《太清剑诀》产生兴趣"，
    // 模型看不到第一轮的"你扮演文本冒险游戏"设定，退化成聊天助手腔。
    //
    // 历史只取 kind === 'completion' 的消息，且**只取同一 preset** ——
    // 切换场景等于新开一局，不该把上一个场景的剧情带过来。
    const history = []
    for (const m of messages.value) {
      if (m.kind !== 'completion' || m.preset !== preset) continue
      if (m.source) history.push({ role: 'user', content: m.source })
      if (m.content) history.push({ role: 'assistant', content: m.content })
    }

    // 续写产物单独一条消息，原文挂在 source 上（界面可折叠显示）
    messages.value.push({
      role: 'assistant',
      content: '',
      thinking: '',
      visibleThinking: '',
      showThinking: true,
      streaming: true,
      raw: '',
      kind: 'completion',
      source: src,
      preset,
      showSource: false,
    })
    isStreaming.value = true
    isThinking.value = true
    error.value = ''
    stats.value = null

    controller = new AbortController()
    try {
      await chatApi.complete(
        {
          modelRef: getModelRef(),
          text: src,
          preset,
          params: getParams(),
          thinkingMode: !!getParams()?.thinking_mode,
          sessionName: getSessionName?.() || '',
          roleId: getRoleId?.() || null,
          history,
        },
        {
          onContent: applyChunk,
          onResult: (r) => {
            stats.value = {
              tokensFed: r.tokens_fed,
              tokensReused: r.tokens_reused,
              stopped: r.stopped,
            }
            const msg = lastAssistant()
            if (msg) {
              msg.thinking = r.thinking || ''
              msg.content = r.content || ''
              msg.visibleThinking = msg.thinking
            }
          },
          onError: (m) => {
            error.value = m
          },
          onDone: finalize,
          signal: controller.signal,
        },
      )
    } catch (err) {
      error.value = err.message || '续写失败'
      finalize()
      return false
    }
    return !error.value
  }

  /**
   * 重生成最后一条回复。
   *
   * 语义：移除上轮 assistant 消息 → 用同一条 user 消息重发 → 后端回滚 state。
   * 这样模型会"忘记"上一条回复，而不是在它基础上续写。
   */
  async function regenerate() {
    if (isStreaming.value || messages.value.length < 2) return false

    const last = messages.value[messages.value.length - 1]
    if (last.role !== 'assistant') return false

    const userMsg = messages.value[messages.value.length - 2]
    if (userMsg.role !== 'user') return false

    // 移除上轮回复，history 取到 user 消息之前的部分
    messages.value.pop()
    const history = historyPayload(messages.value.length - 1)
    beginAssistantMessage()

    controller = new AbortController()
    try {
      await chatApi.regenerate(
        {
          modelRef: getModelRef(),
          message: userMsg.content,
          history,
          params: getParams(),
          thinkingMode: !!getParams()?.thinking_mode,
          sessionName: getSessionName?.() || '',
          roleId: getRoleId?.() || null,
        },
        {
          onContent: applyChunk,
          onResult: (r) => {
            stats.value = {
              tokensFed: r.tokens_fed,
              tokensReused: r.tokens_reused,
              stopped: r.stopped,
            }
            const msg = lastAssistant()
            if (msg) {
              msg.thinking = r.thinking || ''
              msg.content = r.content || ''
              msg.visibleThinking = msg.thinking
            }
          },
          onError: (m) => {
            error.value = m
          },
          onDone: finalize,
          signal: controller.signal,
        },
      )
    } catch (err) {
      error.value = err.message || '重生成失败'
      finalize()
      return false
    }
    return !error.value
  }

  function abort() {
    controller?.abort()
    controller = null
  }

  /**
   * 清空对话：**同时清数据库记录和模型循环态**。
   *
   * ## 这里修过一个真实缺陷
   *
   * 原实现只调 resetState（清内存里的 state），完全没动数据库。
   * 于是用户清空后一刷新，前端 loadHistory 又从 chat-data 读回来 ——
   * 表现为「清空无效，内容自己长回来」。
   *
   * 正确的清空要做两件事：
   *   1. DELETE /api/data/chat-data  删持久化记录（否则刷新就复活）
   *   2. POST /api/chat/reset-state  清模型 state（否则模型仍记得上下文）
   *
   * ## 顺序：先删库，后重置 state
   *
   * 若删库失败，用户应看到错误而不是"假装已清空"——所以删库放在前面，
   * 失败就直接抛出，不清 UI。反过来若先重置 state 再删库失败，
   * 用户会以为清空成功但刷新后内容回来，更难排查。
   *
   * 删库返回 404（本就没有历史记录）属于正常情况，不算失败。
   */
  async function clear() {
    abort()

    const modelRef = getModelRef()
    const session = getSessionName?.() || ''

    if (modelRef) {
      try {
        await dataApi.deleteChatData(modelRef, session)
      } catch (err) {
        // 404 = 本来就没有记录，可以继续清 UI
        if (err?.status !== 404) {
          error.value = `清空失败: ${err.message}`
          throw err
        }
      }
    }

    // 数据库已清，再清模型循环态
    if (modelRef) {
      try {
        await chatApi.resetState(modelRef, session)
      } catch {
        /* 模型态重置失败不影响数据已删除这一事实 */
      }
    }

    messages.value = []
    stats.value = null
    error.value = ''
  }

  function loadMessages(list) {
    messages.value = (list || []).map((m) => ({
      role: m.role,
      content: m.content || '',
      thinking: m.thinking || '',
      visibleThinking: m.thinking || '',
      showThinking: false,
      streaming: false,
    }))
  }

  return {
    messages,
    isStreaming,
    isThinking,
    error,
    stats,
    canSend,
    send,
    complete,
    regenerate,
    abort,
    clear,
    loadMessages,
  }
}
