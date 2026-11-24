/**
 * HTTP 客户端 —— 全前端**唯一**发请求的地方。
 *
 * ## 为什么必须集中
 *
 * 旧版前端 6420 行里散落 49 处 `axios.` 调用，没有 api 层。后果：
 *   - 同一个端点在不同页面用不同参数格式
 *   - 错误处理各写一遍，有的 catch 了有的没 catch
 *   - 后端改字段前端不知道，运行时才炸
 *
 * 现在：所有请求经此，统一 baseURL、统一错误提取、统一超时。
 */

const BASE = import.meta.env.VITE_API_BASE || ''

/** 后端错误统一格式：FastAPI 的 {"detail": "..."} */
export class ApiError extends Error {
  constructor(message, { status = 0, url = '', cause } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.url = url
    this.cause = cause
  }
}

function extractDetail(data, fallback) {
  if (!data) return fallback
  if (typeof data === 'string') return data
  if (typeof data.detail === 'string') return data.detail
  // Pydantic 校验错误是数组
  if (Array.isArray(data.detail)) {
    return data.detail
      .map((d) => `${(d.loc || []).slice(1).join('.') || '参数'}: ${d.msg}`)
      .join('; ')
  }
  return fallback
}

async function request(method, path, { params, body, timeout = 30000, signal } = {}) {
  const url = new URL(path, BASE || window.location.origin)

  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v === undefined || v === null || v === '') continue
      url.searchParams.set(k, v)
    }
  }

  const init = {
    method,
    headers: {},
    signal,
  }
  if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json'
    init.body = JSON.stringify(body)
  }

  // 手动超时：fetch 本身不支持 timeout
  const ctl = new AbortController()
  const timer = setTimeout(() => ctl.abort(), timeout)
  if (signal) signal.addEventListener('abort', () => ctl.abort(), { once: true })
  init.signal = ctl.signal

  let res
  try {
    res = await fetch(url.pathname + url.search, init)
  } catch (err) {
    clearTimeout(timer)
    if (err.name === 'AbortError') {
      throw new ApiError(`请求超时或被取消: ${method} ${path}`, { url: path, cause: err })
    }
    throw new ApiError(`网络错误: ${err.message}`, { url: path, cause: err })
  }
  clearTimeout(timer)

  const text = await res.text()
  let data = null
  if (text) {
    try {
      data = JSON.parse(text)
    } catch {
      data = text
    }
  }

  if (!res.ok) {
    throw new ApiError(extractDetail(data, `请求失败 (${res.status})`), {
      status: res.status,
      url: path,
    })
  }
  return data
}

export const http = {
  get: (p, opts) => request('GET', p, opts),
  post: (p, opts) => request('POST', p, opts),
  put: (p, opts) => request('PUT', p, opts),
  del: (p, opts) => request('DELETE', p, opts),
}

/**
 * SSE 流式请求。
 *
 * 为什么不用 EventSource：它只支持 GET，而对话需要 POST body。
 * 因此用 fetch + ReadableStream 手工解析 SSE。
 *
 * @param {string} path
 * @param {object} body
 * @param {object} handlers
 * @param {(chunk: string) => void} handlers.onContent  增量文本
 * @param {(result: object) => void} handlers.onResult  最终结果（含 thinking 拆分）
 * @param {(msg: string) => void} handlers.onError
 * @param {() => void} handlers.onDone
 * @param {AbortSignal} handlers.signal
 */
export async function streamSSE(path, body, { onContent, onResult, onError, onDone, signal } = {}) {
  const url = new URL(path, BASE || window.location.origin)

  let res
  try {
    res = await fetch(url.pathname + url.search, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
  } catch (err) {
    if (err.name === 'AbortError') {
      onDone?.()
      return
    }
    onError?.(`网络错误: ${err.message}`)
    return
  }

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    let msg = `请求失败 (${res.status})`
    try {
      msg = extractDetail(JSON.parse(text), msg)
    } catch {
      /* 保持默认 */
    }
    onError?.(msg)
    return
  }

  if (!res.body) {
    onError?.('浏览器不支持流式响应')
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      // SSE 以空行分隔事件；最后一段可能是半行，留到下次
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const raw of lines) {
        const line = raw.trim()
        if (!line.startsWith('data:')) continue

        const payload = line.slice(5).trim()
        if (!payload) continue
        if (payload === '[DONE]' || payload === '[FINAL]') continue

        let obj
        try {
          obj = JSON.parse(payload)
        } catch {
          continue // 非法 JSON 忽略，不打断流
        }

        if (obj.error) {
          onError?.(obj.error)
          continue
        }
        // 后端只在结尾发一次 result（含 thinking 拆分与 token 统计）
        if (obj.result) onResult?.(obj.result)
        else if (obj.content) onContent?.(obj.content)
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') onError?.(`流式读取失败: ${err.message}`)
  } finally {
    try {
      reader.releaseLock()
    } catch {
      /* ignore */
    }
    onDone?.()
  }
}
