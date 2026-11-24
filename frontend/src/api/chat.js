/**
 * 对话 API —— 流式生成、模型加载、会话管理。
 */

import { http, streamSSE } from './client'

/** 推理参数默认值。与后端 InferParams 字段一致。 */
export function defaultInferParams() {
  return {
    max_tokens: 2048,
    temperature: 1.0,
    top_p: 0.3,
    top_k: 0,
    alpha_frequency: 1.0,
    alpha_presence: 0.0,
    alpha_decay: 0.996,
    newline_adj: 0,
    global_penalty: false,
    presystem: true,
    max_rounds: 0,
    clean_rounds: 10,
  }
}

/** 角色扮演模式的推荐参数。 */
export function roleplayParams() {
  return {
    temperature: 1.15,
    top_p: 0.9,
    top_k: 60,
    alpha_frequency: 0.3,
    alpha_presence: 0.15,
    alpha_decay: 0.4,
  }
}

export const chatApi = {
  /** 预加载模型，避免首次对话等待 */
  preloadModel: (modelRef) =>
    http.post('/api/chat/preload-model', { params: { model_ref: modelRef }, timeout: 300000 }),

  resetState: (modelRef, session = '') =>
    http.post('/api/chat/reset-state', { params: { model_ref: modelRef, session } }),

  listSessions: (modelRef) =>
    http.get('/api/chat/sessions', { params: { model_ref: modelRef } }),

  createSession: ({ modelRef, name = '', roleId = null, params = {} }) =>
    http.post('/api/chat/sessions', {
      body: { model_ref: modelRef, name, role_id: roleId, params },
    }),

  deleteSession: (id) => http.del(`/api/chat/sessions/${id}`),

  /**
   * 流式对话。
   *
   * 注意：``history`` **不含**本轮用户消息 —— 由 ``message`` 追加。
   * 这与后端 generate() 的契约一致，也与 regenerate 的语义对应
   * （前端先移除上轮 assistant 回复，再以同一 message 重发）。
   */
  stream: ({ modelRef, message, history, params, thinkingMode, sessionName, roleId }, handlers) =>
    streamSSE(
      '/api/chat/chat',
      {
        model_ref: modelRef,
        message,
        history,
        params,
        thinking_mode: thinkingMode,
        session_name: sessionName,
        role_id: roleId,
      },
      handlers,
    ),

  /** 重生成：后端会先回滚 state，让模型"忘记"上一轮回复 */
  regenerate: ({ modelRef, message, history, params, thinkingMode, sessionName, roleId }, handlers) =>
    streamSSE(
      '/api/chat/regenerate',
      {
        model_ref: modelRef,
        message,
        history,
        params,
        thinking_mode: thinkingMode,
        session_name: sessionName,
        role_id: roleId,
      },
      handlers,
    ),
}

/** 元信息端点 */
export const metaApi = {
  health: () => http.get('/api/health'),
  config: () => http.get('/api/config'),
}
