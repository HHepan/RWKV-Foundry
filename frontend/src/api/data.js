/**
 * 数据管理 API —— 数据集、基底模型、checkpoint、角色卡。
 *
 * 每个函数对应一个后端端点，参数与返回类型集中在此。
 * 页面组件不应出现裸 fetch/axios。
 */

import { http } from './client'

// ── 数据集 ───────────────────────────────────────────────────────────────
export const dataApi = {
  /** @returns {Promise<string[]>} 形如 ["./", "./out"] */
  listFolders: () => http.get('/api/data/folders'),

  /** @returns {Promise<string[]>} 该目录下的 .jsonl 文件名 */
  listFiles: (folder = '') => http.get('/api/data/files', { params: { folder } }),

  /**
   * @param {object} p
   * @param {string} p.filename
   * @param {string} [p.folder]
   * @param {number} [p.page]
   * @param {number} [p.size]
   * @param {'all'|'single'|'multi'} [p.roundsFilter]
   */
  page: ({ filename, folder = '', page = 1, size = 10, roundsFilter = 'all' }) =>
    http.get(`/api/data/files/${encodeURIComponent(filename)}`, {
      params: { folder, page, size, rounds_filter: roundsFilter },
    }),

  stats: (filename, folder = '') =>
    http.get('/api/data/files/stats', { params: { filename, folder } }),

  updateRow: (filename, rowId, text, folder = '') =>
    http.put(`/api/data/files/${encodeURIComponent(filename)}/${rowId}`, {
      params: { folder },
      body: { text },
    }),

  deleteRow: (filename, rowId, folder = '') =>
    http.del(`/api/data/files/${encodeURIComponent(filename)}/${rowId}`, {
      params: { folder },
    }),

  deleteFile: (filename, folder = '') =>
    http.del(`/api/data/files/${encodeURIComponent(filename)}`, { params: { folder } }),

  /**
   * @param {object} p
   * @param {{filename:string, folder?:string}[]} p.sourceFiles
   * @param {string} p.newName
   * @param {string} [p.folder]
   * @param {boolean} [p.shuffle]
   * @param {Record<string, number>} [p.counts] 按源文件限制采样数
   */
  merge: ({ sourceFiles, newName, folder = '', shuffle = true, counts = {} }) =>
    http.post('/api/data/merge', {
      body: {
        source_files: sourceFiles,
        new_name: newName,
        folder,
        shuffle,
        counts,
      },
    }),

  // ── 基底模型 ───────────────────────────────────────────────────────────
  listBaseModels: () => http.get('/api/data/base-models'),

  // ── checkpoint ─────────────────────────────────────────────────────────
  listCheckpointFolders: () => http.get('/api/data/checkpoint-folders'),

  listCheckpointFiles: (folder) =>
    http.get('/api/data/checkpoint-files', { params: { folder } }),

  // ── 角色卡 ─────────────────────────────────────────────────────────────
  listRoles: () => http.get('/api/data/roles'),

  saveRole: ({ id, name, content }) =>
    http.post('/api/data/roles', { body: { id, name, content } }),

  deleteRole: (id) => http.del(`/api/data/roles/${encodeURIComponent(id)}`),

  // ── 对话数据 ───────────────────────────────────────────────────────────
  listChatModels: () => http.get('/api/data/chat-models'),

  getChatData: (model, session = '') =>
    http.get('/api/data/chat-data', { params: { model, session } }),

  deleteChatData: (model, session = '') =>
    http.del('/api/data/chat-data', { params: { model, session } }),
}
