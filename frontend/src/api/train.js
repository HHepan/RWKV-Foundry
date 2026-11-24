/**
 * 训练 API —— 配方校验、脚本预览、启停、状态、loss、日志。
 */

import { http } from './client'

export const trainApi = {
  /**
   * 校验配方并拿到内容指纹（hash）。
   * 同一份参数永远同一个 hash —— 这是横向对比实验的前提。
   */
  validateRecipe: (recipe) =>
    http.post('/api/train/recipe/validate', { body: recipe }),

  /** 预览将要执行的训练脚本（不落盘、不启动） */
  previewScript: (recipe, saveFolder) =>
    http.post('/api/train/recipe/preview', {
      params: { save_folder: saveFolder },
      body: recipe,
    }),

  start: (recipe, saveFolder) =>
    http.post('/api/train/start', {
      params: { save_folder: saveFolder },
      body: recipe,
    }),

  stop: () => http.post('/api/train/stop'),

  status: (folder) => http.get('/api/train/status', { params: { folder } }),

  /** @param {number} [sinceStep] 只取该步之后的点，用于增量刷新 */
  loss: (folder, sinceStep = 0) =>
    http.get('/api/train/loss', { params: { folder, since_step: sinceStep } }),

  /** @param {number} [sinceSeq] 只取该序号之后的行 —— 根治旧版每次拉全量 */
  logs: (folder, sinceSeq = 0, limit = 1000) =>
    http.get('/api/train/logs', { params: { folder, since_seq: sinceSeq, limit } }),

  records: () => http.get('/api/train/records'),

  recordDetail: (folder) => http.get(`/api/train/records/${encodeURIComponent(folder)}`),

  deleteRecord: (folder) => http.del(`/api/train/records/${encodeURIComponent(folder)}`),

  folders: () => http.get('/api/train/folders'),
}

/** 模型尺寸 → (n_layer, n_embd)。与后端 MODEL_SIZE_MAP 必须一致。 */
export const MODEL_SIZES = ['0.1B', '0.4B', '1.5B', '2.9B', '7B', '13.3B']

/** 训练默认配方。字段与后端 RecipePayload 一一对应。 */
export function defaultRecipe() {
  return {
    base_model: '',
    dataset: '',
    model_size: '2.9B',
    ctx_len: 512,
    micro_bsz: 1,
    epoch_steps: 1000,
    epoch_count: 1,
    epoch_save: 1,
    lr_init: 2e-5,
    lr_final: 2e-5,
    lora_r: 32,
    lora_alpha: 32,
    lora_dropout: 0.01,
    precision: 'bf16',
    my_testing: 'x070',
  }
}
