/**
 * 前端 API 契约类型。
 *
 * ⚠️ 本文件由 scripts/gen-types.mjs 从后端 OpenAPI 生成，请勿手工编辑。
 * 生成时间: 2026-11-24T15:43:52.552Z
 * 后端来源: http://127.0.0.1:8000
 */

export const API_VERSION = "0.1.0"

/**
 * @typedef {Object} ChatRequest
 * 对话请求。

``history`` **不含**本轮用户消息——由 ``message`` 追加。
这与前端 regenerate 的语义一致：先移除上轮 assistant 回复，再重发。
 * @property {string} [model_ref]
 * @property {string} [message]
 * @property {MessagePayload[]} [history]
 * @property {Object<string, *>} [params]
 * @property {boolean} [thinking_mode]
 * @property {string} [session_name]
 * @property {string} [role_id]
 */

/**
 * @typedef {Object} ConfigResponse
 * @property {string} [version]
 * @property {Object<string, string>} [paths]
 * @property {Object<string, *>} [infer_defaults]
 * @property {Object<string, *>} [train_defaults]
 * @property {number} [vocab_size]
 * @property {string[]} [model_sizes]
 */

/**
 * @typedef {Object} DatasetPageResponse
 * @property {number} [total] 总行数；无索引时为 null
 * @property {number} [page]
 * @property {number} [size]
 * @property {boolean} [has_more]
 * @property {Object<string, *>[]} [data]
 */

/**
 * @typedef {Object} DatasetRowUpdate
 * @property {string} [text]
 */

/**
 * @typedef {Object} DatasetStatsResponse
 * @property {string} [filename]
 * @property {string} [folder]
 * @property {number} [line_count]
 * @property {number} [rounds_single]
 * @property {number} [rounds_multi]
 * @property {boolean} [cached] 是否命中索引缓存
 */

/**
 * @typedef {Object} HTTPValidationError
 * @property {ValidationError[]} [detail]
 */

/**
 * @typedef {Object} HealthResponse
 * @property {boolean} [ok]
 * @property {boolean} [model_loaded]
 * @property {string} [model_ref]
 * @property {string} [active_train]
 * @property {number} [schema_version]
 */

/**
 * @typedef {Object} LogLine
 * @property {number} [seq]
 * @property {string} [line]
 */

/**
 * @typedef {Object} LossPoint
 * @property {number} [step]
 * @property {number} [epoch]
 * @property {number} [loss]
 * @property {number} [lr]
 * @property {number} [kt_s]
 */

/**
 * @typedef {Object} MergeRequest
 * @property {Object<string, string>[]} [source_files] 例如 [{"filename": "a.jsonl", "folder": "out"}]
 * @property {string} [new_name]
 * @property {string} [folder]
 * @property {boolean} [shuffle]
 * @property {Object<string, number>} [counts] 按源文件限制采样数，键为 "folder/filename" 或 "filename"；空表示全取
 */

/**
 * @typedef {Object} MessagePayload
 * @property {string} [role]
 * @property {string} [content]
 * @property {string} [thinking]
 */

/**
 * @typedef {Object} RecipePayload
 * 配方请求体。所有字段可选，未传则用默认值。
 * @property {string} [base_model]
 * @property {string} [dataset]
 * @property {string} [model_size]
 * @property {number} [ctx_len]
 * @property {number} [micro_bsz]
 * @property {number} [epoch_steps]
 * @property {number} [epoch_count]
 * @property {number} [epoch_save]
 * @property {number} [lr_init]
 * @property {number} [lr_final]
 * @property {number} [lora_r]
 * @property {number} [lora_alpha]
 * @property {number} [lora_dropout]
 * @property {string} [precision]
 * @property {string} [my_testing]
 */

/**
 * @typedef {Object} RecipeValidationResponse
 * @property {string} [hash]
 * @property {boolean} [valid]
 * @property {string[]} [problems]
 * @property {number} [n_layer]
 * @property {number} [n_embd]
 * @property {Object<string, *>} [peft_config]
 */

/**
 * @typedef {Object} RolePayload
 * @property {string} [id]
 * @property {string} [name]
 * @property {string} [content]
 */

/**
 * @typedef {Object} RoleResponse
 * @property {string} [id]
 * @property {string} [name]
 * @property {string} [content]
 * @property {string} [updated_at]
 */

/**
 * @typedef {Object} RunRecord
 * @property {number} [id]
 * @property {string} [folder]
 * @property {string} [base_model]
 * @property {string} [dataset]
 * @property {string} [status]
 * @property {string} [started_at]
 * @property {string} [finished_at]
 * @property {number} [exit_code]
 * @property {string} [error]
 * @property {string} [recipe_hash]
 */

/**
 * @typedef {Object} ScriptPreviewResponse
 * @property {string} [hash]
 * @property {string} [script]
 */

/**
 * @typedef {Object} SessionCreateRequest
 * @property {string} [model_ref] 例如 "20260414/rwkv-10" 或 "base_models/rwkv7-g1a-0.1b"
 * @property {string} [name]
 * @property {string} [role_id]
 * @property {Object<string, *>} [params]
 */

/**
 * @typedef {Object} SessionResponse
 * @property {number} [id]
 * @property {string} [model_ref]
 * @property {string} [name]
 * @property {string} [role_id]
 * @property {Object<string, *>} [params]
 * @property {string} [created_at]
 * @property {string} [updated_at]
 * @property {number} [message_count]
 */

/**
 * @typedef {Object} TrainStartResponse
 * @property {string} [folder]
 * @property {number} [run_id]
 * @property {string} [recipe_hash]
 * @property {string} [status]
 * @property {number} [pid]
 * @property {number} [pgid]
 */

/**
 * @typedef {Object} TrainStatusResponse
 * @property {string} [folder]
 * @property {string} [status]
 * @property {number} [pid]
 * @property {number} [exit_code]
 * @property {string} [error]
 * @property {number} [elapsed_seconds]
 * @property {string} [recipe_hash]
 */

/**
 * @typedef {Object} TrainStopResponse
 * @property {string} [status]
 * @property {string} [note]
 */

/**
 * @typedef {Object} ValidationError
 * @property {string|number[]} [loc]
 * @property {string} [msg]
 * @property {string} [type]
 * @property {*} [input]
 * @property {Object<string, *>} [ctx]
 */

/**
 * 后端路由清单（由 OpenAPI 生成，勿手改）
 *
 *   GET    /api/data/folders — Get Folders
 *          -> string[]
 *   GET    /api/data/files — Get Files
 *          -> string[]
 *   GET    /api/data/files/stats — Get File Stats
 *          -> DatasetStatsResponse
 *   GET    /api/data/files/{filename} — Get File Data
 *          -> DatasetPageResponse
 *   DELETE /api/data/files/{filename} — Delete File
 *          -> Object<string, *>
 *   PUT    /api/data/files/{filename}/{row_id} — Update Row
 *          -> Object<string, *>
 *   DELETE /api/data/files/{filename}/{row_id} — Delete Row
 *          -> Object<string, *>
 *   POST   /api/data/merge — Merge Files
 *          -> Object<string, *>
 *   GET    /api/data/base-models — Get Base Models
 *          -> string[]
 *   GET    /api/data/base-model-files — Get Base Model Files
 *          -> string[]
 *   GET    /api/data/checkpoint-folders — Get Checkpoint Folders
 *          -> string[]
 *   GET    /api/data/checkpoint-files — Get Checkpoint Files
 *          -> string[]
 *   GET    /api/data/roles — Get Roles
 *          -> RoleResponse[]
 *   POST   /api/data/roles — Save Role
 *          -> Object<string, *>
 *   DELETE /api/data/roles/{role_id} — Delete Role
 *          -> Object<string, *>
 *   GET    /api/data/chat-models — Get Chat Models
 *          -> Object<string, *>[]
 *   GET    /api/data/chat-data — Get Chat Data
 *          -> Object<string, *>
 *   DELETE /api/data/chat-data — Delete Chat Data
 *          -> Object<string, *>
 *   POST   /api/train/recipe/validate — Validate Recipe
 *          -> RecipeValidationResponse
 *   POST   /api/train/recipe/preview — Preview Script
 *          -> ScriptPreviewResponse
 *   POST   /api/train/start — Start Training
 *          -> TrainStartResponse
 *   POST   /api/train/stop — Stop Training
 *          -> TrainStopResponse
 *   GET    /api/train/status — Get Status
 *          -> TrainStatusResponse
 *   GET    /api/train/loss — Get Loss
 *          -> LossPoint[]
 *   GET    /api/train/loss-file/{folder_name} — Get Loss File
 *          -> LossPoint[]
 *   GET    /api/train/logs — Get Logs
 *          -> LogLine[]
 *   GET    /api/train/records — Get Records
 *          -> RunRecord[]
 *   GET    /api/train/records/{folder_name} — Get Record Detail
 *          -> Object<string, *>
 *   DELETE /api/train/records/{folder_name} — Delete Record
 *          -> Object<string, *>
 *   GET    /api/train/folders — Get Existing Folders
 *          -> string[]
 *   POST   /api/chat/chat — Chat
 *          -> *
 *   POST   /api/chat/regenerate — Regenerate
 *          -> *
 *   POST   /api/chat/preload-model — Preload Model
 *          -> Object<string, *>
 *   POST   /api/chat/reset-state — Reset State
 *          -> Object<string, *>
 *   GET    /api/chat/sessions — List Sessions
 *          -> SessionResponse[]
 *   POST   /api/chat/sessions — Create Session
 *          -> Object<string, *>
 *   DELETE /api/chat/sessions/{session_id} — Delete Session
 *          -> Object<string, *>
 *   GET    /api/health — Health
 *          -> HealthResponse
 *   GET    /api/config — Get Config
 *          -> ConfigResponse
 *   GET    / — Root
 *          -> Object<string, *>
 */

export const ROUTES = Object.freeze({})
