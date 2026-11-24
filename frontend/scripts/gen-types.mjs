/**
 * 从后端 OpenAPI 生成前端类型契约。
 *
 * ## 为什么要有这个脚本
 *
 * "参数面板是假的" 这类 bug 的根因是**前后端没有契约**：
 * 旧版 Chat.vue 把 temperature / top_p 老老实实发给后端，
 * 但后端 generate() 内部用 self.current_params，把请求体丢掉了。
 * 前端不知道，后端也不报错，参数静默失效。
 *
 * 现在后端 schema 是明确的（Pydantic），用 openapi.json 生成 JSDoc typedef，
 * 让编辑器能提示字段与类型。接口变更时重跑本脚本即可发现不一致。
 *
 * 用法:
 *     npm run gen:types                    # 默认 http://127.0.0.1:8000
 *     FOUNDRY_API=http://other:9000 npm run gen:types
 */

import { writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const API = process.env.FOUNDRY_API || 'http://127.0.0.1:8000'
const OUT = resolve(dirname(fileURLToPath(import.meta.url)), '../src/api/types.js')

const SCALAR = { string: 'string', integer: 'number', number: 'number', boolean: 'boolean' }

/** JSON Schema → JSDoc 类型 */
function toJsType(schema) {
  if (!schema) return '*'
  if (schema.$ref) return schema.$ref.split('/').pop()
  if (schema.anyOf) {
    const parts = schema.anyOf.map(toJsType).filter((t) => t !== '*')
    if (parts.length) return parts.join('|')
  }
  if (schema.type === 'array') return `${toJsType(schema.items)}[]`
  if (schema.type === 'object') {
    if (schema.additionalProperties && !schema.properties) {
      return `Object<string, ${toJsType(schema.additionalProperties)}>`
    }
    return 'Object<string, *>'
  }
  return SCALAR[schema.type] || '*'
}

function emitModel(name, schema) {
  const required = new Set(schema.required || [])
  const lines = [`/**`, ` * @typedef {Object} ${name}`]
  if (schema.description) lines.push(` * ${schema.description}`)
  for (const [field, prop] of Object.entries(schema.properties || {})) {
    const type = toJsType(prop)
    const opt = required.has(field) ? '' : '='
    const desc = prop.description ? ` ${prop.description.replace(/\s+/g, ' ')}` : ''
    lines.push(` * @property {${type}} [${field}]${desc}`)
  }
  lines.push(' */', '')
  return lines.join('\n')
}

function emitRoutes(paths) {
  const rows = []
  for (const [path, methods] of Object.entries(paths)) {
    for (const [method, op] of Object.entries(methods)) {
      if (!['get', 'post', 'put', 'delete', 'patch'].includes(method)) continue
      const ok = op.responses?.['200']?.content?.['application/json']?.schema
      rows.push(
        ` *   ${method.toUpperCase().padEnd(6)} ${path}${op.summary ? ` — ${op.summary}` : ''}` +
          `\n *          -> ${toJsType(ok)}`,
      )
    }
  }
  return `/**\n * 后端路由清单（由 OpenAPI 生成，勿手改）\n *\n${rows.join('\n')}\n */\n\nexport const ROUTES = Object.freeze({})\n`
}

async function main() {
  let doc
  try {
    const res = await fetch(`${API}/openapi.json`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    doc = await res.json()
  } catch (err) {
    console.error(`[gen-types] 无法获取 ${API}/openapi.json: ${err.message}`)
    console.error('[gen-types] 请先启动后端，或保留现有 types.js 手动维护。')
    process.exit(1)
  }

  const schemas = doc.components?.schemas || {}
  const blocks = [
    '/**',
    ' * 前端 API 契约类型。',
    ' *',
    ' * ⚠️ 本文件由 scripts/gen-types.mjs 从后端 OpenAPI 生成，请勿手工编辑。',
    ` * 生成时间: ${new Date().toISOString()}`,
    ` * 后端来源: ${API}`,
    ' */',
    '',
    'export const API_VERSION = ' + JSON.stringify(doc.info?.version || '0'),
    '',
  ]

  for (const [name, schema] of Object.entries(schemas)) {
    if (schema.type === 'object' || schema.properties) blocks.push(emitModel(name, schema))
  }
  blocks.push(emitRoutes(doc.paths || {}))

  await writeFile(OUT, blocks.join('\n'), 'utf8')
  console.log(`[gen-types] 已生成 ${Object.keys(schemas).length} 个类型 → ${OUT}`)
}

main()
