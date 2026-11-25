<script setup>
/**
 * 数据管理页。
 *
 * ## 分页策略（性能关键）
 *
 * 后端 `read_page` 支持 `total_override`：给了精确 total 就只读本页，
 * 不给则返回 total=null 但仍只读本页。因此本页流程是：
 *
 *   1. 选中文件 → 先调 `/files/stats`（有 mtime 缓存，快）拿精确行数
 *   2. 用该行数作为 total_override 去分页 → 每次翻页只读一页
 *
 * 若跳过第 1 步直接分页，total 会是 null，页码算不出来。
 * 这一步不是多余——旧版正是为算 total 而把整个文件扫一遍（9.9MB 要 308ms）。
 */

import { computed, onMounted, ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { dataApi } from '@/api/data'

const folders = ref([])
const files = ref([])
const currentFolder = ref('./')
const currentFile = ref('')

const stats = ref(null)
const rows = ref([])
const page = ref(1)
const size = ref(10)
const roundsFilter = ref('all')
const loading = ref(false)
const total = ref(0)

const editVisible = ref(false)
const editForm = ref({ id: 0, text: '' })

const mergeVisible = ref(false)
const mergeForm = ref({ newName: '', shuffle: true, picks: [] })

const hasFile = computed(() => !!currentFile.value)
const pageCount = computed(() => Math.max(1, Math.ceil(total.value / size.value)))

async function loadFolders() {
  try {
    folders.value = await dataApi.listFolders()
  } catch (err) {
    ElMessage.error(`加载目录失败: ${err.message}`)
  }
}

async function loadFiles() {
  if (!currentFolder.value) return
  try {
    files.value = await dataApi.listFiles(currentFolder.value)
    if (currentFile.value && !files.value.includes(currentFile.value)) {
      currentFile.value = ''
      rows.value = []
      stats.value = null
      total.value = 0
    }
  } catch (err) {
    ElMessage.error(`加载文件失败: ${err.message}`)
    files.value = []
  }
}

async function loadStats() {
  if (!hasFile.value) return
  try {
    stats.value = await dataApi.stats(currentFile.value, currentFolder.value)
    total.value = stats.value.line_count
  } catch (err) {
    ElMessage.error(`读取统计失败: ${err.message}`)
    stats.value = null
    total.value = 0
  }
}

async function loadPage() {
  if (!hasFile.value) return
  loading.value = true
  try {
    const res = await dataApi.page({
      filename: currentFile.value,
      folder: currentFolder.value,
      page: page.value,
      size: size.value,
      roundsFilter: roundsFilter.value,
    })
    rows.value = res.data || []
    // 带过滤时后端会返回精确 total；不带过滤时用 stats 的行数
    if (res.total !== null && res.total !== undefined) total.value = res.total
  } catch (err) {
    ElMessage.error(`加载数据失败: ${err.message}`)
    rows.value = []
  } finally {
    loading.value = false
  }
}

async function selectFile(name) {
  currentFile.value = name
  page.value = 1
  await loadStats()
  await loadPage()
}

async function onFolderChange() {
  currentFile.value = ''
  rows.value = []
  stats.value = null
  total.value = 0
  await loadFiles()
}

watch([page, size, roundsFilter], loadPage)

onMounted(async () => {
  await loadFolders()
  await loadFiles()

  // 自动选中第一个有文件的目录，避免用户看到空列表以为没数据。
  // 旧版需要手动展开目录才发现文件，是常见困惑点。
  if (!files.value.length) {
    for (const f of folders.value) {
      if (f === currentFolder.value) continue
      try {
        const list = await dataApi.listFiles(f)
        if (list.length) {
          currentFolder.value = f
          await loadFiles()
          break
        }
      } catch {
        /* 跳过不可读目录 */
      }
    }
  }

  // 仍然只选第一个文件，但不自动打开内容——避免加载未预期的大文件
  if (files.value.length) {
    await selectFile(files.value[0])
  }
})

function openEdit(row) {
  editForm.value = { id: row.id, text: row.text }
  editVisible.value = true
}

async function submitEdit() {
  try {
    await dataApi.updateRow(editForm.value.id, editForm.value.id, editForm.value.text, currentFolder.value)
    ElMessage.success('已保存')
    editVisible.value = false
    await loadPage()
  } catch (err) {
    ElMessage.error(`保存失败: ${err.message}`)
  }
}

async function removeRow(row) {
  try {
    await ElMessageBox.confirm(`确定删除第 ${row.id} 行？此操作不可撤销。`, '删除确认', {
      type: 'warning',
      confirmButtonText: '删除',
    })
    await dataApi.deleteRow(currentFile.value, row.id, currentFolder.value)
    ElMessage.success('已删除')
    await loadStats()
    await loadPage()
  } catch (err) {
    if (err !== 'cancel') ElMessage.error(`删除失败: ${err.message}`)
  }
}

async function removeFile() {
  if (!hasFile.value) return
  try {
    await ElMessageBox.confirm(`确定删除数据集 ${currentFile.value}？`, '删除文件', {
      type: 'warning',
      confirmButtonText: '删除',
    })
    await dataApi.deleteFile(currentFile.value, currentFolder.value)
    ElMessage.success('文件已删除')
    currentFile.value = ''
    stats.value = null
    rows.value = []
    total.value = 0
    await loadFiles()
  } catch (err) {
    if (err !== 'cancel') ElMessage.error(`删除失败: ${err.message}`)
  }
}

function openMerge() {
  mergeForm.value = {
    newName: '',
    shuffle: true,
    picks: files.value.map((f) => ({ filename: f, folder: currentFolder.value, count: 0, checked: false })),
  }
  mergeVisible.value = true
}

async function submitMerge() {
  const picked = mergeForm.value.picks.filter((p) => p.checked)
  if (!picked.length) {
    ElMessage.warning('请至少选择一个源文件')
    return
  }
  if (!mergeForm.value.newName.trim()) {
    ElMessage.warning('请填写输出文件名')
    return
  }
  const counts = {}
  for (const p of picked) {
    if (p.count > 0) counts[`${p.folder}/${p.filename}`] = p.count
  }
  try {
    const res = await dataApi.merge({
      sourceFiles: picked.map((p) => ({ filename: p.filename, folder: p.folder })),
      newName: mergeForm.value.newName.trim(),
      folder: currentFolder.value === './' ? '' : currentFolder.value.replace(/^\.\//, ''),
      shuffle: mergeForm.value.shuffle,
      counts,
    })
    ElMessage.success(`已合并 ${res.total_lines} 行 → ${res.filename}`)
    mergeVisible.value = false
    await loadFiles()
  } catch (err) {
    ElMessage.error(`合并失败: ${err.message}`)
  }
}

function roleTag(role) {
  return role === 'user' ? 'primary' : 'success'
}
</script>

<template>
  <div class="data-layout">
    <!-- 左：目录与文件 -->
    <aside class="panel side">
      <div class="panel-title">
        <span>数据集</span>
        <el-button size="small" text @click="loadFolders">刷新</el-button>
      </div>

      <div class="panel-body side-body">
        <el-select v-model="currentFolder" size="small" class="full" @change="onFolderChange">
          <el-option v-for="f in folders" :key="f" :label="f" :value="f" />
        </el-select>

        <div class="file-list">
          <button
            v-for="f in files"
            :key="f"
            class="file-item"
            :class="{ active: f === currentFile }"
            @click="selectFile(f)"
          >
            {{ f }}
          </button>
          <p v-if="!files.length" class="empty-hint">该目录下没有 .jsonl 文件</p>
        </div>

        <el-button size="small" class="full" :disabled="files.length < 1" @click="openMerge">
          合并数据集
        </el-button>
      </div>
    </aside>

    <!-- 右：统计与内容 -->
    <section class="main-col">
      <div class="panel stats-bar">
        <div class="stat">
          <span class="stat-label">当前文件</span>
          <strong>{{ currentFile || '未选择' }}</strong>
        </div>
        <div class="stat">
          <span class="stat-label">总行数</span>
          <strong>{{ stats ? stats.line_count : '—' }}</strong>
        </div>
        <div class="stat">
          <span class="stat-label">单轮 / 多轮</span>
          <strong>{{ stats ? `${stats.rounds_single} / ${stats.rounds_multi}` : '—' }}</strong>
        </div>
        <div class="stat">
          <span class="stat-label">索引</span>
          <el-tag v-if="stats" :type="stats.cached ? 'success' : 'info'" size="small">
            {{ stats.cached ? '命中缓存' : '刚扫描' }}
          </el-tag>
          <span v-else>—</span>
        </div>
        <div class="stats-actions">
          <el-radio-group v-model="roundsFilter" size="small">
            <el-radio-button value="all">全部</el-radio-button>
            <el-radio-button value="single">单轮</el-radio-button>
            <el-radio-button value="multi">多轮</el-radio-button>
          </el-radio-group>
          <el-button size="small" type="danger" :disabled="!hasFile" @click="removeFile">
            删除文件
          </el-button>
        </div>
      </div>

      <div v-loading="loading" class="panel content">
        <p v-if="!hasFile" class="empty-hint">从左侧选择一个数据集文件</p>
        <p v-else-if="!rows.length" class="empty-hint">没有匹配的数据</p>

        <div v-for="row in rows" :key="row.id" class="row-card">
          <div class="row-head">
            <span class="row-id">#{{ row.id }}</span>
            <el-tag size="small" type="info">{{ row.rounds }} 轮</el-tag>
            <span class="spacer" />
            <el-button size="small" text @click="openEdit(row)">编辑</el-button>
            <el-button size="small" text type="danger" @click="removeRow(row)">删除</el-button>
          </div>

          <div v-if="row.conversations.length" class="conv">
            <div v-for="(c, i) in row.conversations" :key="i" class="conv-line">
              <el-tag :type="roleTag(c.role)" size="small" effect="plain" class="role-tag">
                {{ c.role }}
              </el-tag>
              <span class="conv-text">{{ c.content }}</span>
            </div>
          </div>
          <pre v-else class="mono raw">{{ row.text }}</pre>
        </div>

        <div v-if="hasFile && total > 0" class="pager">
          <el-pagination
            v-model:current-page="page"
            v-model:page-size="size"
            :total="total"
            :page-sizes="[10, 20, 50, 100]"
            layout="total, sizes, prev, pager, next"
            background
            small
          />
        </div>
      </div>
    </section>

    <!-- 编辑对话框 -->
    <el-dialog v-model="editVisible" title="编辑数据行" width="720px">
      <el-input v-model="editForm.text" type="textarea" :rows="14" class="mono" />
      <template #footer>
        <el-button @click="editVisible = false">取消</el-button>
        <el-button type="primary" @click="submitEdit">保存</el-button>
      </template>
    </el-dialog>

    <!-- 合并对话框 -->
    <el-dialog v-model="mergeVisible" title="合并数据集" width="640px">
      <el-form label-width="90px">
        <el-form-item label="输出文件名">
          <el-input v-model="mergeForm.newName" placeholder="例如 merged" />
        </el-form-item>
        <el-form-item label="打乱顺序">
          <el-switch v-model="mergeForm.shuffle" />
        </el-form-item>
        <el-form-item label="源文件">
          <div class="merge-list">
            <div v-for="p in mergeForm.picks" :key="p.filename" class="merge-item">
              <el-checkbox v-model="p.checked">{{ p.filename }}</el-checkbox>
              <el-input-number
                v-model="p.count"
                :min="0"
                size="small"
                controls-position="right"
                placeholder="0=全取"
                class="count-input"
              />
            </div>
          </div>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="mergeVisible = false">取消</el-button>
        <el-button type="primary" @click="submitMerge">合并</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.data-layout {
  display: grid;
  grid-template-columns: 232px 1fr;
  gap: 12px;
  height: 100%;
}

/* ═══════════════════════════════════════════════════════════════════
   移动端适配
   ═══════════════════════════════════════════════════════════════════ */

/* 平板：左侧目录栏收窄 */
@media (max-width: 1099px) {
  .data-layout {
    grid-template-columns: 190px 1fr;
    gap: 8px;
  }
}

/* 手机：单栏，目录行变成顶部横条（文件列表横向滚动） */
@media (max-width: 767px) {
  .data-layout {
    grid-template-columns: 1fr;
    gap: 8px;
  }

  /* 左侧目录栏改为横向排列的紧凑条 */
  .side {
    max-height: 168px;
  }

  .side-body {
    flex-direction: row;
    align-items: center;
    gap: 8px;
    min-height: 0;
    overflow-x: auto;
  }

  /* 下拉与按钮在手机横条里不占满宽 */
  .side-body :deep(.el-select) {
    width: 110px;
    flex: 0 0 auto;
  }

  .side-body > .full {
    width: auto;
    flex: 0 0 auto;
  }

  /* 文件列表变成横向滚动的 chip 流 */
  .file-list {
    flex-direction: row;
    overflow-x: auto;
    overflow-y: hidden;
    min-height: 0;
    gap: 6px;
  }

  .file-item {
    flex: 0 0 auto;
    white-space: nowrap;
    border: 1px solid var(--foundry-border);
  }

  /* 统计栏在手机上换行排列 */
  .stats-bar {
    gap: 14px;
    padding: 10px 12px;
  }

  .stats-actions {
    margin-left: 0;
    width: 100%;
    flex-wrap: wrap;
  }

  /* 消息行正文缩小边距 */
  .conv-text {
    font-size: 12.5px;
  }

  .row-card {
    padding: 8px 10px;
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
  gap: 10px;
  min-height: 0;
  flex: 1;
}

.full {
  width: 100%;
}

.file-list {
  flex: 1;
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-height: 120px;
}

.file-item {
  text-align: left;
  padding: 7px 10px;
  border: none;
  background: transparent;
  border-radius: 6px;
  font-size: 13px;
  color: #3a4152;
  cursor: pointer;
  font-family: var(--foundry-mono);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-item:hover {
  background: #f2f4f7;
}

.file-item.active {
  background: #fdeee8;
  color: #b8432a;
  font-weight: 600;
}

.main-col {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
}

.stats-bar {
  display: flex;
  align-items: center;
  gap: 24px;
  padding: 12px 16px;
  flex-wrap: wrap;
}

.stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.stat-label {
  font-size: 11px;
  color: var(--foundry-text-dim);
}

.stat strong {
  font-size: 14px;
}

.stats-actions {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 10px;
}

.content {
  flex: 1;
  overflow: auto;
  padding: 12px;
  min-height: 0;
}

.row-card {
  border: 1px solid var(--foundry-border);
  border-radius: 8px;
  padding: 10px 12px;
  margin-bottom: 8px;
}

.row-head {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.row-id {
  font-family: var(--foundry-mono);
  font-size: 12px;
  color: var(--foundry-text-dim);
}

.spacer {
  flex: 1;
}

.conv {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.conv-line {
  display: flex;
  gap: 8px;
  align-items: flex-start;
}

.role-tag {
  flex: 0 0 auto;
  margin-top: 1px;
}

.conv-text {
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

.raw {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  color: #3a4152;
}

.pager {
  display: flex;
  justify-content: center;
  padding: 12px 0 4px;
}

.merge-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 260px;
  overflow: auto;
  width: 100%;
}

.merge-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.count-input {
  width: 120px;
}
</style>
