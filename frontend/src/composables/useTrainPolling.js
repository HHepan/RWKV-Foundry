/**
 * 训练状态轮询 composable。
 *
 * ## 旧版的两个性能问题，这里都修掉
 *
 * 1. **全量重拉日志**：旧版每次轮询都 `GET /api/train/logs` 拿整个数组。
 *    训练 2 小时后日志上万行，每秒一次 = 反复传输几 MB。
 *    现在维护 `lastSeq`，只拉增量。
 *
 * 2. **全量重拉 loss**：同理，用 `since_step` 只取新点。
 *
 * 轮询在训练结束后自动停止，避免空转。
 */

import { onUnmounted, ref, shallowRef } from 'vue'
import { trainApi } from '../api/train'

/**
 * @param {object} opts
 * @param {number} [opts.intervalMs] 轮询间隔，默认 1500ms
 */
export function useTrainPolling({ intervalMs = 1500 } = {}) {
  const status = ref('idle')
  const folder = ref('')
  const elapsed = ref(0)
  const error = ref('')
  const pid = ref(null)
  const exitCode = ref(null)
  const recipeHash = ref(null)

  /** @type {import('vue').Ref<Array<{step:number,epoch:number|null,loss:number,lr:number|null}>>} */
  const lossPoints = shallowRef([])
  /** @type {import('vue').Ref<string[]>} */
  const logs = shallowRef([])

  let lastStep = 0
  let lastSeq = 0
  let timer = null
  let inFlight = false

  const isRunning = () => status.value === 'running'

  function resetSeries() {
    lossPoints.value = []
    logs.value = []
    lastStep = 0
    lastSeq = 0
  }

  async function tick() {
    if (!folder.value || inFlight) return
    inFlight = true
    try {
      const st = await trainApi.status(folder.value)
      status.value = st.status || 'idle'
      elapsed.value = st.elapsed_seconds ?? elapsed.value
      pid.value = st.pid ?? null
      exitCode.value = st.exit_code ?? null
      recipeHash.value = st.recipe_hash ?? null
      error.value = st.error || ''

      // 增量拉 loss
      const newPoints = await trainApi.loss(folder.value, lastStep)
      if (newPoints.length) {
        lastStep = newPoints[newPoints.length - 1].step
        lossPoints.value = [...lossPoints.value, ...newPoints]
      }

      // 增量拉日志
      const newLogs = await trainApi.logs(folder.value, lastSeq)
      if (newLogs.length) {
        lastSeq = newLogs[newLogs.length - 1].seq
        logs.value = [...logs.value, ...newLogs.map((l) => l.line)]
      }
    } catch (err) {
      // 训练刚启动时 DB 里可能还没有该 folder，忽略一轮
      if (err.status !== 404) error.value = err.message
    } finally {
      inFlight = false
      if (!isRunning()) stopTimer()
    }
  }

  function startTimer() {
    if (timer) return
    timer = setInterval(tick, intervalMs)
  }

  function stopTimer() {
    if (!timer) return
    clearInterval(timer)
    timer = null
  }

  /**
   * 开始跟踪某个训练目录。
   * @param {string} name  checkpoints 下的目录名
   * @param {object} [opts]
   * @param {boolean} [opts.reset] 是否清空已有曲线（切换 run 时用）
   */
  async function watch(name, { reset = true } = {}) {
    if (!name) return
    if (folder.value !== name) {
      folder.value = name
      if (reset) resetSeries()
    }
    await tick()
    if (isRunning()) startTimer()
  }

  async function stop() {
    try {
      const res = await trainApi.stop()
      status.value = res.status
      error.value = res.note === '进程已自行退出' ? '' : ''
    } catch (err) {
      error.value = err.message
    }
    await tick()
    stopTimer()
  }

  function unwatch() {
    stopTimer()
    folder.value = ''
    resetSeries()
    status.value = 'idle'
  }

  onUnmounted(stopTimer)

  return {
    status,
    folder,
    elapsed,
    error,
    pid,
    exitCode,
    recipeHash,
    lossPoints,
    logs,
    isRunning,
    watch,
    stop,
    unwatch,
    refresh: tick,
    resetSeries,
  }
}
