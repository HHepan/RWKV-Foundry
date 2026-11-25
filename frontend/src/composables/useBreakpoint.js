/**
 * 响应式断点检测。
 *
 * ## 为什么不用纯 CSS 媒体查询
 *
 * 纯 CSS 只能改样式，改不了**结构**。移动端需要：
 *   - 侧栏从「常驻栏」变成「抽屉」（DOM 位置不变，但交互完全不同）
 *   - 推理参数面板从「右侧常驻」变成「弹窗」
 *   - 对话/续写模式切换的按钮位置重排
 *
 * 这些都需要 JS 参与判断。因此用 CSS 做样式、JS 做结构决策。
 *
 * ## 为什么不复制一套 mobile 组件
 *
 * 旧项目 llm-finetune-webui 的 mobile/ 目录有 2335 行，与桌面端功能
 * 完全重复，各写 16 处 axios 调用，行为必然逐渐漂移。
 *
 * 本次采用响应式：**同一套组件，不同断点下换布局**。
 * 只有不可调和的交互差异（如抽屉 vs 常驻栏）才用 v-if 切换 DOM。
 */

import { computed, onUnmounted, ref } from 'vue'

/** 断点阈值（px）。与 CSS 里的 @media 保持一致。 */
export const BREAKPOINTS = {
  /** 小于此值 = 手机竖屏 */
  mobile: 768,
  /** 小于此值 = 平板 / 窄屏桌面 */
  tablet: 1100,
}

/**
 * @param {number} [debounceMs] resize 防抖间隔
 */
export function useBreakpoint(debounceMs = 120) {
  const width = ref(typeof window !== 'undefined' ? window.innerWidth : 1280)
  let timer = null

  function onResize() {
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => {
      width.value = window.innerWidth
    }, debounceMs)
  }

  if (typeof window !== 'undefined') {
    window.addEventListener('resize', onResize, { passive: true })
    onUnmounted(() => {
      if (timer) clearTimeout(timer)
      window.removeEventListener('resize', onResize)
    })
  }

  const isMobile = computed(() => width.value < BREAKPOINTS.mobile)
  const isTablet = computed(
    () => width.value >= BREAKPOINTS.mobile && width.value < BREAKPOINTS.tablet,
  )
  const isDesktop = computed(() => width.value >= BREAKPOINTS.tablet)

  /** 侧栏是否需要折叠成抽屉（手机/平板都折叠，桌面常驻） */
  const needDrawer = computed(() => width.value < BREAKPOINTS.tablet)

  return { width, isMobile, isTablet, isDesktop, needDrawer }
}
