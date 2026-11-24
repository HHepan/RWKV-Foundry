import { createRouter, createWebHistory } from 'vue-router'

/**
 * 只有三个页面。
 *
 * 旧版还有一整套 mobile/ 目录（2335 行，与桌面端功能完全重复，各写 16 处 axios）。
 * 已按决策砍掉——训练参数有 20 多个 el- 组件，手机上调 LoRA 的 r 和 alpha
 * 是伪需求。要移动访问就用响应式 CSS，不是复制一遍代码。
 */
const routes = [
  { path: '/', redirect: '/data' },
  {
    path: '/data',
    name: 'DataManage',
    component: () => import('@/views/DataManage.vue'),
    meta: { title: '数据管理' },
  },
  {
    path: '/train',
    name: 'Train',
    component: () => import('@/views/Train.vue'),
    meta: { title: '微调训练' },
  },
  {
    path: '/chat',
    name: 'Chat',
    component: () => import('@/views/Chat.vue'),
    meta: { title: '对话测试' },
  },
  { path: '/:pathMatch(.*)*', redirect: '/data' },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
