<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { metaApi } from '@/api/chat'
import { useBreakpoint } from '@/composables/useBreakpoint'

const route = useRoute()
const router = useRouter()
const health = ref(null)
const healthError = ref('')

// 响应式：窄屏时侧栏转抽屉
const { isMobile, needDrawer } = useBreakpoint()
const drawerOpen = ref(false)

/** 移动端点导航后自动收起抽屉 */
const go = (path) => {
  drawerOpen.value = false
  router.push(path)
}

const title = computed(() => route.meta.title || 'RWKV-Foundry')

/**
 * 把模型引用压成短标识。
 *
 * 侧栏只有 208px，而模型引用形如：
 *   base_models/rwkv7-g1g-13.3b-20260523-ctx8192
 * 完整显示会撑出横向滚动条。
 *
 * 抽取策略：
 *   1. 去掉 base_models/ 或 checkpoints 目录前缀
 *   2. 从 rwkv7-g1g-13.3b-20260523-ctx8192 里抓 "尺寸" 段（13.3b / 2.9b）
 *   3. 保留前面的版本代号（g1g）便于区分
 *
 * 例：base_models/rwkv7-g1g-13.3b-20260523-ctx8192 → rwkv7-g1g-13.3b
 */
function shortenModelRef(ref) {
  if (!ref) return ''
  const bare = ref.replace(/^[^/]+\//, '').replace(/\.pth$/, '')

  // 匹配 rwkv7-g1g-13.3b-... 中的 m 段
  const m = bare.match(/(rwkv\d*-?[a-z0-9]*)-(\d+\.?\d*[bm])/i)
  if (m) return `${m[1]}-${m[2]}`

  // 回退：取第一二段
  const parts = bare.split('-')
  return parts.length > 2 ? parts.slice(0, 2).join('-') : bare
}

const modelBadge = computed(() => {
  if (!health.value) return { text: '未连接', type: 'info' }
  if (health.value.model_loaded) {
    return { text: shortenModelRef(health.value.model_ref) || '已加载', type: 'success' }
  }
  return { text: '未加载模型', type: 'info' }
})

const trainBadge = computed(() => {
  if (health.value?.active_train) {
    return { text: `训练中 ${health.value.active_train}`, type: 'warning' }
  }
  return null
})

onMounted(async () => {
  try {
    health.value = await metaApi.health()
  } catch (err) {
    healthError.value = err.message
  }
})

const navs = [
  { path: '/data', label: '数据管理', icon: 'Document' },
  { path: '/train', label: '微调训练', icon: 'Cpu' },
  { path: '/chat', label: '对话测试', icon: 'ChatDotRound' },
]
</script>

<template>
  <el-container class="app-shell">
    <!--
      响应式侧栏：
        ≥1100px  常驻侧栏
        <1100px  转抽屉（点击汉堡按钮滑出）

      同一套 nav 内容，只是容器不同 —— 不复制组件。
    -->
    <el-aside v-if="!needDrawer" width="208px" class="app-aside">
      <div class="brand">
        <div class="brand-mark">F</div>
        <div class="brand-text">
          <strong>RWKV-Foundry</strong>
          <small>铸造厂</small>
        </div>
      </div>

      <nav class="nav">
        <router-link
          v-for="n in navs"
          :key="n.path"
          :to="n.path"
          class="nav-item"
          :class="{ active: route.path === n.path }"
        >
          <el-icon><component :is="n.icon" /></el-icon>
          <span>{{ n.label }}</span>
        </router-link>
      </nav>

      <div class="aside-foot">
        <el-tooltip :content="health?.model_ref || '未加载模型'" placement="right" :disabled="!health?.model_ref">
          <el-tag :type="modelBadge.type" size="small" effect="dark" round class="foot-tag">
            {{ modelBadge.text }}
          </el-tag>
        </el-tooltip>
        <el-tag v-if="trainBadge" :type="trainBadge.type" size="small" effect="dark" round class="foot-tag">
          {{ trainBadge.text }}
        </el-tag>
        <p v-if="healthError" class="foot-error">{{ healthError }}</p>
      </div>
    </el-aside>

    <!-- 抽屉版侧栏 -->
    <el-drawer
      v-model="drawerOpen"
      direction="ltr"
      size="240px"
      :with-header="false"
      class="app-drawer"
    >
      <div class="drawer-inner">
        <div class="brand">
          <div class="brand-mark">F</div>
          <div class="brand-text">
            <strong>RWKV-Foundry</strong>
            <small>铸造厂</small>
          </div>
        </div>

        <nav class="nav">
          <button
            v-for="n in navs"
            :key="n.path"
            type="button"
            class="nav-item drawer-item"
            :class="{ active: route.path === n.path }"
            @click="go(n.path)"
          >
            <el-icon><component :is="n.icon" /></el-icon>
            <span>{{ n.label }}</span>
          </button>
        </nav>

        <div class="aside-foot">
          <el-tag :type="modelBadge.type" size="small" effect="dark" round class="foot-tag">
            {{ modelBadge.text }}
          </el-tag>
          <el-tag v-if="trainBadge" :type="trainBadge.type" size="small" effect="dark" round class="foot-tag">
            {{ trainBadge.text }}
          </el-tag>
        </div>
      </div>
    </el-drawer>

    <el-container>
      <el-header class="app-header">
        <!-- 窄屏显示汉堡按钮 -->
        <el-button
          v-if="needDrawer"
          class="hamburger"
          size="small"
          text
          @click="drawerOpen = true"
        >
          <el-icon><Menu /></el-icon>
        </el-button>
        <span class="header-title">{{ title }}</span>
        <slot name="header-actions" />
      </el-header>
      <el-main class="app-main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.app-shell {
  height: 100vh;
}

.app-aside {
  background: #171b26;
  display: flex;
  flex-direction: column;
  border-right: 1px solid #232838;
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  height: 60px;
  padding: 0 16px;
  border-bottom: 1px solid #232838;
}

.brand-mark {
  width: 30px;
  height: 30px;
  border-radius: 8px;
  background: linear-gradient(135deg, #e8734a, #b8432a);
  color: #fff;
  font-weight: 700;
  display: grid;
  place-items: center;
}

.brand-text {
  display: flex;
  flex-direction: column;
  line-height: 1.15;
}

.brand-text strong {
  color: #eef1f7;
  font-size: 13.5px;
  letter-spacing: 0.3px;
}

.brand-text small {
  color: #6b7488;
  font-size: 11px;
}

.nav {
  flex: 1;
  padding: 12px 8px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  color: #9aa3b6;
  text-decoration: none;
  font-size: 13.5px;
  transition: background 0.15s, color 0.15s;
}

.nav-item:hover {
  background: #1f2433;
  color: #d6dbe6;
}

.nav-item.active {
  background: #262c3d;
  color: #ff8a5c;
  font-weight: 600;
}

.aside-foot {
  padding: 12px;
  border-top: 1px solid #232838;
  display: flex;
  flex-direction: column;
  gap: 6px;
  /* 防止长内容撑破侧栏产生横向滚动条 */
  min-width: 0;
  overflow: hidden;
}

/*
 * 侧栏宽度固定 208px，标签内容必须能收缩。
 * el-tag 默认 white-space: nowrap 且无最大宽度限制，
 * 模型名一长就顶出滚动条。
 */
.foot-tag {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  display: block;
}

.foot-error {
  margin: 0;
  font-size: 11px;
  color: #e8734a;
  line-height: 1.4;
}

.app-header {
  background: #fff;
  border-bottom: 1px solid #e6e8ec;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 56px;
}

.header-title {
  font-size: 15px;
  font-weight: 600;
  color: #1f2430;
}

.app-main {
  background: #f5f6f8;
  padding: 16px;
  overflow: auto;
  /*
   * min-width:0 是关键。
   * el-main 是 el-container 的 flex 子项，默认 min-width:auto，
   * 子内容比视口宽时它会被撑开，从而产生整页横向滚动条。
   */
  min-width: 0;
}

/* ═══════════════════════════════════════════════════════════════════
   移动端适配
   ═══════════════════════════════════════════════════════════════════ */

/* 汉堡按钮 */
.hamburger {
  margin-right: 8px;
  font-size: 18px;
}

/* 抽屉内部布局（复用桌面侧栏的结构与颜色） */
.drawer-inner {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #171b26;
}

/* 抽屉里的导航项是 button，需去掉默认样式 */
.drawer-item {
  border: none;
  background: transparent;
  width: 100%;
  text-align: left;
  cursor: pointer;
  font-size: 14px;
  font-family: inherit;
}

.drawer-item.active {
  background: #262c3d;
  color: #ff8a5c;
  font-weight: 600;
}

/* Element Plus 抽屉默认有白底，这里覆盖为深色，与侧栏统一 */
:deep(.app-drawer .el-drawer__body) {
  padding: 0;
  background: #171b26;
}

/* ── 平板 / 窄屏桌面 ─────────────────────────────────────────────── */
@media (max-width: 1099px) {
  .app-header {
    height: 52px;
    padding: 0 12px;
  }

  .app-main {
    padding: 10px;
  }
}

/* ── 手机 ────────────────────────────────────────────────────────── */
@media (max-width: 767px) {
  .app-header {
    height: 48px;
    padding: 0 10px;
  }

  .header-title {
    font-size: 14px;
  }

  /* 手机上内容区贴边，节省横向空间 */
  .app-main {
    padding: 8px;
    /* 底部留出安全区（iPhone 刘海屏 / 手势条） */
    padding-bottom: calc(8px + env(safe-area-inset-bottom, 0px));
  }

  /* 手机上滑块/输入框点击区域要够大 */
  :deep(.el-button--small) {
    min-height: 32px;
    padding: 6px 12px;
  }

  /* 弹窗在手机上接近全屏 */
  :deep(.el-dialog) {
    width: 92vw !important;
    margin-top: 4vh !important;
  }

  :deep(.el-dialog__body) {
    max-height: 70vh;
    overflow-y: auto;
  }
}

/* 横屏手机：进一步压缩头部 */
@media (max-width: 900px) and (orientation: landscape) {
  .app-header {
    height: 44px;
  }
}

</style>
