<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { metaApi } from '@/api/chat'

const route = useRoute()
const health = ref(null)
const healthError = ref('')

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
    <el-aside width="208px" class="app-aside">
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

    <el-container>
      <el-header class="app-header">
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
}
</style>
