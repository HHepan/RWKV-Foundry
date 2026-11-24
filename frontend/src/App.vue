<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { metaApi } from '@/api/chat'

const route = useRoute()
const health = ref(null)
const healthError = ref('')

const title = computed(() => route.meta.title || 'RWKV-Foundry')
const modelBadge = computed(() => {
  if (!health.value) return { text: '未连接', type: 'info' }
  if (health.value.model_loaded) {
    return { text: `已加载 ${health.value.model_ref || ''}`, type: 'success' }
  }
  return { text: '未加载模型', type: 'info' }
})
const trainBadge = computed(() => {
  if (health.value?.active_train) return { text: `训练中 ${health.value.active_train}`, type: 'warning' }
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
        <el-tag :type="modelBadge.type" size="small" effect="dark" round>
          {{ modelBadge.text }}
        </el-tag>
        <el-tag v-if="trainBadge" :type="trainBadge.type" size="small" effect="dark" round>
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
