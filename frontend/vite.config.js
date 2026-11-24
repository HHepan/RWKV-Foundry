import { fileURLToPath, URL } from 'node:url'
import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  // 后端地址可覆盖，避免旧版把 localhost:8000 写死在 vite.config.js 里
  const target = env.FOUNDRY_API || 'http://127.0.0.1:8000'

  return {
    plugins: [vue()],
    resolve: {
      alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
    },
    server: {
      port: 5173,
      // 旧版默认 --host 0.0.0.0 但 CORS 只允许 localhost，用局域网 IP 打开必然失败。
      // 现在后端默认已包含 127.0.0.1:5173，两者对齐。
      host: env.FOUNDRY_HOST || '127.0.0.1',
      proxy: {
        // 走代理可绕开 CORS；SSE 需要关闭缓冲
        '/api': {
          target,
          changeOrigin: true,
          // 关键：SSE 不能被代理缓冲，否则前端收不到增量
          configure: (proxy) => {
            proxy.on('proxyRes', (proxyRes) => {
              proxyRes.headers['cache-control'] = 'no-cache'
              delete proxyRes.headers['content-length']
            })
          },
        },
      },
    },
    build: {
      outDir: 'dist',
      // 图表与 UI 库分离，避免主包过大
      rollupOptions: {
        output: {
          manualChunks: {
            vendor: ['vue', 'vue-router', 'axios'],
            ui: ['element-plus', '@element-plus/icons-vue'],
            chart: ['echarts'],
          },
        },
      },
    },
  }
})
