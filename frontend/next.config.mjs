/** @type {import('next').NextConfig} */
const nextConfig = {
  // Leave room for the 200 MiB dataset plus multipart framing; proxy otherwise truncates at 10 MiB.
  experimental: { proxyClientMaxBodySize: '202mb' },
  typescript: {
    ignoreBuildErrors: false,
  },
  images: {
    // unoptimized: true 已移除 — 启用 Next.js 图片优化（WebP/AVIF 自动转换 + lazy loading）
    formats: ['image/webp', 'image/avif'],
  },
  // 禁用严格模式以避免双重渲染导致的状态问题
  reactStrictMode: false,
  // Turbopack 配置 (Next.js 16+)
  turbopack: {
    resolveAlias: {},
  },
  // 开发服务器配置
  devIndicators: {
    buildActivity: false,
  },
  // 环境变量配置
  // 本地服务通过启动器或 .env.local 配置
  // 输出配置
  // 桌面安装包使用 standalone 模式
  output: 'standalone',
  async redirects() {
    return [
      { source: '/training', destination: '/workbench', permanent: true },
      { source: '/login', destination: '/', permanent: true },
      { source: '/register', destination: '/', permanent: true },
      { source: '/results', destination: '/workbench', permanent: true },
      { source: '/visualizations', destination: '/workbench', permanent: true },
    ]
  },
}

export default nextConfig
