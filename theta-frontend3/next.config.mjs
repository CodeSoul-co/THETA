/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // 禁用严格模式以避免双重渲染导致的状态问题
  reactStrictMode: false,
  // Turbopack 配置 (Next.js 16+)
  turbopack: {},
}

export default nextConfig
