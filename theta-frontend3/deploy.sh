#!/bin/bash
# Next.js 前端服务器部署脚本

set -e

echo "🚀 开始部署 THETA 前端..."

# 检查 Node.js 版本
if ! command -v node &> /dev/null; then
    echo "❌ 错误: 未找到 Node.js"
    exit 1
fi

NODE_VERSION=$(node --version)
echo "✅ Node.js 版本: $NODE_VERSION"

# 检查 pnpm
if ! command -v pnpm &> /dev/null; then
    echo "📦 安装 pnpm..."
    npm install -g pnpm
fi

# 安装依赖
echo "📥 安装依赖..."
pnpm install

# 构建项目
echo "🔨 构建项目..."
pnpm build

echo "✅ 构建完成！"
echo ""
echo "启动生产服务器:"
echo "  pnpm start"
echo ""
echo "或使用 PM2:"
echo "  pm2 start npm --name 'theta-frontend' -- start"
echo "  pm2 save"
echo "  pm2 startup"
