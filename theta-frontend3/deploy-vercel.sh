#!/bin/bash

# Vercel 部署脚本
# 使用方法: ./deploy-vercel.sh [production|preview]

set -e

ENV=${1:-production}

echo "🚀 开始部署到 Vercel ($ENV 环境)..."

# 检查是否安装了 Vercel CLI
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI 未安装"
    echo "📦 正在安装 Vercel CLI..."
    npm install -g vercel
fi

# 检查是否已登录
if ! vercel whoami &> /dev/null; then
    echo "🔐 请先登录 Vercel..."
    vercel login
fi

# 检查环境变量
echo "📋 检查环境变量..."
if [ -z "$NEXT_PUBLIC_API_URL" ]; then
    echo "⚠️  警告: NEXT_PUBLIC_API_URL 未设置"
    echo "   请在 Vercel Dashboard 中设置，或运行:"
    echo "   vercel env add NEXT_PUBLIC_API_URL $ENV"
fi

if [ -z "$NEXT_PUBLIC_DATACLEAN_API_URL" ]; then
    echo "⚠️  警告: NEXT_PUBLIC_DATACLEAN_API_URL 未设置"
    echo "   请在 Vercel Dashboard 中设置，或运行:"
    echo "   vercel env add NEXT_PUBLIC_DATACLEAN_API_URL $ENV"
fi

# 部署
if [ "$ENV" = "production" ]; then
    echo "🌐 部署到生产环境..."
    vercel --prod
else
    echo "🔍 部署预览版本..."
    vercel
fi

echo "✅ 部署完成！"
echo "📝 提示: 在 Vercel Dashboard 中查看部署状态和日志"
