#!/bin/bash

# THETA 前端重启脚本

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/theta-frontend3"

echo "🔄 正在重启 THETA 前端..."

# 停止现有进程
echo "1. 停止现有前端进程..."
lsof -ti:3000 | xargs kill -9 2>/dev/null && echo "   ✓ 已停止端口 3000 上的进程" || echo "   ℹ 端口 3000 未被占用"

# 等待端口释放
sleep 1

# 进入前端目录
cd "$FRONTEND_DIR"

# 检查 pnpm
if ! command -v pnpm &> /dev/null; then
    echo "❌ 错误: 未找到 pnpm"
    echo "请先安装: npm install -g pnpm"
    exit 1
fi

# 启动前端
echo "2. 启动前端开发服务器..."
echo "   → 前端地址: http://localhost:3000"
echo "   → API 地址: http://localhost:6006 (通过 SSH 端口转发)"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

pnpm dev
