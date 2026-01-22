#!/bin/bash
# THETA 前端单独部署脚本

set -e

echo "🚀 THETA 前端单独部署脚本"
echo "================================"

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: 未找到 Docker"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查 Docker Compose 是否安装
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ 错误: 未找到 Docker Compose"
    echo "请先安装 Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# 检查 Docker 服务是否运行
if ! docker info &> /dev/null; then
    echo "❌ 错误: Docker 服务未运行"
    echo "请启动 Docker 服务: sudo systemctl start docker"
    exit 1
fi

echo "✅ Docker 环境检查通过"
echo ""

# 检查是否在 Git 仓库中，如果是则拉取最新代码
if [ -d ".git" ]; then
    echo "📥 检测到 Git 仓库，拉取最新代码..."
    git pull || echo "⚠️  Git pull 失败，继续使用当前代码"
    echo ""
fi

# 检查 .env.frontend 文件
if [ ! -f ".env.frontend" ]; then
    echo "📝 创建 .env.frontend 文件..."
    if [ -f ".env.frontend.example" ]; then
        cp .env.frontend.example .env.frontend
        echo "✅ 已从 .env.frontend.example 创建 .env.frontend 文件"
        echo "⚠️  请编辑 .env.frontend 文件，设置正确的配置值"
        echo ""
        read -p "是否现在编辑 .env.frontend 文件? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ${EDITOR:-nano} .env.frontend
        fi
    else
        echo "⚠️  未找到 .env.frontend.example，请手动创建 .env.frontend 文件"
    fi
fi

# 停止现有容器（如果存在）
echo "🛑 停止现有容器..."
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend down 2>/dev/null || true

# 清理旧镜像（可选，取消注释以启用）
# echo "🧹 清理旧镜像..."
# docker image prune -f

# 构建镜像
echo "🔨 构建 Docker 镜像..."
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend build --no-cache

# 启动服务
echo "🚀 启动服务..."
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo ""
echo "📊 服务状态:"
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend ps

# 检查健康状态
echo ""
echo "🏥 健康检查:"
# 从环境变量文件读取端口
FRONTEND_PORT_VALUE=$(grep FRONTEND_PORT .env.frontend 2>/dev/null | cut -d '=' -f2 | tr -d ' ' || echo "80")
if [ -z "$FRONTEND_PORT_VALUE" ] || [ "$FRONTEND_PORT_VALUE" = "" ]; then
    FRONTEND_PORT_VALUE=80
fi

echo -n "前端 (端口 $FRONTEND_PORT_VALUE): "
if curl -s http://localhost:$FRONTEND_PORT_VALUE > /dev/null 2>&1; then
    echo "✅ 运行正常"
else
    echo "❌ 无法访问，请检查日志: docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs theta-frontend"
fi

echo ""
echo "✅ 前端部署完成！"
echo ""
echo "📋 常用命令:"
echo "  查看日志:     docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs -f"
echo "  停止服务:     docker-compose -f docker-compose.frontend.yml --env-file .env.frontend down"
echo "  重启服务:     docker-compose -f docker-compose.frontend.yml --env-file .env.frontend restart"
echo "  查看状态:     docker-compose -f docker-compose.frontend.yml --env-file .env.frontend ps"
echo ""
echo "🌐 访问地址:"
FRONTEND_PORT_VALUE=${FRONTEND_PORT:-80}
if [ "$FRONTEND_PORT_VALUE" = "80" ]; then
    echo "  前端: http://localhost (标准 HTTP 端口 80)"
else
    echo "  前端: http://localhost:$FRONTEND_PORT_VALUE"
fi
echo ""
echo "⚠️  注意:"
echo "   - 后端服务未部署，部分功能可能不可用"
echo "   - 使用 80 端口需要 root 权限，或使用 Nginx 反向代理"
echo "   - 后端完成后，更新 .env.frontend 中的 API 地址并重新部署"
