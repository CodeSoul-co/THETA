#!/bin/bash
# 离线构建准备脚本
# 在本机运行此脚本，下载所有需要的文件，然后上传到服务器

set -e

echo "📦 准备离线 Docker 构建文件"
echo "================================"

# 创建输出目录
OUTPUT_DIR="docker-offline-build"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "1️⃣  下载 Python 基础镜像..."
docker pull python:3.11-slim
docker save python:3.11-slim -o "$OUTPUT_DIR/python-3.11-slim.tar"

echo ""
echo "2️⃣  下载 Node.js 基础镜像..."
docker pull node:20-alpine
docker save node:20-alpine -o "$OUTPUT_DIR/node-20-alpine.tar"

echo ""
echo "3️⃣  下载 Python 依赖包（wheel 文件）..."
cd ETM/dataclean
mkdir -p ../../$OUTPUT_DIR/wheels

# 创建临时虚拟环境下载依赖
python3 -m venv /tmp/theta-venv
source /tmp/theta-venv/bin/activate
pip install --upgrade pip
pip download -r requirements.txt -d ../../$OUTPUT_DIR/wheels
pip download fastapi uvicorn[standard] python-multipart -d ../../$OUTPUT_DIR/wheels
deactivate
cd ../..

echo ""
echo "4️⃣  打包项目代码..."
tar -czf "$OUTPUT_DIR/theta-code.tar.gz" \
    --exclude='.git' \
    --exclude='node_modules' \
    --exclude='.next' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    ETM/dataclean theta-frontend3 docker-compose.yml docker-deploy.sh docker.env.template

echo ""
echo "5️⃣  创建部署脚本..."
cat > "$OUTPUT_DIR/deploy-offline.sh" << 'EOF'
#!/bin/bash
# 离线部署脚本（在服务器上运行）

set -e

echo "🚀 离线 Docker 部署"
echo "================================"

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: 未找到 Docker"
    exit 1
fi

echo ""
echo "1️⃣  导入 Docker 镜像..."
docker load -i python-3.11-slim.tar
docker load -i node-20-alpine.tar

echo ""
echo "2️⃣  解压项目代码..."
tar -xzf theta-code.tar.gz -C /tmp
cd /tmp

echo ""
echo "3️⃣  构建 Docker 镜像..."
cd ETM/dataclean
# 使用离线 Dockerfile
if [ -f Dockerfile.offline ]; then
    docker build -f Dockerfile.offline -t dataclean-api:latest .
else
    docker build -t dataclean-api:latest .
fi
cd ../../theta-frontend3
docker build -t theta-frontend:latest .
cd ../..

echo ""
echo "4️⃣  启动服务..."
if [ -f docker-compose.yml ]; then
    docker-compose up -d
else
    echo "⚠️  未找到 docker-compose.yml，请手动启动容器"
fi

echo ""
echo "✅ 部署完成！"
EOF

chmod +x "$OUTPUT_DIR/deploy-offline.sh"

echo ""
echo "✅ 准备完成！"
echo ""
echo "📋 下一步："
echo "1. 将 $OUTPUT_DIR 目录上传到服务器"
echo "2. 在服务器上运行:"
echo "   cd $OUTPUT_DIR"
echo "   chmod +x deploy-offline.sh"
echo "   ./deploy-offline.sh"
echo ""
echo "📦 文件列表："
ls -lh "$OUTPUT_DIR"
