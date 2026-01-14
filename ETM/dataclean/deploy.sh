#!/bin/bash
# DataClean API 服务器部署脚本

set -e

echo "🚀 开始部署 DataClean API..."

# 检查 Python 版本
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python 3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python 版本: $PYTHON_VERSION"

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 升级 pip
echo "⬆️  升级 pip..."
pip install --upgrade pip

# 安装依赖
echo "📥 安装依赖..."
pip install -r requirements.txt

# 创建必要的目录
echo "📁 创建目录..."
mkdir -p temp_uploads temp_processed logs

# 设置权限
chmod +x start_api.sh

echo "✅ 部署完成！"
echo ""
echo "启动服务:"
echo "  ./start_api.sh"
echo ""
echo "或使用 systemd:"
echo "  sudo systemctl start dataclean-api"
echo "  sudo systemctl enable dataclean-api"
