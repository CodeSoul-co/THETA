#!/bin/bash
# DataClean API 启动脚本

# 设置端口（可通过环境变量覆盖）
PORT=${PORT:-8001}

# 启动 API 服务
echo "启动 DataClean API 服务..."
echo "端口: $PORT"
echo "API 文档: http://localhost:$PORT/docs"
echo "按 Ctrl+C 停止服务"
echo ""

python3 api.py
