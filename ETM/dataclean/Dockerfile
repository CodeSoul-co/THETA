FROM python:3.11-slim

WORKDIR /app

# 配置 Debian 软件源（使用官方源，更稳定）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security/ bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list

# 安装系统依赖（使用官方源作为备选）
RUN apt-get update || ( \
        sed -i 's/mirrors.aliyun.com/deb.debian.org/g' /etc/apt/sources.list* && \
        apt-get update \
    ) && \
    apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p temp_uploads temp_processed && \
    chmod 755 temp_uploads temp_processed

# 暴露端口
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# 启动命令
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
