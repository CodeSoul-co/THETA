# 🚀 THETA 项目快速部署指南

> 💡 **使用宝塔面板？** 查看 [宝塔面板部署指南](./BT_PANEL_DEPLOY.md) 获取图形化部署教程。

## 5 分钟快速部署到服务器

### 步骤 1: 准备服务器

确保你的服务器满足以下要求：
- Ubuntu 20.04+ / CentOS 7+ / Debian 11+
- Root 或 sudo 权限
- 至少 2GB RAM
- 至少 10GB 磁盘空间

### 步骤 2: 安装 Docker

```bash
# 一键安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo systemctl start docker
sudo systemctl enable docker

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 验证安装
docker --version
docker-compose --version
```

### 步骤 3: 克隆项目

```bash
# 创建项目目录
sudo mkdir -p /opt/theta
cd /opt/theta

# 克隆项目（替换为你的仓库地址）
git clone https://github.com/your-username/THETA.git
cd THETA
```

### 步骤 4: 配置环境变量

```bash
# 复制环境变量模板
cp docker.env.template .env

# 编辑配置文件
nano .env
```

**重要配置项**（根据你的实际情况修改）：

```bash
# 如果使用域名，替换为你的域名
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_DATACLEAN_API_URL=https://dataclean.yourdomain.com

# 如果直接使用 IP，替换为你的服务器 IP
# NEXT_PUBLIC_API_URL=http://your-server-ip:8000
# NEXT_PUBLIC_DATACLEAN_API_URL=http://your-server-ip:8001

# 设置 CORS（允许的前端域名）
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# 设置千问 API Key（可选，用于 AI 助手）
QWEN_API_KEY=your-qwen-api-key

# 设置 JWT 密钥（请修改为随机字符串）
SECRET_KEY=your-random-secret-key-here
```

### 步骤 5: 一键部署

```bash
# 运行自动部署脚本
sudo chmod +x docker-deploy.sh
sudo ./docker-deploy.sh
```

或者手动部署：

```bash
# 构建并启动所有服务
docker-compose up -d --build

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 步骤 6: 验证部署

```bash
# 检查服务是否运行
docker-compose ps

# 测试 API 健康检查
curl http://localhost:8000/health
curl http://localhost:8001/health

# 测试前端
curl http://localhost:3000
```

### 步骤 7: 配置防火墙（可选）

```bash
# Ubuntu/Debian (UFW)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp     # HTTP
sudo ufw allow 443/tcp    # HTTPS
sudo ufw allow 3000/tcp   # 前端（如果直接暴露）
sudo ufw allow 8000/tcp   # 后端 API（如果直接暴露）
sudo ufw allow 8001/tcp   # DataClean API（如果直接暴露）
sudo ufw enable
```

### 步骤 8: 配置 Nginx 反向代理（推荐）

如果你有域名，建议使用 Nginx 反向代理：

```bash
# 安装 Nginx
sudo apt update
sudo apt install -y nginx

# 创建前端配置
sudo nano /etc/nginx/sites-available/theta-frontend
```

前端配置：

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

API 配置：

```bash
sudo nano /etc/nginx/sites-available/theta-api
```

```nginx
# ETM Agent API
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# DataClean API
server {
    listen 80;
    server_name dataclean.yourdomain.com;

    location / {
        proxy_pass http://localhost:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

启用配置：

```bash
sudo ln -s /etc/nginx/sites-available/theta-frontend /etc/nginx/sites-enabled/
sudo ln -s /etc/nginx/sites-available/theta-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 步骤 9: 配置 HTTPS（推荐）

```bash
# 安装 Certbot
sudo apt install -y certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
sudo certbot --nginx -d api.yourdomain.com
sudo certbot --nginx -d dataclean.yourdomain.com

# 测试自动续期
sudo certbot renew --dry-run
```

## ✅ 部署完成！

访问你的域名或 IP 地址：
- **前端**: `http://yourdomain.com` 或 `http://your-server-ip:3000`
- **ETM Agent API**: `http://api.yourdomain.com` 或 `http://your-server-ip:8000`
- **DataClean API**: `http://dataclean.yourdomain.com` 或 `http://your-server-ip:8001`

## 📋 常用命令

```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 重启服务
docker-compose restart

# 停止服务
docker-compose down

# 更新代码
git pull
docker-compose up -d --build

# 查看特定服务日志
docker-compose logs -f theta-frontend
docker-compose logs -f etm-agent-api
docker-compose logs -f dataclean-api
```

## 🐛 故障排查

### 服务无法启动

```bash
# 查看详细日志
docker-compose logs [service-name]

# 检查端口占用
sudo netstat -tlnp | grep -E '8000|8001|3000'
```

### 前端无法连接后端

1. 检查 `.env` 文件中的 `NEXT_PUBLIC_API_URL` 和 `NEXT_PUBLIC_DATACLEAN_API_URL`
2. 确保后端服务正在运行：`docker-compose ps`
3. 检查 CORS 配置：`ALLOWED_ORIGINS`
4. 查看浏览器控制台错误信息

### 构建失败

```bash
# 清理并重新构建
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

## 📚 更多信息

详细部署文档请查看：
- `SERVER_DEPLOY_GUIDE.md` - 完整部署指南
- `DOCKER_DEPLOY.md` - Docker 部署详细说明

---

**需要帮助？** 查看项目 Issues 或联系技术支持。
