# Docker 快速部署指南

## 🚀 5 分钟快速部署

### 步骤 1: 安装 Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo systemctl start docker
sudo systemctl enable docker

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 步骤 2: 克隆项目

```bash
cd /opt
sudo git clone https://github.com/CodeSoul-co/THETA.git
cd THETA
sudo git checkout frontend-3
```

### 步骤 3: 一键部署

```bash
# 运行自动部署脚本
sudo chmod +x docker-deploy.sh
sudo ./docker-deploy.sh
```

脚本会自动完成所有配置和部署！

### 步骤 4: 访问应用

- **前端**: http://your-server-ip:3000
- **后端 API**: http://your-server-ip:8001
- **健康检查**: http://your-server-ip:8001/health

## 📋 常用命令

```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 更新代码
git pull
docker-compose up -d --build
```

## ⚙️ 配置说明

### 环境变量配置（可选）

如果需要自定义配置，编辑 `.env` 文件：

```bash
sudo nano .env
```

主要配置项：
- `API_PORT`: 后端端口（默认 8001）
- `FRONTEND_PORT`: 前端端口（默认 3000）
- `ALLOWED_ORIGINS`: CORS 允许的源
- `NEXT_PUBLIC_DATACLEAN_API_URL`: 前端访问后端的 URL

### 配置 Nginx 反向代理（可选）

如果需要使用域名访问：

```bash
# 1. 安装 Nginx
sudo apt install -y nginx

# 2. 复制配置
sudo cp theta-frontend3/nginx-frontend.conf.example /etc/nginx/sites-available/theta-frontend
sudo cp ETM/dataclean/nginx.conf.example /etc/nginx/sites-available/dataclean-api

# 3. 编辑配置（修改域名）
sudo nano /etc/nginx/sites-available/theta-frontend
sudo nano /etc/nginx/sites-available/dataclean-api

# 4. 启用配置
sudo ln -s /etc/nginx/sites-available/theta-frontend /etc/nginx/sites-enabled/
sudo ln -s /etc/nginx/sites-available/dataclean-api /etc/nginx/sites-enabled/

# 5. 重启 Nginx
sudo nginx -t
sudo systemctl restart nginx
```

## 🔧 故障排查

### 服务无法启动

```bash
# 查看日志
docker-compose logs [service-name]

# 检查端口占用
sudo netstat -tlnp | grep -E '8001|3000'
```

### 端口冲突

修改 `.env` 文件中的端口配置，然后重启：

```bash
docker-compose down
docker-compose up -d
```

## 📚 详细文档

更多详细信息请查看：
- `DOCKER_DEPLOY.md` - 完整部署指南
- `SERVER_DEPLOYMENT.md` - 服务器部署指南
