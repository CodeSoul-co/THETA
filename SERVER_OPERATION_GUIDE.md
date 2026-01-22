# 服务器操作指南

本指南说明如何在服务器上部署和更新 THETA 前端项目。

---

## 🚀 首次部署

### 步骤 1: 连接到服务器

```bash
# 使用 SSH 连接到服务器
ssh root@your-server-ip
# 或
ssh username@your-server-ip
```

### 步骤 2: 安装 Docker（如果未安装）

```bash
# Ubuntu/Debian
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
git clone https://github.com/CodeSoul-co/THETA.git
cd THETA

# 切换到正确的分支（如果需要）
git checkout frontend-3
```

### 步骤 4: 配置环境变量

```bash
# 创建前端环境变量文件
cat > .env.frontend << 'EOF'
# 前端端口（标准 HTTP 端口）
FRONTEND_PORT=80

# 后端 API 地址（如果后端未完成，暂时使用 localhost）
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_DATACLEAN_API_URL=http://localhost:8001

# 如果后端部署在其他服务器，设置为实际地址：
# NEXT_PUBLIC_API_URL=https://api.yourdomain.com
# NEXT_PUBLIC_DATACLEAN_API_URL=https://dataclean.yourdomain.com
EOF

# 编辑配置文件（根据需要修改）
nano .env.frontend
```

### 步骤 5: 首次部署

```bash
# 给脚本添加执行权限
sudo chmod +x deploy-frontend.sh

# 运行部署脚本
sudo ./deploy-frontend.sh
```

或者手动部署：

```bash
# 构建并启动
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d --build

# 查看日志
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs -f
```

### 步骤 6: 验证部署

```bash
# 检查容器状态
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend ps

# 测试访问
curl http://localhost:80
```

---

## 🔄 更新部署（代码更新后）

### 方法一：使用更新脚本（推荐）

```bash
cd /opt/theta/THETA

# 给脚本添加执行权限（首次使用）
sudo chmod +x deploy-frontend-update.sh

# 运行更新脚本（会自动 git pull + 重新构建 + 重启）
sudo ./deploy-frontend-update.sh
```

### 方法二：手动更新

```bash
cd /opt/theta/THETA

# 1. 拉取最新代码
git pull

# 2. 停止现有容器
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend down

# 3. 重新构建并启动（重要：必须加 --build）
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d --build

# 4. 查看日志确认
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs -f
```

---

## 📋 常用操作命令

### 查看服务状态

```bash
cd /opt/theta/THETA
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend ps
```

### 查看日志

```bash
# 查看所有日志
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs -f

# 查看最近 50 行日志
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs --tail=50

# 查看特定服务的日志
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs -f theta-frontend
```

### 重启服务

```bash
# 重启（不重新构建）
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend restart

# 停止服务
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend down

# 启动服务
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d
```

### 重新构建（代码或环境变量更新后）

```bash
# 停止
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend down

# 重新构建并启动
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d --build
```

---

## 🔧 配置 Nginx 反向代理（推荐）

### 安装 Nginx

```bash
sudo apt update
sudo apt install -y nginx
```

### 配置反向代理

```bash
# 创建 Nginx 配置
sudo nano /etc/nginx/sites-available/theta-frontend
```

添加以下配置：

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:3000;  # 容器内部端口
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

启用配置：

```bash
# 创建软链接
sudo ln -s /etc/nginx/sites-available/theta-frontend /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重启 Nginx
sudo systemctl restart nginx
```

### 配置 SSL（HTTPS）

```bash
# 安装 Certbot
sudo apt install -y certbot python3-certbot-nginx

# 申请证书
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# 测试自动续期
sudo certbot renew --dry-run
```

---

## 🐛 故障排查

### 问题 1: 容器无法启动

```bash
# 查看详细日志
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs theta-frontend

# 检查端口占用
sudo netstat -tlnp | grep 80
```

### 问题 2: 更新后没有生效

```bash
# 确认代码已更新
git log -1

# 确认重新构建了（必须使用 --build）
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d --build

# 清理缓存重新构建
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend build --no-cache
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d
```

### 问题 3: 端口被占用

```bash
# 检查端口占用
sudo netstat -tlnp | grep 80

# 如果 80 端口被占用，修改 .env.frontend 中的端口
nano .env.frontend
# 修改 FRONTEND_PORT=3000
```

### 问题 4: 权限问题

```bash
# 如果使用 80 端口，需要 root 权限
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d

# 或使用 Nginx 反向代理（推荐，不需要 root）
```

---

## 📝 环境变量说明

### 必需的环境变量

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `FRONTEND_PORT` | 前端端口 | `80` 或 `3000` |
| `NEXT_PUBLIC_API_URL` | ETM Agent API 地址 | `http://localhost:8000` |
| `NEXT_PUBLIC_DATACLEAN_API_URL` | DataClean API 地址 | `http://localhost:8001` |

### 修改环境变量后

如果修改了 `.env.frontend` 文件中的 `NEXT_PUBLIC_*` 变量，**必须重新构建**：

```bash
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend down
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d --build
```

如果只修改了 `FRONTEND_PORT`，只需要重启：

```bash
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend restart
```

---

## 🔄 完整更新流程总结

```bash
# 1. 连接到服务器
ssh root@your-server-ip

# 2. 进入项目目录
cd /opt/theta/THETA

# 3. 拉取最新代码
git pull

# 4. 停止容器
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend down

# 5. 重新构建并启动
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d --build

# 6. 查看日志确认
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs -f
```

**或者使用一键脚本**：

```bash
cd /opt/theta/THETA
sudo ./deploy-frontend-update.sh
```

---

## 📚 相关文档

- `FRONTEND_ONLY_DEPLOY.md` - 前端单独部署详细指南
- `UPDATE_FRONTEND.md` - 更新部署详细说明
- `BT_PANEL_DEPLOY.md` - 宝塔面板部署指南
- `QUICK_DEPLOY.md` - 快速部署指南

---

## ✅ 检查清单

部署前确认：
- [ ] Docker 和 Docker Compose 已安装
- [ ] 项目已克隆到服务器
- [ ] `.env.frontend` 文件已配置
- [ ] 端口 80 或 3000 可用
- [ ] 防火墙已开放相应端口

更新前确认：
- [ ] 代码已推送到 Git 仓库
- [ ] 服务器上已执行 `git pull`
- [ ] 使用 `--build` 参数重新构建
- [ ] 查看日志确认启动成功

---

**记住：代码更新后，必须重新构建 Docker 镜像才能生效！** 🔄
