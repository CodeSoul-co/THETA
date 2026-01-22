# THETA 项目服务器部署完整指南

本指南提供两种部署方式：**Docker Compose（推荐）** 和 **手动部署**。

---

## 📋 前置要求

- **服务器**: Ubuntu 20.04+ / CentOS 7+ / Debian 11+
- **权限**: Root 或 sudo 权限
- **网络**: 公网 IP 或域名
- **端口**: 确保以下端口可用
  - `3000` - 前端应用
  - `8000` - ETM Agent API
  - `8001` - DataClean API
  - `80/443` - Nginx（可选）

---

## 🐳 方案一：Docker Compose 部署（推荐）

### 优点
- ✅ 环境隔离，不污染系统
- ✅ 一键启动所有服务
- ✅ 易于管理和更新
- ✅ 自动重启和健康检查

### 步骤 1: 安装 Docker 和 Docker Compose

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

### 步骤 2: 克隆项目

```bash
# 创建项目目录
sudo mkdir -p /opt/theta
cd /opt/theta

# 克隆项目（替换为你的仓库地址）
git clone https://github.com/your-username/THETA.git
cd THETA

# 或直接下载并解压
# wget https://github.com/your-username/THETA/archive/main.zip
# unzip main.zip
# cd THETA-main
```

### 步骤 3: 配置环境变量

```bash
# 创建 .env 文件
cat > .env << 'EOF'
# 端口配置
API_PORT=8001
FRONTEND_PORT=3000
BACKEND_PORT=8000

# CORS 配置（允许的前端域名）
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com

# 前端环境变量（构建时使用）
NEXT_PUBLIC_API_URL=http://your-server-ip:8000
NEXT_PUBLIC_DATACLEAN_API_URL=http://your-server-ip:8001

# 后端配置（可选）
QWEN_API_KEY=your-qwen-api-key
SECRET_KEY=your-secret-key-change-this
DATABASE_URL=sqlite:///./users.db
EOF

# 编辑配置文件（修改为你的实际值）
nano .env
```

**重要配置说明**：
- `NEXT_PUBLIC_API_URL`: 前端访问后端 ETM Agent API 的地址
- `NEXT_PUBLIC_DATACLEAN_API_URL`: 前端访问 DataClean API 的地址
- 如果使用域名，将 `your-server-ip` 替换为你的域名
- 如果使用 HTTPS，将 `http://` 改为 `https://`

### 步骤 4: 启动服务

```bash
# 构建并启动所有服务
docker-compose up -d --build

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f theta-frontend
docker-compose logs -f dataclean-api
```

### 步骤 5: 验证部署

```bash
# 检查前端
curl http://localhost:3000

# 检查后端 API
curl http://localhost:8000/health
curl http://localhost:8001/health

# 查看所有容器状态
docker-compose ps
```

### 步骤 6: 配置 Nginx 反向代理（可选但推荐）

#### 6.1 安装 Nginx

```bash
sudo apt update
sudo apt install -y nginx
```

#### 6.2 配置前端反向代理

```bash
# 创建 Nginx 配置
sudo nano /etc/nginx/sites-available/theta-frontend
```

添加以下配置（替换 `yourdomain.com` 为你的域名）：

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    # 前端应用
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

    # WebSocket 支持
    location /_next/webpack-hmr {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### 6.3 配置后端 API 反向代理

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

#### 6.4 启用配置

```bash
# 创建软链接
sudo ln -s /etc/nginx/sites-available/theta-frontend /etc/nginx/sites-enabled/
sudo ln -s /etc/nginx/sites-available/theta-api /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重启 Nginx
sudo systemctl restart nginx
sudo systemctl enable nginx
```

### 步骤 7: 配置 HTTPS（使用 Let's Encrypt）

```bash
# 安装 Certbot
sudo apt install -y certbot python3-certbot-nginx

# 获取证书（替换为你的域名）
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
sudo certbot --nginx -d api.yourdomain.com
sudo certbot --nginx -d dataclean.yourdomain.com

# 测试自动续期
sudo certbot renew --dry-run
```

### 常用 Docker Compose 命令

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看日志
docker-compose logs -f

# 更新代码后重新构建
git pull
docker-compose up -d --build

# 清理未使用的资源
docker system prune -a
```

---

## 🛠️ 方案二：手动部署

### 后端部署（ETM Agent API）

#### 1. 安装 Python 和依赖

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv nginx

# 创建虚拟环境
cd /opt/theta/langgraph_agent/backend
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 2. 配置环境变量

```bash
# 创建 .env 文件
cat > .env << 'EOF'
QWEN_API_KEY=your-qwen-api-key
SECRET_KEY=your-secret-key-change-this
DATABASE_URL=sqlite:///./users.db
NEXT_PUBLIC_API_URL=http://your-server-ip:8000
EOF
```

#### 3. 创建 systemd 服务

```bash
sudo nano /etc/systemd/system/theta-backend.service
```

```ini
[Unit]
Description=THETA ETM Agent API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/theta/langgraph_agent/backend
Environment="PATH=/opt/theta/langgraph_agent/backend/venv/bin"
ExecStart=/opt/theta/langgraph_agent/backend/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 启动服务
sudo systemctl daemon-reload
sudo systemctl start theta-backend
sudo systemctl enable theta-backend
sudo systemctl status theta-backend
```

### 前端部署（Next.js）

#### 1. 安装 Node.js 和 pnpm

```bash
# 安装 Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# 安装 pnpm
npm install -g pnpm
```

#### 2. 构建前端

```bash
cd /opt/theta/theta-frontend3

# 安装依赖
pnpm install

# 创建生产环境变量文件
cat > .env.production << 'EOF'
NEXT_PUBLIC_API_URL=http://your-server-ip:8000
NEXT_PUBLIC_DATACLEAN_API_URL=http://your-server-ip:8001
EOF

# 构建
pnpm build
```

#### 3. 使用 PM2 管理进程

```bash
# 安装 PM2
npm install -g pm2

# 编辑 ecosystem.config.js
nano ecosystem.config.js
```

修改配置：

```javascript
module.exports = {
  apps: [{
    name: 'theta-frontend',
    script: 'npm',
    args: 'start',
    cwd: '/opt/theta/theta-frontend3',
    instances: 1,
    exec_mode: 'fork',
    env: {
      NODE_ENV: 'production',
      PORT: 3000,
      NEXT_PUBLIC_API_URL: 'http://your-server-ip:8000',
      NEXT_PUBLIC_DATACLEAN_API_URL: 'http://your-server-ip:8001'
    },
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G'
  }]
}
```

```bash
# 启动应用
pm2 start ecosystem.config.js

# 保存配置
pm2 save

# 设置开机自启
pm2 startup
# 按照提示执行生成的命令
```

---

## 🔒 安全配置

### 1. 配置防火墙

```bash
# Ubuntu (UFW)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable

# CentOS (firewalld)
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --reload
```

### 2. 限制 SSH 访问

```bash
# 编辑 SSH 配置
sudo nano /etc/ssh/sshd_config

# 修改以下配置
PermitRootLogin no
PasswordAuthentication no  # 使用密钥认证
Port 2222  # 修改默认端口

# 重启 SSH
sudo systemctl restart sshd
```

### 3. 定期更新系统

```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y

# CentOS
sudo yum update -y
```

---

## 🔄 更新部署

### Docker Compose 方式

```bash
cd /opt/theta/THETA

# 拉取最新代码
git pull

# 重新构建并启动
docker-compose down
docker-compose up -d --build

# 清理旧镜像
docker image prune -a
```

### 手动部署方式

```bash
# 后端更新
cd /opt/theta/langgraph_agent/backend
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart theta-backend

# 前端更新
cd /opt/theta/theta-frontend3
git pull
pnpm install
pnpm build
pm2 restart theta-frontend
```

---

## 🐛 故障排查

### 1. 服务无法启动

```bash
# 查看日志
docker-compose logs [service-name]
# 或
sudo journalctl -u theta-backend -n 50
pm2 logs theta-frontend

# 检查端口占用
sudo netstat -tlnp | grep -E '8000|8001|3000'
```

### 2. 前端无法连接后端

- 检查环境变量 `NEXT_PUBLIC_API_URL` 是否正确
- 检查后端服务是否运行
- 检查防火墙和 CORS 配置
- 查看浏览器控制台错误信息

### 3. 构建失败

```bash
# 清理缓存
docker-compose down -v
rm -rf node_modules .next
pnpm install
pnpm build
```

---

## 📊 监控和维护

### 查看资源使用

```bash
# Docker 资源使用
docker stats

# 系统资源
htop
# 或
top
```

### 日志管理

```bash
# Docker 日志
docker-compose logs --tail=100 -f

# PM2 日志
pm2 logs --lines 100

# 系统日志
sudo journalctl -u theta-backend -f
```

---

## ✅ 部署检查清单

- [ ] Docker 和 Docker Compose 已安装
- [ ] 项目代码已克隆到服务器
- [ ] 环境变量已正确配置
- [ ] 所有服务已启动并运行
- [ ] 端口已开放（防火墙配置）
- [ ] Nginx 反向代理已配置（可选）
- [ ] HTTPS 证书已配置（可选）
- [ ] 域名 DNS 已解析（可选）
- [ ] 功能测试通过
- [ ] 监控和日志已配置

---

## 📞 获取帮助

如果遇到问题：

1. 查看日志文件
2. 检查环境变量配置
3. 验证网络连接
4. 查看项目 Issues
5. 联系技术支持

---

**部署完成后，访问你的域名或 IP 地址即可使用 THETA 项目！** 🎉
