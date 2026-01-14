# 服务器部署指南

本指南介绍如何在您自己的服务器上部署 THETA 项目（前端 + 后端）。

## 前置要求

- Ubuntu 20.04+ 或 CentOS 7+ 服务器
- Root 或 sudo 权限
- 域名（可选，用于 HTTPS）

## 方案一：Docker Compose 部署（推荐）

### 优点
- 简单快速
- 环境隔离
- 易于管理

### 部署步骤

1. **安装 Docker 和 Docker Compose**
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 验证安装
docker --version
docker-compose --version
```

2. **克隆项目**
```bash
cd /opt
git clone https://github.com/CodeSoul-co/THETA.git
cd THETA
git checkout frontend-3
```

3. **配置环境变量**
```bash
# 编辑 docker-compose.yml，修改环境变量
nano docker-compose.yml
```

4. **启动服务**
```bash
docker-compose up -d
```

5. **查看日志**
```bash
docker-compose logs -f
```

6. **配置 Nginx（可选）**
```bash
# 复制 Nginx 配置
sudo cp theta-frontend3/nginx-frontend.conf.example /etc/nginx/sites-available/theta-frontend
sudo cp ETM/dataclean/nginx.conf.example /etc/nginx/sites-available/dataclean-api

# 编辑配置文件，修改域名
sudo nano /etc/nginx/sites-available/theta-frontend
sudo nano /etc/nginx/sites-available/dataclean-api

# 创建软链接
sudo ln -s /etc/nginx/sites-available/theta-frontend /etc/nginx/sites-enabled/
sudo ln -s /etc/nginx/sites-available/dataclean-api /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重启 Nginx
sudo systemctl restart nginx
```

## 方案二：手动部署

### 后端部署（DataClean API）

1. **安装依赖**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv nginx

# CentOS/RHEL
sudo yum install -y python3 python3-pip nginx
```

2. **部署应用**
```bash
# 创建目录
sudo mkdir -p /opt/dataclean
cd /opt/dataclean

# 克隆或复制项目文件
sudo git clone https://github.com/CodeSoul-co/THETA.git
cd THETA/ETM/dataclean

# 运行部署脚本
chmod +x deploy.sh
sudo ./deploy.sh
```

3. **配置 systemd 服务**
```bash
# 复制服务文件
sudo cp dataclean-api.service /etc/systemd/system/

# 编辑服务文件，修改路径
sudo nano /etc/systemd/system/dataclean-api.service

# 重新加载 systemd
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start dataclean-api
sudo systemctl enable dataclean-api

# 查看状态
sudo systemctl status dataclean-api
```

4. **配置 Nginx**
```bash
# 复制配置文件
sudo cp nginx.conf.example /etc/nginx/sites-available/dataclean-api

# 编辑配置
sudo nano /etc/nginx/sites-available/dataclean-api

# 创建软链接
sudo ln -s /etc/nginx/sites-available/dataclean-api /etc/nginx/sites-enabled/

# 测试并重启
sudo nginx -t
sudo systemctl restart nginx
```

### 前端部署（Next.js）

1. **安装 Node.js 和 pnpm**
```bash
# 安装 Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# 安装 pnpm
npm install -g pnpm
```

2. **部署应用**
```bash
# 创建目录
sudo mkdir -p /opt/theta-frontend3
cd /opt/theta-frontend3

# 克隆项目
sudo git clone https://github.com/CodeSoul-co/THETA.git
cd THETA/theta-frontend3

# 运行部署脚本
chmod +x deploy.sh
sudo ./deploy.sh
```

3. **配置环境变量**
```bash
# 创建 .env.production
nano .env.production

# 添加以下内容（修改为实际 API 地址）
NEXT_PUBLIC_DATACLEAN_API_URL=http://api.yourdomain.com
```

4. **使用 PM2 管理进程**
```bash
# 安装 PM2
npm install -g pm2

# 编辑 ecosystem.config.js，修改路径和 API URL
nano ecosystem.config.js

# 启动应用
pm2 start ecosystem.config.js

# 保存配置
pm2 save

# 设置开机自启
pm2 startup
```

5. **配置 Nginx**
```bash
# 复制配置文件
sudo cp nginx-frontend.conf.example /etc/nginx/sites-available/theta-frontend

# 编辑配置
sudo nano /etc/nginx/sites-available/theta-frontend

# 创建软链接
sudo ln -s /etc/nginx/sites-available/theta-frontend /etc/nginx/sites-enabled/

# 测试并重启
sudo nginx -t
sudo systemctl restart nginx
```

## 配置 HTTPS（使用 Let's Encrypt）

```bash
# 安装 Certbot
sudo apt install -y certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
sudo certbot --nginx -d api.yourdomain.com

# 自动续期
sudo certbot renew --dry-run
```

## 防火墙配置

```bash
# Ubuntu (UFW)
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# CentOS (firewalld)
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --reload
```

## 常用命令

### Docker Compose
```bash
# 启动
docker-compose up -d

# 停止
docker-compose down

# 查看日志
docker-compose logs -f

# 重启
docker-compose restart

# 更新
git pull
docker-compose up -d --build
```

### Systemd (后端)
```bash
# 启动
sudo systemctl start dataclean-api

# 停止
sudo systemctl stop dataclean-api

# 重启
sudo systemctl restart dataclean-api

# 查看日志
sudo journalctl -u dataclean-api -f
```

### PM2 (前端)
```bash
# 启动
pm2 start ecosystem.config.js

# 停止
pm2 stop theta-frontend

# 重启
pm2 restart theta-frontend

# 查看日志
pm2 logs theta-frontend

# 查看状态
pm2 status
```

## 故障排查

1. **检查服务状态**
```bash
# 后端
sudo systemctl status dataclean-api
curl http://localhost:8001/health

# 前端
pm2 status
curl http://localhost:3000
```

2. **查看日志**
```bash
# 后端日志
sudo journalctl -u dataclean-api -n 50

# 前端日志
pm2 logs theta-frontend

# Nginx 日志
sudo tail -f /var/log/nginx/error.log
```

3. **检查端口占用**
```bash
sudo netstat -tlnp | grep -E '8001|3000'
```

## 安全建议

1. 使用 HTTPS
2. 配置防火墙
3. 定期更新系统
4. 使用强密码
5. 限制 SSH 访问
6. 定期备份数据

## 目录结构

```
/opt/
├── dataclean/          # 后端应用
│   ├── venv/          # Python 虚拟环境
│   ├── temp_uploads/  # 上传文件目录
│   └── logs/          # 日志目录
└── theta-frontend3/   # 前端应用
    ├── .next/         # Next.js 构建输出
    └── logs/          # 日志目录
```

## 更新部署

```bash
# 后端更新
cd /opt/dataclean
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart dataclean-api

# 前端更新
cd /opt/theta-frontend3
git pull
pnpm install
pnpm build
pm2 restart theta-frontend
```
