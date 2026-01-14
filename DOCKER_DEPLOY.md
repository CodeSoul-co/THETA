# Docker 部署指南

本指南介绍如何使用 Docker 在服务器上部署 THETA 项目。

## 前置要求

- Ubuntu 20.04+ / CentOS 7+ / Debian 11+ 服务器
- Root 或 sudo 权限
- 至少 2GB RAM
- 至少 10GB 磁盘空间

## 快速开始

### 1. 安装 Docker 和 Docker Compose

#### Ubuntu/Debian
```bash
# 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 启动 Docker 服务
sudo systemctl start docker
sudo systemctl enable docker

# 验证安装
docker --version
docker-compose --version
```

#### CentOS/RHEL
```bash
# 安装 Docker
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 验证安装
docker --version
docker-compose --version
```

### 2. 克隆项目

```bash
cd /opt
sudo git clone https://github.com/CodeSoul-co/THETA.git
cd THETA
sudo git checkout frontend-3
```

### 3. 配置环境变量

```bash
# 复制环境变量模板
sudo cp .env.example .env

# 编辑环境变量（根据实际情况修改）
sudo nano .env
```

**重要配置项：**
- `API_PORT`: 后端 API 端口（默认 8001）
- `FRONTEND_PORT`: 前端端口（默认 3000）
- `ALLOWED_ORIGINS`: CORS 允许的源（生产环境设置为实际域名）
- `NEXT_PUBLIC_DATACLEAN_API_URL`: 前端访问后端的 URL

### 4. 一键部署

```bash
# 添加执行权限
sudo chmod +x docker-deploy.sh

# 运行部署脚本
sudo ./docker-deploy.sh
```

脚本会自动：
- 检查 Docker 环境
- 创建必要的目录
- 构建 Docker 镜像
- 启动所有服务
- 进行健康检查

### 5. 验证部署

```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 测试 API
curl http://localhost:8001/health

# 测试前端
curl http://localhost:3000
```

## 手动部署步骤

如果不使用自动脚本，可以手动执行：

```bash
# 1. 创建目录
mkdir -p ETM/dataclean/temp_uploads ETM/dataclean/temp_processed

# 2. 构建镜像
docker-compose build

# 3. 启动服务
docker-compose up -d

# 4. 查看日志
docker-compose logs -f
```

## 配置 Nginx 反向代理（可选）

### 安装 Nginx

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y nginx

# CentOS/RHEL
sudo yum install -y nginx
```

### 配置前端反向代理

```bash
# 复制配置模板
sudo cp theta-frontend3/nginx-frontend.conf.example /etc/nginx/sites-available/theta-frontend

# 编辑配置（修改域名）
sudo nano /etc/nginx/sites-available/theta-frontend

# 创建软链接
sudo ln -s /etc/nginx/sites-available/theta-frontend /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重启 Nginx
sudo systemctl restart nginx
```

### 配置后端 API 反向代理

```bash
# 复制配置模板
sudo cp ETM/dataclean/nginx.conf.example /etc/nginx/sites-available/dataclean-api

# 编辑配置（修改域名）
sudo nano /etc/nginx/sites-available/dataclean-api

# 创建软链接
sudo ln -s /etc/nginx/sites-available/dataclean-api /etc/nginx/sites-enabled/

# 测试并重启
sudo nginx -t
sudo systemctl restart nginx
```

**重要：** 使用 Nginx 反向代理时，需要修改 `.env` 文件中的 `NEXT_PUBLIC_DATACLEAN_API_URL` 为实际的 API 域名。

## 配置 HTTPS（使用 Let's Encrypt）

```bash
# 安装 Certbot
sudo apt install -y certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
sudo certbot --nginx -d api.yourdomain.com

# 测试自动续期
sudo certbot renew --dry-run
```

## 常用命令

### 服务管理

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f dataclean-api
docker-compose logs -f theta-frontend
```

### 更新部署

```bash
# 拉取最新代码
git pull

# 重新构建并启动
docker-compose up -d --build

# 清理未使用的镜像
docker system prune -a
```

### 数据管理

```bash
# 查看数据卷
docker volume ls

# 备份数据
docker run --rm -v theta_temp_uploads:/data -v $(pwd):/backup alpine tar czf /backup/uploads-backup.tar.gz /data

# 恢复数据
docker run --rm -v theta_temp_uploads:/data -v $(pwd):/backup alpine tar xzf /backup/uploads-backup.tar.gz -C /
```

## 故障排查

### 1. 容器无法启动

```bash
# 查看详细日志
docker-compose logs [service-name]

# 检查容器状态
docker-compose ps

# 进入容器调试
docker-compose exec [service-name] sh
```

### 2. 端口冲突

```bash
# 检查端口占用
sudo netstat -tlnp | grep -E '8001|3000'

# 修改 .env 文件中的端口配置
nano .env
```

### 3. 构建失败

```bash
# 清理构建缓存
docker-compose build --no-cache

# 查看构建日志
docker-compose build
```

### 4. 网络问题

```bash
# 检查网络
docker network ls
docker network inspect theta_theta-network

# 测试容器间通信
docker-compose exec theta-frontend ping dataclean-api
```

### 5. 权限问题

```bash
# 修复目录权限
sudo chown -R $USER:$USER ETM/dataclean/temp_uploads
sudo chmod -R 755 ETM/dataclean/temp_uploads
```

## 性能优化

### 1. 资源限制

在 `docker-compose.yml` 中添加资源限制：

```yaml
services:
  dataclean-api:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

### 2. 日志管理

```bash
# 限制日志大小（在 docker-compose.yml 中）
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

## 安全建议

1. **防火墙配置**
```bash
# Ubuntu (UFW)
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

2. **使用非 root 用户运行 Docker**
```bash
sudo usermod -aG docker $USER
# 重新登录使配置生效
```

3. **定期更新镜像**
```bash
docker-compose pull
docker-compose up -d
```

4. **备份数据**
定期备份数据卷和配置文件

## 监控和维护

### 查看资源使用

```bash
docker stats
```

### 查看容器日志

```bash
# 实时日志
docker-compose logs -f

# 最近 100 行
docker-compose logs --tail=100
```

### 健康检查

```bash
# API 健康检查
curl http://localhost:8001/health

# 前端检查
curl http://localhost:3000
```

## 目录结构

```
/opt/THETA/
├── docker-compose.yml      # Docker Compose 配置
├── .env                     # 环境变量（需要创建）
├── .env.example            # 环境变量模板
├── docker-deploy.sh        # 一键部署脚本
├── ETM/dataclean/          # 后端代码
│   ├── Dockerfile
│   └── ...
└── theta-frontend3/        # 前端代码
    ├── Dockerfile
    └── ...
```

## 访问地址

部署成功后：
- **前端**: http://your-server-ip:3000
- **后端 API**: http://your-server-ip:8001
- **API 健康检查**: http://your-server-ip:8001/health

配置 Nginx 后：
- **前端**: https://yourdomain.com
- **后端 API**: https://api.yourdomain.com
