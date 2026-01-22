# 宝塔面板部署 THETA 项目指南

本指南详细介绍如何在宝塔面板中部署 THETA 项目。

---

## 📋 前置要求

- 已安装宝塔面板的 Linux 服务器（Ubuntu/Debian/CentOS）
- 宝塔面板版本 7.0+
- Root 权限

---

## 🔧 步骤 1: 安装和配置宝塔面板

### 1.1 安装宝塔面板

如果还没有安装，执行以下命令：

```bash
# CentOS
yum install -y wget && wget -O install.sh http://download.bt.cn/install/install_6.0.sh && sh install.sh

# Ubuntu/Debian
wget -O install.sh http://download.bt.cn/install/install-ubuntu_6.0.sh && sudo bash install.sh
```

安装完成后，会显示面板地址、用户名和密码，请妥善保存。

### 1.2 登录宝塔面板

访问显示的地址（通常是 `http://your-server-ip:8888`），使用提供的用户名和密码登录。

### 1.3 安装必要软件

在宝塔面板中，点击 **软件商店**，安装以下软件：

- ✅ **Docker 管理器**（或 Docker）
- ✅ **Nginx**（用于反向代理）
- ✅ **PM2 管理器**（可选，用于 Node.js 进程管理）

---

## 🐳 步骤 2: 安装 Docker（如果未安装）

### 方法一：通过宝塔面板安装

1. 打开 **软件商店**
2. 搜索 "Docker" 或 "Docker 管理器"
3. 点击 **安装**

### 方法二：通过终端安装

在宝塔面板中打开 **终端**，执行：

```bash
# 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 验证安装
docker --version
docker-compose --version
```

---

## 📁 步骤 3: 部署项目文件

### 3.1 使用宝塔文件管理器

1. 打开 **文件** 管理器
2. 进入 `/www/wwwroot` 目录（或你喜欢的目录）
3. 点击 **上传** 或使用 **终端** 克隆项目

### 3.2 使用终端克隆项目

在宝塔面板的 **终端** 中执行：

```bash
# 进入网站目录
cd /www/wwwroot

# 克隆项目（替换为你的仓库地址）
git clone https://github.com/your-username/THETA.git

# 进入项目目录
cd THETA
```

### 3.3 使用宝塔文件管理器上传

如果项目在本地，可以：
1. 在本地打包项目：`zip -r THETA.zip THETA/`
2. 在宝塔文件管理器中上传 `THETA.zip`
3. 解压到 `/www/wwwroot/THETA`

---

## ⚙️ 步骤 4: 配置环境变量

### 4.1 创建 .env 文件

在宝塔文件管理器中：

1. 进入 `/www/wwwroot/THETA` 目录
2. 找到 `docker.env.template` 文件
3. 复制并重命名为 `.env`
4. 编辑 `.env` 文件

### 4.2 配置环境变量

在宝塔文件管理器中双击 `.env` 文件进行编辑，或使用终端：

```bash
cd /www/wwwroot/THETA
cp docker.env.template .env
nano .env
```

**重要配置项**：

```bash
# ========== 端口配置 ==========
BACKEND_PORT=8000
API_PORT=8001
FRONTEND_PORT=3000

# ========== CORS 配置 ==========
# 替换为你的实际域名
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# ========== 前端环境变量 ==========
# 如果使用域名，设置为：
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_DATACLEAN_API_URL=https://dataclean.yourdomain.com

# 如果直接使用 IP，设置为：
# NEXT_PUBLIC_API_URL=http://your-server-ip:8000
# NEXT_PUBLIC_DATACLEAN_API_URL=http://your-server-ip:8001

# ========== 后端配置 ==========
QWEN_API_KEY=your-qwen-api-key
SECRET_KEY=your-random-secret-key-here
DATABASE_URL=sqlite:///./users.db
```

---

## 🚀 步骤 5: 启动 Docker 服务

### 方法一：使用宝塔 Docker 管理器

1. 打开 **Docker 管理器**
2. 点击 **Compose** 标签
3. 点击 **创建项目**
4. 选择项目目录：`/www/wwwroot/THETA`
5. 选择 `docker-compose.yml` 文件
6. 点击 **启动**

### 方法二：使用终端

在宝塔面板的 **终端** 中执行：

```bash
cd /www/wwwroot/THETA

# 构建并启动服务
docker-compose up -d --build

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 5.1 验证服务运行

在终端中执行：

```bash
# 检查容器状态
docker ps

# 测试 API
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:3000
```

---

## 🌐 步骤 6: 配置 Nginx 反向代理

### 6.1 添加网站

1. 打开 **网站** → **添加站点**
2. 填写域名（如 `yourdomain.com`）
3. 选择 **纯静态** 或 **PHP 项目**（不影响，我们只用反向代理）
4. 点击 **提交**

### 6.2 配置前端反向代理

1. 点击网站右侧的 **设置**
2. 选择 **反向代理** 标签
3. 点击 **添加反向代理**
4. 配置如下：

```
代理名称: theta-frontend
目标URL: http://127.0.0.1:3000
发送域名: $host
```

5. 点击 **高级**，添加以下配置：

```nginx
# WebSocket 支持
proxy_http_version 1.1;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
```

6. 点击 **提交**

### 6.3 配置 API 反向代理

#### 6.3.1 ETM Agent API

1. 添加新网站：`api.yourdomain.com`
2. 在网站设置中配置反向代理：
   - 目标URL: `http://127.0.0.1:8000`
   - 其他配置同上

#### 6.3.2 DataClean API

1. 添加新网站：`dataclean.yourdomain.com`
2. 在网站设置中配置反向代理：
   - 目标URL: `http://127.0.0.1:8001`
   - 其他配置同上

### 6.4 手动编辑 Nginx 配置（可选）

如果需要更精细的控制，可以：

1. 在网站设置中点击 **配置文件**
2. 编辑 Nginx 配置

**前端配置示例**：

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

**API 配置示例**：

```nginx
# ETM Agent API
server {
    listen 80;
    server_name api.yourdomain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
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
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

3. 点击 **保存**，然后 **重载配置**

---

## 🔒 步骤 7: 配置 SSL 证书（HTTPS）

### 7.1 使用宝塔面板一键申请

1. 在网站设置中，点击 **SSL** 标签
2. 选择 **Let's Encrypt**
3. 勾选需要申请证书的域名
4. 点击 **申请**
5. 等待申请完成（通常几秒钟）
6. 开启 **强制 HTTPS**

### 7.2 为所有域名申请证书

为以下域名分别申请证书：
- `yourdomain.com` 和 `www.yourdomain.com`
- `api.yourdomain.com`
- `dataclean.yourdomain.com`

### 7.3 更新环境变量

申请 SSL 后，更新 `.env` 文件：

```bash
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_DATACLEAN_API_URL=https://dataclean.yourdomain.com
```

然后重启 Docker 服务：

```bash
cd /www/wwwroot/THETA
docker-compose down
docker-compose up -d --build
```

---

## 🔥 步骤 8: 配置防火墙

### 8.1 在宝塔面板中配置

1. 打开 **安全** 设置
2. 在 **系统防火墙** 中：
   - 开放端口 `3000`（前端，如果直接访问）
   - 开放端口 `8000`（ETM Agent API，如果直接访问）
   - 开放端口 `8001`（DataClean API，如果直接访问）
   - 开放端口 `80`（HTTP）
   - 开放端口 `443`（HTTPS）

### 8.2 如果使用 Nginx 反向代理

如果所有服务都通过 Nginx 反向代理访问，**不需要**开放 `3000`、`8000`、`8001` 端口，只需要开放 `80` 和 `443`。

---

## 📊 步骤 9: 监控和管理

### 9.1 使用宝塔 Docker 管理器

1. 打开 **Docker 管理器**
2. 查看容器列表和状态
3. 可以执行启动、停止、重启、查看日志等操作

### 9.2 查看日志

**方法一：宝塔面板**
- 在 Docker 管理器中点击容器，查看日志

**方法二：终端**
```bash
cd /www/wwwroot/THETA

# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f theta-frontend
docker-compose logs -f etm-agent-api
docker-compose logs -f dataclean-api
```

### 9.3 设置定时任务（可选）

如果需要定期备份或更新：

1. 打开 **计划任务**
2. 添加 **Shell 脚本** 任务
3. 设置执行周期
4. 添加脚本：

```bash
#!/bin/bash
# 备份 Docker 数据卷
cd /www/wwwroot/THETA
docker-compose exec etm-agent-api tar czf /tmp/backup-$(date +%Y%m%d).tar.gz /app/data /app/result
```

---

## 🔄 步骤 10: 更新项目

### 10.1 使用宝塔文件管理器

1. 在文件管理器中进入项目目录
2. 如果使用 Git，可以在终端执行：

```bash
cd /www/wwwroot/THETA
git pull
docker-compose down
docker-compose up -d --build
```

### 10.2 使用宝塔终端

在宝塔面板的终端中执行更新命令。

---

## 🐛 故障排查

### 问题 1: 容器无法启动

**检查方法**：
```bash
cd /www/wwwroot/THETA
docker-compose logs [service-name]
```

**常见原因**：
- 端口被占用
- 环境变量配置错误
- 磁盘空间不足

### 问题 2: 前端无法访问后端

**检查清单**：
1. ✅ 检查 `.env` 文件中的 `NEXT_PUBLIC_API_URL` 和 `NEXT_PUBLIC_DATACLEAN_API_URL`
2. ✅ 检查后端服务是否运行：`docker-compose ps`
3. ✅ 检查 Nginx 反向代理配置
4. ✅ 检查 CORS 配置：`ALLOWED_ORIGINS`
5. ✅ 查看浏览器控制台错误信息

### 问题 3: SSL 证书申请失败

**解决方法**：
1. 确保域名已正确解析到服务器 IP
2. 确保端口 80 已开放
3. 检查是否有其他服务占用 80 端口
4. 等待 DNS 解析生效（可能需要几分钟）

### 问题 4: Nginx 配置错误

**检查方法**：
1. 在网站设置中点击 **测试配置**
2. 查看错误信息
3. 检查配置文件语法

---

## 📋 宝塔面板部署检查清单

- [ ] 宝塔面板已安装并可以访问
- [ ] Docker 和 Docker Compose 已安装
- [ ] 项目文件已上传到服务器
- [ ] `.env` 文件已配置
- [ ] Docker 服务已启动
- [ ] 网站已添加并配置反向代理
- [ ] SSL 证书已申请并启用
- [ ] 防火墙端口已开放
- [ ] 功能测试通过

---

## 🎯 快速部署命令总结

在宝塔面板终端中执行：

```bash
# 1. 进入项目目录
cd /www/wwwroot/THETA

# 2. 配置环境变量（如果还没有）
cp docker.env.template .env
nano .env  # 编辑配置

# 3. 启动服务
docker-compose up -d --build

# 4. 查看状态
docker-compose ps

# 5. 查看日志
docker-compose logs -f
```

---

## 📚 相关文档

- `QUICK_DEPLOY.md` - 快速部署指南
- `SERVER_DEPLOY_GUIDE.md` - 完整服务器部署指南
- `DOCKER_DEPLOY.md` - Docker 部署详细说明

---

## 💡 宝塔面板使用技巧

### 1. 文件管理
- 使用文件管理器可以方便地上传、编辑、删除文件
- 支持在线编辑代码文件
- 支持压缩和解压文件

### 2. 数据库管理
- 如果需要使用 MySQL/PostgreSQL，可以在软件商店安装
- 使用 phpMyAdmin 管理数据库

### 3. 备份管理
- 使用宝塔的备份功能定期备份项目文件
- 可以设置自动备份到云存储

### 4. 监控面板
- 查看服务器资源使用情况
- 监控网站访问量
- 查看系统日志

---

**部署完成后，访问你的域名即可使用 THETA 项目！** 🎉

如有问题，请查看日志或联系技术支持。
