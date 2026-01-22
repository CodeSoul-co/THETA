# 解决端口 80 被占用的问题

## 问题说明

错误信息：`failed to bind host port 0.0.0.0:80/tcp: address already in use`

这表示端口 80 已经被其他服务占用（通常是 Nginx、Apache 或其他 Web 服务器）。

---

## 🔍 检查端口占用

### 方法一：查看占用端口的进程

```bash
# 使用 netstat
sudo netstat -tlnp | grep :80

# 或使用 ss
sudo ss -tlnp | grep :80

# 或使用 lsof
sudo lsof -i :80
```

### 方法二：查看 Nginx 是否运行

```bash
# 检查 Nginx 状态
sudo systemctl status nginx

# 查看 Nginx 进程
ps aux | grep nginx
```

---

## ✅ 解决方案

### 方案一：使用 Nginx 反向代理（推荐）

这是最佳实践，容器使用 3000 端口，Nginx 监听 80 端口。

#### 1. 修改环境变量

```bash
# 编辑 .env.frontend
nano .env.frontend

# 修改为：
FRONTEND_PORT=3000  # 容器内部端口
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_DATACLEAN_API_URL=http://localhost:8001
```

#### 2. 配置 Nginx 反向代理

```bash
# 创建或编辑 Nginx 配置
sudo nano /etc/nginx/sites-available/theta-frontend
```

添加以下配置：

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:3000;  # 转发到容器的 3000 端口
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

#### 3. 重新部署

```bash
cd /www/wwwroot/theta.code-soul.com
sudo ./deploy-frontend-update.sh
```

---

### 方案二：停止占用 80 端口的服务

如果不需要使用 Nginx，可以停止它：

```bash
# 停止 Nginx
sudo systemctl stop nginx

# 禁用开机自启（可选）
sudo systemctl disable nginx

# 然后重新部署
sudo ./deploy-frontend-update.sh
```

---

### 方案三：修改容器端口

如果不想使用 Nginx，可以改用其他端口：

```bash
# 编辑 .env.frontend
nano .env.frontend

# 修改为：
FRONTEND_PORT=3000  # 或其他可用端口，如 8080, 3001 等

# 重新部署
sudo ./deploy-frontend-update.sh
```

然后访问：`http://your-server-ip:3000`

---

## 🎯 推荐配置（Nginx + Docker）

### 完整配置示例

**`.env.frontend`**:
```bash
FRONTEND_PORT=3000
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_DATACLEAN_API_URL=http://localhost:8001
```

**Nginx 配置** (`/etc/nginx/sites-available/theta-frontend`):
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
    }
}
```

**优势**：
- ✅ 容器不需要 root 权限
- ✅ 可以使用标准 80 端口
- ✅ 可以配置 SSL/HTTPS
- ✅ 更安全、更灵活

---

## 🔧 快速修复命令

### 如果使用 Nginx 反向代理：

```bash
# 1. 修改端口为 3000
cd /www/wwwroot/theta.code-soul.com
sed -i 's/FRONTEND_PORT=80/FRONTEND_PORT=3000/' .env.frontend

# 2. 配置 Nginx（如果还没有）
sudo nano /etc/nginx/sites-available/theta-frontend
# 添加上面的 Nginx 配置

# 3. 启用并重启 Nginx
sudo ln -s /etc/nginx/sites-available/theta-frontend /etc/nginx/sites-enabled/ 2>/dev/null || true
sudo nginx -t
sudo systemctl restart nginx

# 4. 重新部署
sudo ./deploy-frontend-update.sh
```

### 如果直接使用端口 3000：

```bash
# 修改端口
cd /www/wwwroot/theta.code-soul.com
sed -i 's/FRONTEND_PORT=80/FRONTEND_PORT=3000/' .env.frontend

# 重新部署
sudo ./deploy-frontend-update.sh
```

---

## 📝 验证

部署完成后验证：

```bash
# 检查容器状态
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend ps

# 测试访问
curl http://localhost:3000  # 如果使用 3000 端口
# 或
curl http://localhost:80    # 如果使用 Nginx 反向代理

# 查看日志
sudo docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs -f
```

---

## 💡 建议

**推荐使用方案一（Nginx 反向代理）**，因为：
1. 更安全（容器不需要 root 权限）
2. 更灵活（可以配置多个站点）
3. 更容易配置 SSL/HTTPS
4. 符合生产环境最佳实践

---

**修复后，重新运行部署脚本即可！** 🚀
