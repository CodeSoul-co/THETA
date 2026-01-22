# 前端更新部署快速指南

当代码更新后，需要重新部署前端才能生效。

---

## 🚀 快速更新（推荐）

### 使用更新脚本（一键完成）

```bash
cd /path/to/THETA
sudo chmod +x deploy-frontend-update.sh
sudo ./deploy-frontend-update.sh
```

这个脚本会自动：
1. ✅ 拉取最新代码（git pull）
2. ✅ 停止现有容器
3. ✅ 重新构建镜像
4. ✅ 启动新容器

---

## 📝 手动更新步骤

### 步骤 1: 拉取最新代码

```bash
cd /path/to/THETA

# 如果使用 Git
git pull

# 或者从远程仓库拉取
git fetch origin
git pull origin main  # 或 master，根据你的分支名
```

### 步骤 2: 停止现有容器

```bash
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend down
```

### 步骤 3: 重新构建并启动

```bash
# 重新构建镜像（重要：--build 参数）
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d --build
```

### 步骤 4: 验证部署

```bash
# 查看容器状态
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend ps

# 查看日志
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs -f

# 测试访问
curl http://localhost:80  # 或你的端口
```

---

## ⚠️ 重要提示

### 1. 为什么需要重新构建？

- **代码更新**：新的前端代码需要重新构建到 Docker 镜像中
- **环境变量变化**：如果 `.env.frontend` 中的 `NEXT_PUBLIC_*` 变量变化，需要重新构建
- **依赖更新**：如果 `package.json` 变化，需要重新安装依赖

### 2. 什么时候需要重新构建？

- ✅ 代码更新（git pull 后）
- ✅ 环境变量 `NEXT_PUBLIC_*` 变化
- ✅ `package.json` 或依赖变化
- ✅ `next.config.mjs` 配置变化

### 3. 什么时候只需要重启？

- ✅ 仅修改了 `.env.frontend` 中的非 `NEXT_PUBLIC_*` 变量（如 `FRONTEND_PORT`）
- ✅ 仅需要重启服务

```bash
# 仅重启，不重新构建
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend restart
```

---

## 🔍 检查更新是否生效

### 方法 1: 查看构建时间

```bash
# 查看镜像构建时间
docker images | grep theta-frontend

# 查看容器启动时间
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}" | grep theta-frontend
```

### 方法 2: 查看代码版本

在前端页面中检查：
- 查看页面源代码
- 检查浏览器控制台
- 查看网络请求

### 方法 3: 查看日志

```bash
# 查看最新日志
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs --tail=50

# 实时查看日志
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs -f
```

---

## 🐛 更新后没有生效？

### 检查清单

1. **确认代码已更新**
   ```bash
   git log -1  # 查看最新提交
   git status  # 确认没有未提交的更改
   ```

2. **确认重新构建了镜像**
   ```bash
   # 必须使用 --build 参数
   docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d --build
   ```

3. **确认容器已重启**
   ```bash
   docker-compose -f docker-compose.frontend.yml --env-file .env.frontend ps
   # 检查 STATUS 列，应该是 "Up X seconds"
   ```

4. **清理缓存**
   ```bash
   # 清理 Docker 构建缓存
   docker-compose -f docker-compose.frontend.yml --env-file .env.frontend build --no-cache
   
   # 清理浏览器缓存
   # 在浏览器中按 Ctrl+Shift+R (Windows) 或 Cmd+Shift+R (Mac)
   ```

5. **检查端口是否正确**
   ```bash
   # 检查端口映射
   docker port theta-frontend
   
   # 检查端口是否被占用
   sudo netstat -tlnp | grep 80
   ```

---

## 📋 完整更新命令总结

```bash
# 进入项目目录
cd /path/to/THETA

# 拉取代码
git pull

# 停止容器
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend down

# 重新构建并启动
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d --build

# 查看日志确认
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs -f
```

---

## 💡 提示

- 使用更新脚本 `deploy-frontend-update.sh` 可以自动完成所有步骤
- 如果使用宝塔面板，可以在终端中执行这些命令
- 如果使用 Vercel，代码推送后会自动重新部署

---

**记住：代码更新后，必须重新构建 Docker 镜像才能生效！** 🔄
