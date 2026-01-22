# 前端单独部署指南

本指南介绍如何单独部署 THETA 前端，无需等待后端完成。

---

## ⚠️ 重要提示：使用标准 Web 端口

**默认配置已改为标准 HTTP 端口 80**，但需要注意：

1. **使用 80 端口需要 root 权限**
   - Docker 容器需要以 root 用户运行，或
   - 使用 Nginx 反向代理（推荐）

2. **推荐方案：Nginx 反向代理**
   - 容器内部仍使用 3000 端口
   - Nginx 监听 80 端口，转发到容器的 3000 端口
   - 这样更安全，不需要 root 权限

3. **如果 80 端口被占用**
   - 修改 `.env.frontend` 中的 `FRONTEND_PORT=3000`
   - 或使用其他可用端口

---

## 📋 部署方案选择

### 方案一：Vercel 部署（推荐，最简单）
- ✅ 零配置，自动 HTTPS
- ✅ 全球 CDN 加速
- ✅ 自动部署
- ⏱️ 5 分钟完成

### 方案二：Docker 部署（服务器）
- ✅ 完全控制
- ✅ 可配合 Nginx
- ⏱️ 10 分钟完成

### 方案三：宝塔面板部署（服务器）
- ✅ 图形化界面
- ✅ 一键 SSL
- ⏱️ 15 分钟完成

---

## 🚀 方案一：Vercel 部署（推荐）

### 步骤 1: 准备代码

确保前端代码已推送到 Git 仓库（GitHub/GitLab/Bitbucket）。

### 步骤 2: 部署到 Vercel

1. **访问 Vercel**
   - 打开 [vercel.com](https://vercel.com)
   - 使用 GitHub/GitLab 账号登录

2. **导入项目**
   - 点击 "Add New Project"
   - 选择你的代码仓库
   - **重要**：设置 **Root Directory** 为 `theta-frontend3`

3. **配置项目**
   - Framework Preset: Next.js（自动检测）
   - Build Command: `npm run build`（默认）
   - Output Directory: `.next`（默认）

4. **设置环境变量**
   在 "Environment Variables" 中添加：
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-api.com
   NEXT_PUBLIC_DATACLEAN_API_URL=https://your-dataclean-api.com
   ```
   
   ⚠️ **如果后端未完成**，可以暂时设置为：
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   NEXT_PUBLIC_DATACLEAN_API_URL=http://localhost:8001
   ```
   或者使用 mock 服务地址。

5. **部署**
   - 点击 "Deploy"
   - 等待构建完成（2-5 分钟）

### 步骤 3: 访问前端

部署完成后，你会获得一个 Vercel 域名：
- `https://your-project.vercel.app`

### 步骤 4: 配置自定义域名（可选）

1. 在项目设置中点击 "Domains"
2. 添加你的域名
3. 按照提示配置 DNS 记录

---

## 🐳 方案二：Docker 单独部署前端

### 步骤 1: 创建前端专用 docker-compose.yml

在项目根目录创建 `docker-compose.frontend.yml`：

```yaml
version: '3.8'

services:
  theta-frontend:
    build:
      context: ./theta-frontend3
      dockerfile: Dockerfile
      args:
        - NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL:-http://localhost:8000}
        - NEXT_PUBLIC_DATACLEAN_API_URL=${NEXT_PUBLIC_DATACLEAN_API_URL:-http://localhost:8001}
    container_name: theta-frontend
    ports:
      - "${FRONTEND_PORT:-80}:3000"
    environment:
      - NODE_ENV=production
      - NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL:-http://localhost:8000}
      - NEXT_PUBLIC_DATACLEAN_API_URL=${NEXT_PUBLIC_DATACLEAN_API_URL:-http://localhost:8001}
    restart: unless-stopped
    networks:
      - theta-network

networks:
  theta-network:
    driver: bridge
```

### 步骤 2: 创建环境变量文件

创建 `.env.frontend` 文件：

```bash
# 前端端口（默认 80，标准 HTTP 端口）
# 如果 80 端口被占用或需要 root 权限，可以改为其他端口（如 3000）
FRONTEND_PORT=80

# 后端 API 地址（如果后端未完成，可以暂时留空或使用 mock 服务）
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_DATACLEAN_API_URL=http://localhost:8001

# 如果后端部署在其他服务器，设置为实际地址：
# NEXT_PUBLIC_API_URL=https://api.yourdomain.com
# NEXT_PUBLIC_DATACLEAN_API_URL=https://dataclean.yourdomain.com
```

### 步骤 3: 部署

#### 方法一：使用更新脚本（推荐，自动拉取代码）

```bash
# 进入项目目录
cd /path/to/THETA

# 使用更新部署脚本（会自动 git pull 并重新构建）
sudo chmod +x deploy-frontend-update.sh
sudo ./deploy-frontend-update.sh
```

#### 方法二：手动部署

```bash
# 进入项目目录
cd /path/to/THETA

# 如果使用 Git，先拉取最新代码
git pull

# 使用前端专用的 docker-compose 文件
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d --build

# 查看日志
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend logs -f

# 查看状态
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend ps
```

### 步骤 4: 配置 Nginx 反向代理（推荐）

**推荐使用 Nginx 反向代理**，这样容器可以使用非 root 用户运行，更安全。

```nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    location / {
        # 容器内部使用 3000 端口，Nginx 转发请求
        proxy_pass http://localhost:3000;
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

---

## 🎛️ 方案三：宝塔面板单独部署前端

### 步骤 1: 安装 Docker

在宝塔面板的 **软件商店** 中安装 **Docker 管理器**。

### 步骤 2: 上传项目文件

1. 打开 **文件** 管理器
2. 进入 `/www/wwwroot` 目录
3. 上传或克隆项目到 `THETA` 目录

### 步骤 3: 创建前端部署配置

在宝塔文件管理器中：

1. 进入 `/www/wwwroot/THETA` 目录
2. 创建 `.env.frontend` 文件：

```bash
# 使用标准 HTTP 端口 80
FRONTEND_PORT=80
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_DATACLEAN_API_URL=http://localhost:8001

# 如果 80 端口被占用，可以改为其他端口（如 3000）
# FRONTEND_PORT=3000
```

3. 创建 `docker-compose.frontend.yml`（参考方案二）

### 步骤 4: 启动服务

在宝塔 **终端** 中执行：

```bash
cd /www/wwwroot/THETA
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d --build
```

### 步骤 5: 配置网站和反向代理

1. 在 **网站** 中添加站点
2. 配置反向代理：
   - 如果容器映射到 80 端口：指向 `http://127.0.0.1:80`
   - 如果容器映射到 3000 端口：指向 `http://127.0.0.1:3000`
3. 申请 SSL 证书

详细步骤请参考 `BT_PANEL_DEPLOY.md`。

---

## 🛠️ 方案四：PM2 部署（不使用 Docker）

### 步骤 1: 安装 Node.js 和 pnpm

```bash
# 安装 Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# 安装 pnpm
npm install -g pnpm
```

### 步骤 2: 构建前端

```bash
cd /path/to/THETA/theta-frontend3

# 安装依赖
pnpm install

# 创建生产环境变量文件
cat > .env.production << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_DATACLEAN_API_URL=http://localhost:8001
EOF

# 构建
pnpm build
```

### 步骤 3: 使用 PM2 启动

```bash
# 安装 PM2
npm install -g pm2

# 启动应用
pm2 start npm --name "theta-frontend" -- start

# 保存配置
pm2 save

# 设置开机自启
pm2 startup
```

### 步骤 4: 配置 Nginx

参考方案二的 Nginx 配置。

---

## 🔧 后端未完成时的处理方案

### 方案 A: 使用 Mock 数据

前端可以配置为使用 mock 服务或本地 mock 数据。

1. **创建 mock API 服务**（可选）
   - 使用 [JSON Server](https://github.com/typicode/json-server)
   - 或使用 [Mock Service Worker](https://mswjs.io/)

2. **修改前端 API 客户端**
   在 `theta-frontend3/lib/api/etm-agent.ts` 中添加 mock 模式：

```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK === 'true';

// Mock 数据示例
const mockTasks: TaskResponse[] = [
  {
    task_id: 'mock-task-1',
    status: 'completed',
    progress: 100,
    dataset: 'sample-dataset',
    mode: 'zero_shot',
    num_topics: 20,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  }
];

export const ETMAgentAPI = {
  async getTasks(): Promise<TaskResponse[]> {
    if (USE_MOCK) {
      return mockTasks;
    }
    // ... 实际 API 调用
  },
  // ... 其他方法
};
```

### 方案 B: 暂时禁用后端功能

在前端中添加功能开关：

```typescript
// 在配置文件中
export const FEATURES = {
  ENABLE_BACKEND: process.env.NEXT_PUBLIC_ENABLE_BACKEND !== 'false',
  ENABLE_DATACLEAN: process.env.NEXT_PUBLIC_ENABLE_DATACLEAN !== 'false',
};
```

### 方案 C: 显示"后端开发中"提示

在需要后端 API 的页面显示友好提示：

```tsx
{!backendAvailable && (
  <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
    <p className="text-yellow-800">
      ⚠️ 后端服务正在开发中，部分功能暂时不可用。
    </p>
  </div>
)}
```

---

## 📝 环境变量说明

### 必需的环境变量

| 变量名 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `NEXT_PUBLIC_API_URL` | ETM Agent API 地址 | `http://localhost:8000` | `https://api.example.com` |
| `NEXT_PUBLIC_DATACLEAN_API_URL` | DataClean API 地址 | `http://localhost:8001` | `https://dataclean.example.com` |

### 可选的环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `NEXT_PUBLIC_USE_MOCK` | 是否使用 Mock 数据 | `false` |
| `NEXT_PUBLIC_ENABLE_BACKEND` | 是否启用后端功能 | `true` |

---

## ✅ 部署检查清单

- [ ] 前端代码已推送到 Git 仓库（Vercel 部署）
- [ ] 环境变量已正确配置
- [ ] 构建成功（`npm run build` 无错误）
- [ ] 服务已启动并运行
- [ ] 可以访问前端页面
- [ ] Nginx 反向代理已配置（如果使用）
- [ ] SSL 证书已配置（如果使用域名）

---

## 🐛 常见问题

### 问题 1: 构建失败

**解决方法**：
```bash
# 清理缓存
rm -rf node_modules .next
pnpm install
pnpm build
```

### 问题 2: 前端无法连接后端（后端未完成）

**解决方法**：
- 暂时设置环境变量为 mock 地址
- 或在前端显示"后端开发中"提示
- 或使用 JSON Server 提供 mock API

### 问题 3: 端口被占用

**解决方法**：
```bash
# 检查端口占用
sudo netstat -tlnp | grep 80

# 修改端口
# 在 .env.frontend 中设置 FRONTEND_PORT=3000（如果 80 端口被占用）
```

---

## 🔄 更新部署（重要！）

### 当代码更新后，需要重新部署

**完整更新流程**：

```bash
# 1. 进入项目目录
cd /path/to/THETA

# 2. 拉取最新代码（如果使用 Git）
git pull

# 3. 停止现有容器
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend down

# 4. 重新构建并启动（--build 会重新构建镜像）
docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d --build
```

**或者使用更新脚本（自动完成以上步骤）**：

```bash
cd /path/to/THETA
sudo chmod +x deploy-frontend-update.sh
sudo ./deploy-frontend-update.sh
```

### 后端完成后如何连接

当后端开发完成后：

1. **更新环境变量**
   ```bash
   # 编辑 .env.frontend
   nano .env.frontend
   
   # 修改为实际后端地址
   NEXT_PUBLIC_API_URL=https://api.yourdomain.com
   NEXT_PUBLIC_DATACLEAN_API_URL=https://dataclean.yourdomain.com
   ```

2. **重新构建和部署**
   ```bash
   # 停止容器
   docker-compose -f docker-compose.frontend.yml --env-file .env.frontend down
   
   # 重新构建（环境变量变化需要重新构建）
   docker-compose -f docker-compose.frontend.yml --env-file .env.frontend up -d --build
   ```

3. **测试连接**
   - 访问前端页面
   - 测试 API 调用
   - 检查浏览器控制台是否有错误

---

## 📚 相关文档

- `VERCEL_DEPLOY.md` - Vercel 详细部署指南
- `BT_PANEL_DEPLOY.md` - 宝塔面板部署指南
- `SERVER_DEPLOY_GUIDE.md` - 完整服务器部署指南

---

**部署完成后，即使后端未完成，前端也可以正常展示界面和 UI 交互！** 🎉
