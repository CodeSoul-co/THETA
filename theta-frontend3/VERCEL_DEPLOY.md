# Vercel 部署指南

本文档说明如何将 THETA 前端项目部署到 Vercel。

## 📋 前置要求

1. **Vercel 账号**: 访问 [vercel.com](https://vercel.com) 注册账号
2. **GitHub/GitLab/Bitbucket 账号**: 用于连接代码仓库
3. **后端服务已部署**: 确保后端 API 服务已部署并可访问

## 🚀 快速部署

### 方法一：通过 Vercel Dashboard（推荐）

1. **登录 Vercel**
   - 访问 [vercel.com](https://vercel.com)
   - 使用 GitHub/GitLab/Bitbucket 账号登录

2. **导入项目**
   - 点击 "Add New Project"
   - 选择你的代码仓库（THETA）
   - 选择根目录：`theta-frontend3`

3. **配置项目**
   - **Framework Preset**: Next.js（自动检测）
   - **Root Directory**: `theta-frontend3`
   - **Build Command**: `npm run build`（默认）
   - **Output Directory**: `.next`（默认）
   - **Install Command**: `npm install`（默认）

4. **设置环境变量**
   在 "Environment Variables" 中添加：
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-api.com
   NEXT_PUBLIC_DATACLEAN_API_URL=https://your-dataclean-api.com
   ```
   ⚠️ **重要**: 将 `your-backend-api.com` 和 `your-dataclean-api.com` 替换为你的实际后端 API 地址

5. **部署**
   - 点击 "Deploy"
   - 等待构建完成（通常 2-5 分钟）

### 方法二：通过 Vercel CLI

1. **安装 Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **登录 Vercel**
   ```bash
   vercel login
   ```

3. **进入前端目录**
   ```bash
   cd theta-frontend3
   ```

4. **部署**
   ```bash
   vercel
   ```
   
   首次部署会提示：
   - Set up and deploy? → `Y`
   - Which scope? → 选择你的账号
   - Link to existing project? → `N`（首次部署）
   - Project name? → `theta-frontend`（或自定义）
   - Directory? → `./`
   - Override settings? → `N`

5. **设置环境变量**
   ```bash
   vercel env add NEXT_PUBLIC_API_URL
   # 输入: https://your-backend-api.com
   
   vercel env add NEXT_PUBLIC_DATACLEAN_API_URL
   # 输入: https://your-dataclean-api.com
   ```

6. **重新部署以应用环境变量**
   ```bash
   vercel --prod
   ```

## 🔧 环境变量配置

### 必需的环境变量

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `NEXT_PUBLIC_API_URL` | 后端 ETM Agent API 地址 | `https://api.example.com` |
| `NEXT_PUBLIC_DATACLEAN_API_URL` | 数据清洗 API 地址 | `https://dataclean.example.com` |

### 在 Vercel Dashboard 中设置

1. 进入项目设置 → Environment Variables
2. 添加每个环境变量：
   - **Key**: `NEXT_PUBLIC_API_URL`
   - **Value**: 你的后端 API URL
   - **Environment**: Production, Preview, Development（全选）

### 使用 Vercel CLI 设置

```bash
# 生产环境
vercel env add NEXT_PUBLIC_API_URL production

# 预览环境
vercel env add NEXT_PUBLIC_API_URL preview

# 开发环境
vercel env add NEXT_PUBLIC_API_URL development
```

## 🌐 自定义域名

1. **添加域名**
   - 进入项目设置 → Domains
   - 输入你的域名（如 `app.example.com`）
   - 按照提示配置 DNS 记录

2. **DNS 配置**
   - 添加 CNAME 记录：
     ```
     类型: CNAME
     名称: app（或 @）
     值: cname.vercel-dns.com
     ```

## 🔄 持续部署

Vercel 会自动：
- 监听 Git 推送
- 在每次 push 到主分支时自动部署
- 为每个 Pull Request 创建预览部署

### 手动触发部署

```bash
vercel --prod
```

## 📝 配置文件说明

### `vercel.json`

项目根目录的 `vercel.json` 包含：
- 构建配置
- 环境变量引用
- 路由重写规则
- 安全头设置

### `next.config.mjs`

Next.js 配置文件，已优化适配 Vercel。

## 🐛 常见问题

### 1. 构建失败

**问题**: 构建时出现错误

**解决**:
- 检查 `package.json` 中的依赖是否正确
- 查看构建日志中的具体错误信息
- 确保 Node.js 版本兼容（Vercel 默认使用 Node.js 18+）

### 2. API 请求失败（CORS 错误）

**问题**: 前端无法访问后端 API

**解决**:
- 确保后端 API 已配置 CORS，允许 Vercel 域名访问
- 检查环境变量 `NEXT_PUBLIC_API_URL` 是否正确设置
- 确保后端服务已部署并可访问

### 3. 环境变量未生效

**问题**: 环境变量设置后仍使用默认值

**解决**:
- 确保环境变量名称以 `NEXT_PUBLIC_` 开头（Next.js 要求）
- 重新部署项目以应用新的环境变量
- 检查环境变量的作用域（Production/Preview/Development）

### 4. 静态资源加载失败

**问题**: 图片、字体等静态资源无法加载

**解决**:
- 检查 `public` 目录中的文件是否正确
- 确保资源路径使用相对路径或绝对路径
- 检查 `next.config.mjs` 中的 `images.unoptimized` 配置

## 📊 监控和分析

Vercel 提供：
- **Analytics**: 访问量、性能指标
- **Logs**: 实时日志查看
- **Speed Insights**: 性能分析

在项目设置中启用这些功能。

## 🔐 安全建议

1. **环境变量安全**
   - 不要在代码中硬编码敏感信息
   - 使用 Vercel 的环境变量管理
   - 定期轮换 API 密钥

2. **HTTPS**
   - Vercel 自动提供 HTTPS
   - 确保所有 API 调用使用 HTTPS

3. **CORS 配置**
   - 在后端配置正确的 CORS 策略
   - 只允许信任的域名访问

## 📚 相关文档

- [Vercel 官方文档](https://vercel.com/docs)
- [Next.js 部署文档](https://nextjs.org/docs/deployment)
- [环境变量配置](https://vercel.com/docs/concepts/projects/environment-variables)

## ✅ 部署检查清单

- [ ] Vercel 账号已创建
- [ ] 代码已推送到 Git 仓库
- [ ] 后端 API 已部署
- [ ] 环境变量已配置
- [ ] 构建成功
- [ ] 域名已配置（可选）
- [ ] CORS 已配置
- [ ] 功能测试通过

## 🎉 部署完成

部署成功后，你会获得：
- 生产环境 URL: `https://your-project.vercel.app`
- 预览环境 URL: 每个 PR 都有独立的预览 URL
- 自动 HTTPS
- 全球 CDN 加速
- 自动部署

---

**需要帮助？** 查看 [Vercel 支持](https://vercel.com/support) 或项目 Issues。
