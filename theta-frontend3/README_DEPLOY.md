# 部署指南

## Netlify 部署

### 前置要求
- GitHub 仓库已设置
- Netlify 账户

### 部署步骤

1. **在 Netlify 中创建新站点**
   - 登录 Netlify
   - 点击 "Add new site" -> "Import an existing project"
   - 选择 GitHub 仓库 `CodeSoul-co/THETA`
   - 选择分支 `frontend-3`

2. **配置构建设置**
   - Base directory: `theta-frontend3`
   - Build command: `pnpm install && pnpm build`
   - Publish directory: `theta-frontend3/.next`

3. **设置环境变量**
   在 Netlify 站点设置中添加：
   ```
   NEXT_PUBLIC_DATACLEAN_API_URL=https://your-api-domain.com
   ```
   如果 DataClean API 部署在其他地方，请设置正确的 URL。

4. **部署**
   - Netlify 会自动检测到 `netlify.toml` 配置
   - 点击 "Deploy site" 开始部署

### 注意事项

- 确保 DataClean API 后端服务已部署并可访问
- 如果 API 在本地，需要先部署 API 服务
- 生产环境建议使用 HTTPS

## 本地构建测试

```bash
cd theta-frontend3
pnpm install
pnpm build
pnpm start
```

## 环境变量

复制 `.env.example` 为 `.env.local` 并填写实际值：

```bash
cp .env.example .env.local
```
