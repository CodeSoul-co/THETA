# Translate Commit Messages to English

This guide explains how to translate Chinese commit messages to English.

## ⚠️ Important Warning

**This will rewrite git history!** If these commits are already pushed to a remote repository, you'll need to force push after translation. Make sure to coordinate with your team before doing this.

## Quick Start

Run the translation script:

```bash
./translate_commits.sh
```

The script will:
1. Check for uncommitted changes
2. Ask for confirmation
3. Translate the last 20 commits from Chinese to English
4. Show you how to push the changes

## Commit Message Translations

The following commit messages will be translated:

| Chinese | English |
|---------|---------|
| fix: 修复部署问题并清理无用文件 | fix: fix deployment issues and clean up unused files |
| fix: 添加前端 lib 目录到版本控制 | fix: add frontend lib directory to version control |
| fix: 恢复缺失的 lib/utils.ts 和 lib/api/etm-agent.ts | fix: restore missing lib/utils.ts and lib/api/etm-agent.ts |
| feat: 添加 Railway 部署配置 | feat: add Railway deployment configuration |
| feat: 添加向量化模块和完善前端界面 | feat: add embedding module and improve frontend interface |
| chore: 删除无用文件并整理项目结构 | chore: remove unused files and organize project structure |
| docs: 更新项目进度，记录今日工作内容（前端优化、后端集成、Docker部署配置） | docs: update project progress, record today's work (frontend optimization, backend integration, Docker deployment configuration) |
| chore: 清理无用文件并更新 .gitignore | chore: clean up unused files and update .gitignore |
| fix: 改进 Docker daemon 检查逻辑 | fix: improve Docker daemon check logic |
| fix: 添加 Docker 环境检查到离线构建脚本 | fix: add Docker environment check to offline build script |
| docs: 添加离线 Docker 构建指南 | docs: add offline Docker build guide |
| fix: 修复 Docker 构建网络问题并添加离线构建方案 | fix: fix Docker build network issues and add offline build solution |
| fix: 修复 Docker 构建错误 | fix: fix Docker build errors |
| feat: 腾讯云服务器 Docker 部署优化 | feat: optimize Docker deployment for Tencent Cloud server |
| fix: Docker 部署脚本兼容 docker.env.template | fix: make Docker deployment script compatible with docker.env.template |
| docs: 添加 Docker 快速部署指南 | docs: add Docker quick deployment guide |
| feat: 优化 Docker 部署配置和脚本 | feat: optimize Docker deployment configuration and scripts |
| feat: 添加服务器部署配置和脚本 | feat: add server deployment configuration and scripts |
| feat: 添加后端部署配置文件 | feat: add backend deployment configuration files |
| fix: 优化 Netlify 配置 | fix: optimize Netlify configuration |

## After Translation

### View Changes

```bash
git log --oneline -20
```

### Push Changes (if already pushed to remote)

```bash
git push --force-with-lease origin frontend-3
```

**Note:** Use `--force-with-lease` instead of `--force` for safety. It will fail if someone else has pushed changes.

### Restore Original State (if needed)

```bash
git reset --hard refs/original/refs/heads/frontend-3
```

## Manual Method (Alternative)

If you prefer to do it manually using interactive rebase:

1. Start interactive rebase:
   ```bash
   git rebase -i HEAD~20
   ```

2. Change `pick` to `reword` for commits you want to edit

3. For each commit, replace the Chinese message with the English translation from the table above

4. Save and close the editor

5. Git will open another editor for each commit message - replace with English version

## Troubleshooting

### Script fails with "uncommitted changes"

Commit or stash your changes first:
```bash
git add .
git commit -m "WIP: uncommitted changes"
# or
git stash
```

### Want to translate more/fewer commits

Edit the script and change `HEAD~20` to the desired number:
```bash
git filter-branch -f --msg-filter "$TEMP_SCRIPT" HEAD~30..HEAD  # Last 30 commits
```

### Need to add more translations

Edit the `translate_commits.sh` script and add more cases to the `case` statement.
