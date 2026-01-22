#!/bin/bash
# Translate Chinese commit messages to English
# This script uses git filter-branch with inline translation logic

cd "$(dirname "$0")"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not a git repository"
    exit 1
fi

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "Warning: You have uncommitted changes."
    echo "Please commit or stash them before running this script."
    exit 1
fi

echo "This script will translate Chinese commit messages to English."
echo "It will rewrite the last 20 commits."
echo ""
echo "WARNING: This will rewrite git history!"
echo "If these commits are already pushed, you'll need to force push."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Create a temporary script for message translation
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << 'EOF'
#!/bin/bash
# Translate commit message
msg=$(cat)
# Remove trailing whitespace and normalize
msg=$(echo "$msg" | sed 's/[[:space:]]*$//' | tr -d '\r')

# Use pattern matching for more flexibility
if echo "$msg" | grep -q "修复部署问题并清理无用文件"; then
    echo "fix: fix deployment issues and clean up unused files"
elif echo "$msg" | grep -q "添加前端 lib 目录到版本控制"; then
    echo "fix: add frontend lib directory to version control"
elif echo "$msg" | grep -q "恢复缺失的 lib/utils.ts 和 lib/api/etm-agent.ts"; then
    echo "fix: restore missing lib/utils.ts and lib/api/etm-agent.ts"
elif echo "$msg" | grep -q "添加 Railway 部署配置"; then
    echo "feat: add Railway deployment configuration"
elif echo "$msg" | grep -q "添加向量化模块和完善前端界面"; then
    echo "feat: add embedding module and improve frontend interface"
elif echo "$msg" | grep -q "删除无用文件并整理项目结构"; then
    echo "chore: remove unused files and organize project structure"
elif echo "$msg" | grep -q "更新项目进度，记录今日工作内容"; then
    echo "docs: update project progress, record today's work (frontend optimization, backend integration, Docker deployment configuration)"
elif echo "$msg" | grep -q "清理无用文件并更新 .gitignore"; then
    echo "chore: clean up unused files and update .gitignore"
elif echo "$msg" | grep -q "改进 Docker daemon 检查逻辑"; then
    echo "fix: improve Docker daemon check logic"
elif echo "$msg" | grep -q "添加 Docker 环境检查到离线构建脚本"; then
    echo "fix: add Docker environment check to offline build script"
elif echo "$msg" | grep -q "添加离线 Docker 构建指南"; then
    echo "docs: add offline Docker build guide"
elif echo "$msg" | grep -q "修复 Docker 构建网络问题并添加离线构建方案"; then
    echo "fix: fix Docker build network issues and add offline build solution"
elif echo "$msg" | grep -q "修复 Docker 构建错误"; then
    echo "fix: fix Docker build errors"
elif echo "$msg" | grep -q "腾讯云服务器 Docker 部署优化"; then
    echo "feat: optimize Docker deployment for Tencent Cloud server"
elif echo "$msg" | grep -q "Docker 部署脚本兼容 docker.env.template"; then
    echo "fix: make Docker deployment script compatible with docker.env.template"
elif echo "$msg" | grep -q "添加 Docker 快速部署指南"; then
    echo "docs: add Docker quick deployment guide"
elif echo "$msg" | grep -q "优化 Docker 部署配置和脚本"; then
    echo "feat: optimize Docker deployment configuration and scripts"
elif echo "$msg" | grep -q "添加服务器部署配置和脚本"; then
    echo "feat: add server deployment configuration and scripts"
elif echo "$msg" | grep -q "添加后端部署配置文件"; then
    echo "feat: add backend deployment configuration files"
elif echo "$msg" | grep -q "优化 Netlify 配置"; then
    echo "fix: optimize Netlify configuration"
else
    # Return original message if no translation
    echo "$msg"
fi
EOF

chmod +x "$TEMP_SCRIPT"

# Use git filter-branch to rewrite commit messages
echo "Translating commit messages..."
# Get the 20th commit hash (counting from HEAD)
BASE_COMMIT=$(git log --format="%H" -20 | tail -1)
echo "Translating commits from $BASE_COMMIT to HEAD..."
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --msg-filter "$TEMP_SCRIPT" $BASE_COMMIT^..HEAD

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Commit messages translated successfully!"
    echo ""
    echo "To see the changes:"
    echo "  git log --oneline -20"
    echo ""
    echo "To push the changes (if already pushed to remote):"
    echo "  git push --force-with-lease origin $(git branch --show-current)"
    echo ""
    echo "To restore original state (if needed):"
    echo "  git reset --hard refs/original/refs/heads/$(git branch --show-current)"
else
    echo ""
    echo "✗ Failed to translate commit messages."
    echo "To restore original state:"
    echo "  git reset --hard refs/original/refs/heads/$(git branch --show-current)"
fi

# Clean up
rm -f "$TEMP_SCRIPT"
