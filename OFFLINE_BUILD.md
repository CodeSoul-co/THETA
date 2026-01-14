# 离线 Docker 构建指南

当服务器无法访问 Docker Hub 或 Debian 软件源时，可以使用离线构建方案。

## 方案一：修复网络问题（推荐先尝试）

### 1. 修复 Dockerfile（已自动修复）

已更新 `ETM/dataclean/Dockerfile`，使用阿里云镜像源，如果失败会自动回退到官方源。

### 2. 配置 Docker 镜像加速器

在服务器上执行：

```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://mirror.ccs.tencentyun.com",
    "https://hub-mirror.c.163.com"
  ],
  "dns": ["8.8.8.8", "1.1.1.1"]
}
EOF

sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 3. 配置 Debian 软件源（如果仍有问题）

如果 `apt-get update` 仍然失败，在 Dockerfile 构建前手动配置：

```bash
# 在服务器上测试网络
curl -I http://mirrors.aliyun.com
curl -I http://deb.debian.org
```

如果都失败，使用离线构建方案。

---

## 方案二：离线构建（网络完全不可用时）

### 步骤 1：在本机准备离线文件

在你的**本地机器**（能联网的电脑）上运行：

```bash
cd /path/to/THETA
chmod +x prepare-offline-build.sh
./prepare-offline-build.sh
```

这个脚本会：
- 下载 `python:3.11-slim` 和 `node:20-alpine` Docker 镜像
- 下载所有 Python 依赖包（wheel 文件）
- 打包项目代码
- 创建部署脚本

完成后会生成 `docker-offline-build` 目录。

### 步骤 2：上传到服务器

将 `docker-offline-build` 目录上传到服务器：

```bash
# 使用 scp
scp -r docker-offline-build user@your-server:/root/

# 或使用 rsync
rsync -avz docker-offline-build/ user@your-server:/root/docker-offline-build/
```

### 步骤 3：在服务器上部署

SSH 到服务器，然后：

```bash
cd /root/docker-offline-build
chmod +x deploy-offline.sh
./deploy-offline.sh
```

---

## 方案三：手动离线构建（更灵活）

### 1. 在本机导出镜像

```bash
# 导出 Python 镜像
docker pull python:3.11-slim
docker save python:3.11-slim -o python-3.11-slim.tar

# 导出 Node.js 镜像
docker pull node:20-alpine
docker save node:20-alpine -o node-20-alpine.tar
```

### 2. 在本机下载 Python 依赖

```bash
cd ETM/dataclean
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip download -r requirements.txt -d wheels/
pip download fastapi uvicorn[standard] python-multipart -d wheels/
deactivate
```

### 3. 上传到服务器

```bash
# 上传镜像
scp python-3.11-slim.tar node-20-alpine.tar user@server:/root/

# 上传代码和依赖
scp -r ETM/dataclean user@server:/root/THETA/
scp -r wheels user@server:/root/THETA/ETM/dataclean/
```

### 4. 在服务器上构建

```bash
# 导入镜像
docker load -i python-3.11-slim.tar
docker load -i node-20-alpine.tar

# 使用离线 Dockerfile 构建
cd /root/THETA/ETM/dataclean
docker build -f Dockerfile.offline -t dataclean-api:latest .
```

---

## 故障排查

### 问题 1：`apt-get update` 失败

**解决方案**：
- 检查服务器 DNS：`ping mirrors.aliyun.com`
- 如果 DNS 失败，在 Dockerfile 中使用 IP 地址或配置 DNS
- 使用离线构建方案

### 问题 2：Docker Hub 连接超时

**解决方案**：
- 配置 Docker 镜像加速器（见方案一）
- 使用离线构建方案

### 问题 3：Python 依赖下载失败

**解决方案**：
- 使用国内 PyPI 镜像：
  ```bash
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
  ```
- 或使用离线构建方案

---

## 快速检查清单

- [ ] 服务器能访问 `mirrors.aliyun.com` 吗？
- [ ] 服务器能访问 `deb.debian.org` 吗？
- [ ] Docker 镜像加速器配置了吗？
- [ ] 如果都失败，使用离线构建方案

---

## 联系支持

如果以上方案都无法解决问题，请提供：
1. 服务器网络环境（国内/海外，云服务商）
2. `curl -I http://mirrors.aliyun.com` 的输出
3. `docker info` 的输出
4. 完整的错误日志
