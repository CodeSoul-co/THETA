# 服务器文件目录结构

本文档说明 THETA 项目在服务器上的文件目录结构，帮助理解和维护项目。

## 目录概览

```
/opt/theta/                    # 项目根目录（默认部署路径）
├── assets/                    # 静态资源
├── embedding/                 # 嵌入模型处理
├── ETM/                       # ETM 主题模型核心
├── langgraph_agent/           # LangGraph 智能代理后端
├── theta-frontend3/           # Next.js 前端应用
├── data/                      # 用户上传数据（运行时生成）
├── result/                    # 训练结果（运行时生成）
├── users.db                   # 用户数据库（SQLite）
├── docker-compose.yml         # Docker Compose 配置
├── docker-deploy.sh           # Docker 一键部署脚本
├── start.sh                   # 本地开发启动脚本
└── *.md                       # 项目文档
```

---

## 详细目录结构

### 1. 项目根目录 (`/opt/theta/`)

```
theta/
├── assets/                    # 静态资源
│   ├── THETA.png             # 项目 Logo
│   └── readme.md             # 资源说明
│
├── embedding/                 # 嵌入模型处理模块
│   ├── balanced_sampler.py   # 平衡采样器
│   ├── data_loader.py        # 数据加载器
│   ├── embedder.py           # 嵌入生成器
│   ├── main.py               # 主程序入口
│   ├── trainer.py            # 训练器（基础版）
│   └── trainer_v2.py         # 训练器（增强版）
│
├── ETM/                       # ETM 主题模型核心模块
│   ├── agent/                # AI 代理模块
│   │   ├── api/              # API 接口
│   │   ├── core/             # 核心逻辑
│   │   ├── memory/           # 记忆系统
│   │   ├── modules/          # 功能模块
│   │   ├── utils/            # 工具函数
│   │   ├── README.md         # 代理模块说明
│   │   └── start_api.sh      # 启动脚本
│   │
│   ├── dataclean/            # 数据清洗模块
│   │   ├── src/              # 源代码
│   │   │   ├── cleaner.py    # 清洗器
│   │   │   ├── converter.py  # 格式转换器
│   │   │   ├── consolidator.py # 数据合并器
│   │   │   ├── processor.py  # 处理器
│   │   │   └── cli.py        # 命令行接口
│   │   ├── api.py            # FastAPI 服务
│   │   ├── main.py           # 主程序
│   │   ├── Dockerfile        # Docker 镜像定义
│   │   ├── requirements.txt  # Python 依赖
│   │   └── *.md              # 文档文件
│   │
│   ├── engine_a/             # 引擎 A（BOW 生成）
│   │   ├── bow_generator.py  # BOW 矩阵生成器
│   │   └── vocab_builder.py  # 词汇表构建器
│   │
│   ├── engine_c/             # 引擎 C（ETM 核心）
│   │   ├── etm.py            # ETM 模型实现
│   │   ├── encoder.py        # 编码器
│   │   ├── decoder.py        # 解码器
│   │   └── vocab_embedder.py # 词汇嵌入器
│   │
│   ├── preprocessing/        # 预处理模块
│   │   └── embedding_processor.py # 嵌入处理器
│   │
│   ├── trainer/              # 训练器模块
│   │   └── trainer.py        # 训练器实现
│   │
│   ├── evaluation/           # 评估模块
│   │   ├── metrics.py        # 评估指标
│   │   └── topic_metrics.py  # 主题指标
│   │
│   ├── visualization/        # 可视化模块
│   │   └── topic_visualizer.py # 主题可视化器
│   │
│   ├── config.py             # 配置管理
│   ├── train_etm.py          # ETM 训练脚本
│   ├── run_etm_simple.py     # 简化运行脚本
│   └── requirements.txt      # Python 依赖
│
├── langgraph_agent/          # LangGraph 智能代理后端
│   ├── backend/              # 后端应用
│   │   ├── app/              # 应用主目录
│   │   │   ├── agents/       # 代理节点
│   │   │   │   ├── etm_agent.py  # ETM 代理
│   │   │   │   └── nodes.py  # 节点定义
│   │   │   ├── api/          # API 路由
│   │   │   │   ├── routes.py # 路由定义
│   │   │   │   ├── auth.py   # 认证路由
│   │   │   │   └── websocket.py # WebSocket 支持
│   │   │   ├── core/         # 核心配置
│   │   │   │   ├── config.py # 配置管理
│   │   │   │   └── logging.py # 日志配置
│   │   │   ├── models/       # 数据模型
│   │   │   │   └── user.py   # 用户模型
│   │   │   ├── schemas/      # Pydantic 模式
│   │   │   │   ├── agent.py  # 代理模式
│   │   │   │   ├── auth.py   # 认证模式
│   │   │   │   └── data.py   # 数据模式
│   │   │   ├── services/     # 业务逻辑
│   │   │   │   ├── auth_service.py   # 认证服务
│   │   │   │   └── chat_service.py   # 聊天服务
│   │   │   ├── static/       # 静态文件
│   │   │   │   └── index.html # 默认页面
│   │   │   └── main.py       # FastAPI 主程序
│   │   ├── requirements.txt  # Python 依赖
│   │   ├── run.py            # 启动脚本
│   │   └── *.md              # 文档文件
│   └── start_backend.sh      # 后端启动脚本
│
├── theta-frontend3/          # Next.js 前端应用
│   ├── app/                  # Next.js App Router
│   │   ├── admin/            # 管理后台
│   │   │   └── monitor/      # 服务监控页面
│   │   ├── login/            # 登录页面
│   │   ├── register/         # 注册页面
│   │   ├── results/          # 结果分析页面
│   │   ├── training/         # 训练页面
│   │   ├── visualizations/   # 可视化页面
│   │   ├── layout.tsx        # 根布局
│   │   ├── page.tsx          # 首页
│   │   └── globals.css       # 全局样式
│   │
│   ├── components/           # React 组件
│   │   ├── ui/               # UI 组件库（shadcn/ui）
│   │   ├── data-processing.tsx    # 数据处理组件
│   │   ├── heatmap-grid.tsx       # 热力图组件
│   │   ├── markdown-renderer.tsx  # Markdown 渲染器
│   │   ├── protected-route.tsx    # 路由保护组件
│   │   ├── terminal.tsx            # 终端组件
│   │   ├── theme-provider.tsx      # 主题提供者
│   │   ├── topic-bubble-map.tsx    # 主题气泡图
│   │   ├── training-chart.tsx      # 训练图表
│   │   ├── typing-message.tsx      # 打字效果消息
│   │   └── workspace-layout.tsx    # 工作空间布局
│   │
│   ├── contexts/             # React Context
│   │   └── auth-context.tsx  # 认证上下文
│   │
│   ├── hooks/                # 自定义 Hooks
│   │   ├── use-etm-websocket.ts  # WebSocket Hook
│   │   ├── use-mobile.ts         # 移动端检测 Hook
│   │   ├── use-toast.ts          # Toast Hook
│   │   └── use-typewriter.ts     # 打字效果 Hook
│   │
│   ├── lib/                  # 工具库
│   │   ├── api/              # API 客户端
│   │   │   ├── auth.ts       # 认证 API
│   │   │   ├── dataclean.ts  # 数据清洗 API
│   │   │   └── etm-agent.ts  # ETM 代理 API
│   │   └── utils.ts          # 工具函数
│   │
│   ├── public/               # 静态资源
│   │   ├── icon.svg          # 图标
│   │   └── *.png             # 图片资源
│   │
│   ├── styles/               # 样式文件
│   │   └── globals.css       # 全局样式
│   │
│   ├── Dockerfile            # Docker 镜像定义
│   ├── package.json          # Node.js 依赖配置
│   ├── pnpm-lock.yaml        # pnpm 锁文件
│   ├── next.config.mjs       # Next.js 配置
│   ├── tsconfig.json         # TypeScript 配置
│   └── *.md                  # 文档文件
│
├── data/                     # 用户上传数据（运行时生成，会被 .gitignore 忽略）
│   └── {dataset_name}/       # 数据集目录
│       └── *.csv, *.txt, *.json  # 数据文件
│
├── result/                   # 训练结果（运行时生成，会被 .gitignore 忽略）
│   └── {dataset_name}/       # 数据集结果目录
│       ├── embedding/        # 嵌入文件
│       │   ├── *_bow.npz     # BOW 矩阵
│       │   ├── *_bow_meta.json # BOW 元数据
│       │   └── *_vocab.json  # 词汇表
│       ├── topics/           # 主题文件
│       └── visualizations/   # 可视化文件
│
├── users.db                  # SQLite 用户数据库（运行时生成）
│
├── docker-compose.yml        # Docker Compose 配置文件
├── docker-deploy.sh          # Docker 一键部署脚本
├── docker.env.template       # Docker 环境变量模板
├── prepare-offline-build.sh  # 离线构建准备脚本
├── start.sh                  # 本地开发启动脚本
│
└── 文档文件/
    ├── README.md             # 项目主文档（英文）
    ├── README_CN.md          # 项目主文档（中文）
    ├── DEVELOPMENT_PROGRESS.md  # 开发进度文档
    ├── DOCKER_DEPLOY.md      # Docker 部署指南
    ├── QUICK_START_DOCKER.md # Docker 快速开始指南
    ├── OFFLINE_BUILD.md      # 离线构建指南
    └── SERVER_DEPLOYMENT.md  # 服务器部署指南
```

---

## Docker 容器目录映射

使用 Docker 部署时，以下目录会被映射到 Docker 卷：

### 数据卷映射

```yaml
volumes:
  dataclean-uploads:      # 数据清洗上传文件卷
    → /app/temp_uploads   # 容器内路径
  dataclean-processed:    # 数据清洗处理结果卷
    → /app/temp_processed # 容器内路径
```

### 挂载路径（如需持久化数据）

如果需要持久化数据，可以在 `docker-compose.yml` 中添加：

```yaml
volumes:
  - ./data:/opt/theta/data           # 用户数据
  - ./result:/opt/theta/result       # 训练结果
  - ./users.db:/opt/theta/users.db   # 用户数据库
```

---

## 运行时生成目录说明

以下目录和文件在运行时生成，不在版本控制中：

### 1. `data/` - 用户上传数据

- **用途**: 存储用户上传的原始数据集
- **格式**: CSV, TXT, JSON
- **生命周期**: 用户上传时创建，可手动删除
- **备份**: 建议定期备份重要数据集

### 2. `result/` - 训练结果

- **用途**: 存储训练生成的结果文件
- **包含**:
  - BOW 矩阵文件 (`*.npz`)
  - 词汇表文件 (`*.json`)
  - 主题模型文件
  - 可视化图表
- **生命周期**: 训练时自动生成
- **大小**: 可能很大，建议定期清理

### 3. `users.db` - 用户数据库

- **用途**: SQLite 数据库，存储用户账号信息
- **包含**: 用户名、邮箱、密码哈希、创建时间等
- **备份**: **必须定期备份**
- **位置**: 项目根目录

### 4. Python 缓存文件

- `__pycache__/` - Python 字节码缓存目录
- `*.pyc` - Python 编译文件
- **清理**: 可安全删除，会在下次运行时重新生成

---

## 重要配置文件位置

### 环境变量文件

- **`.env`** - 生产环境变量（不提交到 Git）
- **`docker.env.template`** - Docker 环境变量模板

### 服务配置

- **`docker-compose.yml`** - Docker Compose 服务配置
- **`ETM/config.py`** - ETM 模型配置
- **`langgraph_agent/backend/app/core/config.py`** - 后端服务配置
- **`theta-frontend3/next.config.mjs`** - Next.js 前端配置

### 依赖文件

- **`ETM/requirements.txt`** - ETM 模块 Python 依赖
- **`ETM/dataclean/requirements.txt`** - 数据清洗 API Python 依赖
- **`langgraph_agent/backend/requirements.txt`** - 后端 API Python 依赖
- **`theta-frontend3/package.json`** - 前端 Node.js 依赖

---

## 日志文件位置

### Docker 容器日志

```bash
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
docker-compose logs dataclean-api
docker-compose logs theta-frontend

# 实时跟踪日志
docker-compose logs -f dataclean-api
```

### 应用日志

- **数据清洗 API**: 标准输出（容器日志）
- **后端 API**: 标准输出（容器日志）
- **前端应用**: Next.js 日志（容器日志）

---

## 备份建议

### 必须备份的文件/目录

1. **`users.db`** - 用户数据库
2. **`.env`** - 环境变量配置
3. **`data/`** - 用户上传的重要数据集
4. **`result/`** - 重要的训练结果

### 备份脚本示例

```bash
#!/bin/bash
# backup.sh - THETA 项目备份脚本

BACKUP_DIR="/backup/theta_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 备份用户数据库
cp /opt/theta/users.db "$BACKUP_DIR/"

# 备份环境变量
cp /opt/theta/.env "$BACKUP_DIR/" 2>/dev/null || true

# 备份数据目录（如果存在）
if [ -d "/opt/theta/data" ]; then
    tar -czf "$BACKUP_DIR/data.tar.gz" -C /opt/theta data/
fi

# 备份结果目录（如果存在）
if [ -d "/opt/theta/result" ]; then
    tar -czf "$BACKUP_DIR/result.tar.gz" -C /opt/theta result/
fi

echo "备份完成: $BACKUP_DIR"
```

---

## 磁盘空间管理

### 大型目录（可能占用大量空间）

1. **`theta-frontend3/node_modules/`** - 前端依赖（~500MB）
2. **`result/`** - 训练结果（可能 GB 级）
3. **`data/`** - 用户数据（可能 GB 级）
4. **Docker 镜像和卷** - 可能占用数 GB

### 清理建议

```bash
# 清理 Docker 未使用的资源
docker system prune -a

# 清理训练结果（保留最近 30 天）
find /opt/theta/result -type d -mtime +30 -exec rm -rf {} +

# 清理 Python 缓存
find /opt/theta -type d -name __pycache__ -exec rm -rf {} +
find /opt/theta -type f -name "*.pyc" -delete
```

---

## 权限管理

### 推荐的文件权限

```bash
# 项目根目录
chmod 755 /opt/theta

# 脚本文件
chmod +x /opt/theta/*.sh
chmod +x /opt/theta/**/*.sh

# 数据目录（如果存在）
chmod 755 /opt/theta/data
chmod 644 /opt/theta/data/**/*

# 结果目录（如果存在）
chmod 755 /opt/theta/result
chmod 644 /opt/theta/result/**/*
```

---

## 目录大小检查命令

```bash
# 查看项目总大小
du -sh /opt/theta

# 查看各子目录大小
du -sh /opt/theta/*

# 查看最大的目录
du -h /opt/theta | sort -rh | head -20

# 查看 Docker 卷大小
docker system df
```

---

## 相关文档

- [DOCKER_DEPLOY.md](./DOCKER_DEPLOY.md) - Docker 部署详细指南
- [QUICK_START_DOCKER.md](./QUICK_START_DOCKER.md) - 快速开始指南
- [SERVER_DEPLOYMENT.md](./SERVER_DEPLOYMENT.md) - 服务器部署指南

---

**最后更新**: 2025-01-XX  
**维护者**: THETA 项目团队
