# UniAI Python AI 服务

> 企业级知识库平台 — Python AI 引擎，提供文档解析、智能分块、向量化索引、混合检索与多智能体问答能力。

## 目录

- [功能模块](#功能模块)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 接口](#api-接口)
- [部署指南](#部署指南)
- [开发指南](#开发指南)
- [常见问题](#常见问题)

---

## 功能模块

| 模块 | 路径 | 说明 |
|------|------|------|
| 文档解析引擎 | `app/services/parsing/` | MinerU+Marker PDF 解析、DOCX/Excel/HTML/MD/TXT，含质量评估 |
| 智能分块 | `app/services/chunking/` | 结构感知 + 语义感知分块，表格/代码特殊处理 |
| 向量化与索引 | `app/services/embedding/` `app/services/indexing/` | DashScope text-embedding-v3，Milvus + ES 双写 |
| 混合检索引擎 | `app/services/retrieval/` | 向量 + 稀疏 + BM25 三路检索、RRF 融合、BGE Rerank、权限过滤 |
| 多智能体问答 | `app/services/qa/` `app/agents/` | 查询理解 → LangGraph 多智能体 → SSE 流式输出 |
| 基础设施层 | `app/core/` | PgSQL / Redis / MinIO / Milvus / ES / RabbitMQ / LLM 连接 |
| 异步任务 | `app/tasks/` | Celery 解析任务，RabbitMQ 消息消费 |

---

## 目录结构

```
uni-ai-python/
├── app/
│   ├── main.py                    # FastAPI 应用入口
│   ├── config.py                  # 配置加载（YAML + 环境变量）
│   ├── core/                      # 基础设施客户端
│   │   ├── database.py            # PostgreSQL (asyncpg + SQLAlchemy)
│   │   ├── redis_client.py        # Redis
│   │   ├── minio_client.py        # MinIO
│   │   ├── milvus_client.py       # Milvus 向量库
│   │   ├── es_client.py           # Elasticsearch
│   │   ├── mq_consumer.py         # RabbitMQ 消费者
│   │   ├── llm_provider.py        # LiteLLM 多模型
│   │   ├── retrieval.py           # 检索服务门面
│   │   └── qa.py                  # QA 服务门面
│   ├── models/                    # Pydantic / SQLAlchemy 数据模型
│   ├── api/v1/                    # FastAPI 路由
│   │   ├── parse.py               # 文档解析接口
│   │   ├── knowledge.py           # 知识库管理接口
│   │   ├── chat.py                # 问答/对话接口
│   │   └── system.py              # 健康检查/监控
│   ├── services/
│   │   ├── parsing/               # 解析引擎
│   │   ├── chunking/              # 分块策略
│   │   ├── embedding/             # Embedding 模型
│   │   ├── indexing/              # Milvus + ES 索引写入
│   │   ├── retrieval/             # 混合检索
│   │   └── qa/                    # 问答引擎
│   ├── agents/                    # LangGraph 智能体节点
│   ├── tasks/                     # Celery 异步任务
│   └── utils/                     # 异常/工具
├── configs/
│   ├── config.yaml                # 默认配置
│   ├── config.dev.yaml            # 开发环境覆盖
│   ├── config.test.yaml           # 测试环境覆盖
│   └── config.prod.yaml           # 生产环境覆盖
├── docker/
│   ├── Dockerfile                 # 多阶段构建
│   ├── docker-compose.infra.yml   # 基础设施（独立启动）
│   ├── docker-compose.yml         # Python AI 服务
│   └── nginx/
│       └── nginx.conf             # Nginx 反向代理配置
├── scripts/
│   ├── init_milvus.py             # Milvus Collection 初始化
│   ├── init_es.py                 # ES Index 初始化
│   ├── download_models.py         # 下载本地模型
│   └── test_parse_chunk.py        # 解析+分块冒烟测试
├── tests/
│   ├── conftest.py
│   └── test_chunking.py
├── .env.example                   # 环境变量模板
├── .gitignore
├── pyproject.toml                 # 依赖 + 工具配置
└── Makefile                       # 常用命令
```

---

## 快速开始

### 环境要求

- Python 3.11+
- Docker & Docker Compose（启动基础设施）
- 8 GB+ RAM（运行 BGE Reranker；无 GPU 也可在 CPU 跑）

### 1. 克隆并进入目录

```bash
git clone <repo-url>
cd uni-ai-platform/uni-ai-python
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填写以下 API Key：

```dotenv
# 必填：DashScope（Qwen LLM + Embedding）
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx

# 可选：其他 LLM 提供商
DEEPSEEK_API_KEY=sk-xxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

### 3. 启动基础设施

```bash
make docker-up-infra
# 等待所有服务健康，通常 30–60 秒
```

### 4. 安装依赖

```bash
pip install -e ".[dev]"
```

### 5. 初始化索引

```bash
make init-db        # 创建 Milvus Collection + ES Index
```

### 6. 启动服务

```bash
make dev            # 开发模式（热重载）
# 或
uvicorn app.main:app --reload --port 8100
```

访问：
- **Swagger UI**: http://localhost:8100/docs
- **ReDoc**: http://localhost:8100/redoc
- **健康检查**: http://localhost:8100/api/v1/system/health

### 7. 启动 Celery Worker（可选，文档异步解析需要）

```bash
make worker
```

---

## 配置说明

服务使用 **YAML 多环境配置** + **环境变量覆盖**。

### 加载顺序

```
configs/config.yaml          ← 默认值（必须存在）
    + configs/config.{env}.yaml  ← 按 UNI_AI_ENV 加载（dev/test/prod）
        + 环境变量 UNI_AI__<SECTION>__<KEY>  ← 最高优先级
```

### 关键配置项

```yaml
# configs/config.yaml（节选）

llm:
  default_model: "openai/qwen-max"   # 默认 LLM
  temperature: 0.1

embedding:
  model: "text-embedding-v3"         # DashScope online（无需本地 GPU）
  dimension: 1024
  batch_size: 25

retrieval:
  mode: "hybrid"                     # hybrid / vector_only / fulltext_only
  vector_weight: 0.4
  bm25_weight: 0.3
  rerank: true
```

### 环境变量覆盖示例

```bash
# 覆盖数据库连接
export UNI_AI__DATABASE__URL=postgresql+asyncpg://user:pass@prod-db:5432/uni_ai

# 覆盖 Redis
export UNI_AI__REDIS__URL=redis://:password@prod-redis:6379/0

# 切换环境
export UNI_AI_ENV=prod
```

---

## API 接口

> 完整文档见 http://localhost:8100/docs（Swagger UI）

### 文档解析

```
POST /api/v1/parse/document
     Body: { file_key, knowledge_id }        # MinIO 文件键值
     → 返回解析任务 ID，通过 GET /api/v1/parse/status/{task_id} 查询进度
```

### 知识库管理

```
POST   /api/v1/knowledge/index              # 触发向量化+索引（同步）
DELETE /api/v1/knowledge/document/{doc_id}  # 删除文档及索引
GET    /api/v1/knowledge/search             # 混合检索（调试用）
```

### 智能问答

```
POST /api/v1/chat/completions              # SSE 流式问答
     Body: {
       messages: [{role, content}, ...],
       knowledge_ids: [...],
       mode: "auto" | "simple" | "deep",
       stream: true
     }

POST /api/v1/chat/query                    # 非流式问答（同步返回）
```

### 系统

```
GET /api/v1/system/health                  # 健康检查
GET /api/v1/system/metrics                 # Prometheus 指标（/metrics）
```

---

## 部署指南

### Docker 一键部署

```bash
# 构建镜像
make docker-build

# 启动基础设施 + AI 服务
docker compose -f docker/docker-compose.infra.yml \
               -f docker/docker-compose.yml up -d

# 查看日志
docker logs -f uni-ai-python
```

### 生产环境要点

1. **修改默认密码**：PostgreSQL、MinIO、RabbitMQ 的默认账密必须在生产环境中修改。
2. **绑定 Secrets**：所有 API Key 通过 Docker Secrets 或 K8s Secret 注入，不写入镜像。
3. **持久化卷**：确认 `pg_data`、`milvus_data`、`es_data` 等卷映射到可靠存储。
4. **Nginx**：`docker/nginx/nginx.conf` 提供了反向代理配置，建议在服务前加 TLS 终止。
5. **Worker 扩容**：`docker compose up -d --scale uni-ai-celery=4` 横向扩展 Celery Worker。

### 资源建议（生产最低配置）

| 服务 | CPU | RAM |
|------|-----|-----|
| Python AI 服务 | 4 Core | 8 GB |
| Milvus | 4 Core | 16 GB |
| Elasticsearch | 4 Core | 8 GB |
| PostgreSQL | 2 Core | 4 GB |
| Redis | 1 Core | 2 GB |

---

## 开发指南

### 常用命令

```bash
make dev              # 启动开发服务器（热重载）
make worker           # 启动 Celery Worker
make test             # 运行测试
make test-cov         # 测试 + 覆盖率报告
make lint             # Ruff 代码检查
make format           # Ruff 格式化 + 自动修复
make docker-up-infra  # 启动基础设施
make docker-down-infra # 停止基础设施
make init-db          # 初始化 Milvus + ES 索引
make download-models  # 下载本地 Reranker 模型
```

### 添加新的文档解析器

1. 在 `app/services/parsing/` 中创建新类，继承 `BaseParser`。
2. 实现 `can_handle(file_type)` 和 `parse(file_path, **kwargs) -> ParseResult`。
3. 在 `app/services/parsing/engine.py` 的 `_parsers` 列表中注册。

### 添加新的 LLM 提供商

在 `configs/config.yaml` 的 `llm.models` 中增加配置项即可，LiteLLM 统一抽象无需改代码：

```yaml
llm:
  models:
    - name: "openai/your-model"
      api_key: "${YOUR_API_KEY}"
      api_base: "https://your-provider.com/v1"
```

### 代码规范

- **类型注解**：所有公共函数必须有完整的 type hints
- **异步优先**：I/O 操作一律 `async/await`
- **日志**：使用 `from loguru import logger`，禁止使用 `print`
- **配置**：通过 `get_settings()` 获取，不要硬编码
- **测试**：新功能需附带 `tests/` 目录下的测试用例

---

## 常见问题

**Q: 启动报 `pymilvus.exceptions.MilvusException: connection failed`**
A: 确认 `make docker-up-infra` 已执行，Milvus 完全启动需要约 30 秒。用 `docker logs uni-ai-milvus` 查看状态。

**Q: Embedding 请求报错**
A: 检查 `.env` 中 `DASHSCOPE_API_KEY` 是否已填写正确的阿里云灵积 API Key。

**Q: PDF 解析慢或失败**
A: MinerU 需要下载模型文件（首次约 1–2 GB）。如果服务器无法访问 HuggingFace，可配置镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Q: Reranker 运行太慢**
A: 在无 GPU 环境下，BGE Reranker 在 CPU 上较慢。可在 `config.dev.yaml` 中将 `reranker.device` 设为 `cpu` 并减小 `top_k`。

**Q: 如何查看完整请求日志**
A: 设置 `debug: true`（`config.dev.yaml` 已默认开启），或通过环境变量 `UNI_AI__DEBUG=true` 开启。
