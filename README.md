# RAG 企业文档智能问答系统

基于 RAG (Retrieval-Augmented Generation) 架构的企业级文档智能问答系统，支持 PDF、Word、Markdown 等常见格式文档的自动处理与语义检索，提供带答案溯源的智能问答服务。

## 系统架构

```
[文档摄入]
  PDF/DOCX/MD → 加载 → 清洗 → 语义分块 → Embedding → Milvus + BM25

[问答流程]
  用户问题 → 混合检索(BM25+向量) → 重排序 → Prompt构建 → LLM生成 → 答案+溯源
```

## 核心特性

- **多格式文档处理**：自动加载 PDF、Word、Markdown、TXT 文档
- **语义分块**：基于段落边界的智能文本分割，保留文档结构
- **混合检索**：BM25 关键词检索 + 语义向量检索 + Cross-encoder 重排序
- **多轮对话**：基于会话的对话记忆管理，支持长上下文
- **答案溯源**：每个回答附带原始文档片段，有效抑制模型幻觉
- **可配置**：YAML 驱动配置，支持调整分块策略、检索权重、Prompt 模板
- **多 LLM 支持**：OpenAI API / Ollama 本地模型灵活切换
- **Bad Case 分析**：内置 Bad Case 收集与分析工具，持续优化检索质量
- **并发支持**：基于 FastAPI 的异步服务，支持 20 路并发访问

## 快速开始

### 环境要求

- Python 3.10+
- (可选) CUDA GPU 用于模型加速

### 安装

```bash
# 1. 克隆或进入项目目录
cd RAG

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（使用 OpenAI 时）
# Windows PowerShell:
$env:OPENAI_API_KEY = "sk-your-api-key"

# Linux/Mac:
export OPENAI_API_KEY="sk-your-api-key"
```

### 配置

编辑 `config.yaml` 文件，主要配置项：

```yaml
# 文档处理
documents:
  chunk_size: 512          # 分块大小（字符数）

# 检索
retrieval:
  bm25_weight: 0.3         # BM25 权重
  vector_weight: 0.7       # 向量检索权重
  top_k_final: 5           # 最终传递给 LLM 的文档数

# LLM 选择
llm:
  provider: "openai"       # "openai" 或 "ollama"
```

使用本地 Ollama 模型：
```yaml
llm:
  provider: "ollama"
  ollama:
    model: "qwen2.5:7b"
    base_url: "http://localhost:11434"
```

### 放入文档

将你的文档放入 `data/documents/` 目录：

```
data/documents/
  ├── employee_handbook.md
  ├── lab_equipment_policy.md
  ├── security_training.md
  ├── 技术手册.pdf
  └── 管理制度.docx
```

### 启动服务

```bash
# 仅执行文档摄入
python run.py --ingest-only

# 启动 API 服务
python run.py

# 指定端口和配置
python run.py --config config.yaml --port 8080
```

服务启动后访问：
- API 文档 (Swagger): http://localhost:8000/docs
- 健康检查: http://localhost:8000/api/v1/health

## API 使用

### 问答接口

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "公司的年假政策是什么？",
    "session_id": "user_001"
  }'
```

响应示例：
```json
{
  "answer": "根据员工手册第三章，员工每年享有5天带薪年假...",
  "sources": [
    {
      "chunk_id": "abc123",
      "document_name": "employee_handbook.md",
      "content": "员工每年享有5天带薪年假...",
      "score": 0.92,
      "chunk_index": 3
    }
  ],
  "session_id": "user_001",
  "processing_time_ms": 1423,
  "context_docs": 5
}
```

### 检索接口（仅检索，不生成答案）

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "实验室设备申请", "top_k": 5}'
```

### 文档管理

```bash
# 查看索引状态
curl http://localhost:8000/api/v1/documents

# 触发重新摄入
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -H "Content-Type: application/json" \
  -d '{"drop_existing": true}'
```

## 项目结构

```
src/
├── config/          配置管理
│   └── loader.py    ConfigLoader (YAML + 环境变量)
├── document/        文档处理
│   ├── loader.py    多格式加载器 (PDF/DOCX/MD)
│   ├── cleaner.py   文本清洗器
│   └── chunker.py   语义分块策略
├── embedding/       向量化
│   └── embedder.py  HuggingFace Embedding 引擎
├── retrieval/       检索模块
│   ├── bm25.py      BM25 关键词检索
│   ├── vector.py    Milvus 向量检索
│   ├── reranker.py  Cross-encoder 重排序
│   └── hybrid.py    混合检索编排
├── llm/             LLM 提供商
│   ├── base.py      抽象基类
│   ├── openai_llm.py OpenAI 实现
│   └── ollama_llm.py Ollama 实现
├── memory/          对话记忆
│   └── conversation.py 会话管理
├── pipeline/        流水线
│   ├── ingest.py    文档摄入流水线
│   └── query.py     问答查询流水线
├── api/             API 服务
│   ├── server.py    FastAPI 应用
│   ├── routes.py    API 路由
│   └── schemas.py   Pydantic 模型
└── evaluation/      评估
    ├── badcase.py   Bad Case 分析
    └── metrics.py   评估指标
```

## 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_chunker.py -v
pytest tests/test_retrieval.py -v

# 运行测试并查看覆盖率
pytest tests/ --cov=src --cov-report=html
```

## 检索质量调优

### 调整分块策略

```yaml
documents:
  chunk_size: 256        # 减小分块可提高精确度（但可能丢失上下文）
  chunk_overlap: 50      # 增大重叠可减少信息断裂
```

### 调整检索权重

```yaml
retrieval:
  bm25_weight: 0.5       # 增大 BM25 权重可改善关键词匹配
  vector_weight: 0.5     # 增大向量权重可改善语义匹配
  top_k_final: 8         # 增大候选数可提高召回率（但增加 LLM 消耗）
```

### Bad Case 分析流程

1. 收集 Bad Case → `POST /api/v1/chat` 时记录不满意的回答
2. 分析模式 → 使用 `BadCaseAnalyzer.analyze()` 查看分类分布
3. 导出报告 → `BadCaseAnalyzer.export_report()` 生成 Markdown 报告
4. 参数调整 → 根据建议调整分块大小、检索权重或 Prompt 模板

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 文档加载 | PyMuPDF, python-docx, LangChain |
| 文本分块 | LangChain RecursiveCharacterTextSplitter + 自定义语义分块 |
| Embedding | BAAI/bge-large-zh-v1.5 |
| 向量数据库 | Milvus Lite (嵌入式部署) |
| BM25 | rank_bm25 + jieba 分词 |
| 重排序 | BAAI/bge-reranker-large |
| LLM | OpenAI GPT-4o / Ollama (Qwen2.5) |
| API 框架 | FastAPI + uvicorn |
| 配置管理 | YAML + 环境变量插值 |

## License

MIT
