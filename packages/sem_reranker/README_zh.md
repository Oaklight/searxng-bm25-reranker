# SearXNG Semantic Reranker

SearXNG 外部插件。它用远程向量模型、远程交叉编码器，或两者同时，对搜索结果重排序。模型托管与 SearXNG 进程分离：插件只需要 API 地址、密钥和模型名。

- **两个独立阶段** — 向量相似度与交叉编码器重排序；默认都关闭。
- **多种 API 格式** — 支持 OpenAI、Cohere、Jina、Voyage 的请求与响应格式，以及 SiliconFlow 预设。
- **无运行时依赖** — HTTP、缓存、重试、余弦相似度和 RRF 都来自复制到本地的 [zerodep](https://github.com/Oaklight/zerodep) 模块。
- **安全回退** — 提供方超时或报错时保留原来的排序。
- **可单独使用，也可与 `bm25_reranker` 协同** — 由一个协调者融合引擎顺序、BM25 和向量分数。

## 安装

```bash
pip install searxng-sem-reranker
```

注册插件，并至少启用一个阶段：

```yaml
plugins:
  searxng_sem_reranker.SXNGPlugin:
    active: true

sem_reranker:
  rerank:
    enabled: true
    format: jina
    url: https://api.siliconflow.cn/v1/rerank
    model: BAAI/bge-reranker-v2-m3
    api_key_env: SILICONFLOW_API_KEY
    top_k: 10
    timeout_ms: 3000
```

`plugins:` 只接受 `active`。其他选项写在顶层 `sem_reranker:` 块中。用户写出的 `plugins:` 块会替换 SearXNG 的默认插件列表，因此必须列出所有需要的插件。

## 配置

```yaml
sem_reranker:
  embedding:
    enabled: false
    format: openai                 # openai | cohere | jina | voyage
    url: https://api.siliconflow.cn/v1/embeddings
    model: BAAI/bge-m3             # 或 Qwen/Qwen3-Embedding-0.6B（32k）
    api_key_env: SILICONFLOW_API_KEY
    api_key: ""
    top_k: 0                       # 0 表示关闭
    weight: 1.0
    max_chars: 4000
    timeout_ms: 3000

  rerank:
    enabled: false
    format: jina                   # jina | cohere | voyage
    url: https://api.siliconflow.cn/v1/rerank
    model: BAAI/bge-reranker-v2-m3 # 或 Qwen/Qwen3-Reranker-0.6B（32k）
    api_key_env: SILICONFLOW_API_KEY
    api_key: ""
    top_k: 10
    weight: 1.0
    max_chars: 4000
    timeout_ms: 3000

  fusion:
    rrf_k: 20
    engine_weight: 1.0
    bm25_weight: 1.0
    neural_weight: 1.0

  cache:
    ttl: 300
    max_entries: 512
```

密钥先从 `api_key_env` 读取，再回退到 `api_key`。密钥不会写入日志。也可以直接设置密钥：把 `api_key_env` 留空，把密钥写进 `api_key`：

```yaml
  rerank:
    api_key_env: ""
    api_key: "your-key"
```

如果 `settings.yml` 会被多人读取，优先使用环境变量，因为任何能读文件的人都能读到明文密钥。

没有提供方预设：每个阶段字段都必须显式写出。

`embedding` 与 `rerank` 相互独立。两者可以使用不同的 `format`、`url`、`model`、`api_key_env` 和 `api_key`。`url` 是完整端点。

例如，embedding 与 rerank 可以来自不同主机，并使用不同密钥：

```yaml
sem_reranker:
  embedding:
    enabled: true
    format: openai
    url: https://embed-gateway.example/v1/embeddings
    model: BAAI/bge-m3
    api_key_env: EMBED_API_KEY

  rerank:
    enabled: true
    format: jina
    url: https://rerank-gateway.example/v1/rerank
    model: BAAI/bge-reranker-v2-m3
    api_key_env: RERANK_API_KEY
```

### 示例模型

下列模型由 SiliconFlow 提供。配置的端点支持的任何模型都可以使用。

| 阶段 | 模型 | 上下文 |
| --- | --- | --- |
| embedding | `BAAI/bge-m3` | 8k |
| embedding | `Qwen/Qwen3-Embedding-0.6B` | 32k |
| rerank | `BAAI/bge-reranker-v2-m3` | 8k |
| rerank | `Qwen/Qwen3-Reranker-0.6B` | 32k |

插件不假设向量维度，也不假设模型专用分词器。`max_chars` 在请求前截断过长的查询或文档；正常的 SearXNG 摘要远低于 8k 限制。

### 提供方格式

| 格式 | 阶段 | 请求 | 响应 |
| --- | --- | --- | --- |
| openai | embed | `model`、`input`、`encoding_format` | `data[].embedding` |
| cohere | embed | `model`、`texts`、`input_type`、`embedding_types` | `embeddings.float[]` |
| jina | embed | `model`、`input`、`task` | `data[].embedding` |
| voyage | embed | `model`、`input`、`input_type` | `data[].embedding` |
| jina | rerank | `model`、`query`、`documents`、`top_n` | `results[].{index,relevance_score}` |
| cohere | rerank | `model`、`query`、`documents`、`top_n` | `results[].{index,relevance_score}` |
| voyage | rerank | `model`、`query`、`documents`、`top_k` | `data[].{index,relevance_score}` |

同一个 `format` 可用于使用相同格式的自建网关。把 `url` 指向网关端点。

## 与 `bm25_reranker` 协同

SearXNG 插件顺序不稳定，所以只能有一个插件改写 `positions`。显式设置协调者：

```yaml
rerankers:
  coordinator: sem_reranker
```

协调者计算所有已启用的信号，并只改写一次位置。另一个插件直接返回。如果指定的协调者未启用，当前运行的插件接管。如果另一个包未安装，协调者只使用自己的信号。

当 BM25 作为协调者时，`rerankers.escalate` 决定是否调用本插件。指标见 BM25 的 README：覆盖率、分差和有分数结果数。

## 延迟

在本机对 SiliconFlow 实测，连接已预热：

| 调用 | 文档数 | 时间 |
| --- | --- | --- |
| rerank | 10 | 约 0.83–1.09 秒 |
| rerank | 30 | 约 1.1 秒 |
| embedding | 10 | 约 2.07–2.20 秒 |
| embedding | 30 | 约 2.6 秒 |

建议默认只开 rerank。embedding 阶段按需开启。

## 开发

```bash
make install-dev
make test
```

在仓库根目录运行 `pytest`，共享的 `conftest.py` 才能加载。设置 `SILICONFLOW_API_KEY` 后可运行实时测试；未设置时自动跳过。

## 许可证

[AGPL-3.0](LICENSE)。
