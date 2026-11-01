# SearXNG Rerankers

两个相互独立的 SearXNG 重排序插件。它们放在同一个仓库里，可以单独安装，也可以同时启用并协同工作。

| 包 | 发行名 | 插件 id | 信号 |
| --- | --- | --- | --- |
| [`packages/bm25_reranker`](packages/bm25_reranker) | `searxng-bm25-reranker` | `bm25_reranker` | BM25F 文本相关性 |
| [`packages/sem_reranker`](packages/sem_reranker) | `searxng-sem-reranker` | `sem_reranker` | 向量与交叉编码器分数 |

两个包都没有运行时依赖。第三方代码只有经 [zerodep](https://github.com/Oaklight/zerodep) CLI 复制到 `_vendor/` 的模块。

## 安装

```bash
pip install searxng-bm25-reranker
pip install searxng-sem-reranker
```

本地开发：

```bash
make install-dev
```

## 注册插件

在 `plugins:` 下注册插件类，在顶层块中写插件选项。顶层块的键是插件 id：

```yaml
plugins:
  searxng_bm25_reranker.SXNGPlugin:
    active: true
  searxng_sem_reranker.SXNGPlugin:
    active: true

bm25_reranker:
  rrf_k: 20

sem_reranker:
  rerank:
    enabled: true
    format: jina
    url: https://api.siliconflow.cn/v1/rerank
    model: BAAI/bge-reranker-v2-m3
    api_key_env: SILICONFLOW_API_KEY
    top_k: 10
    timeout_ms: 3000

rerankers:
  coordinator: sem_reranker
```

每个阶段都有自己的 `format`、`url`、`api_key_env` 和 `api_key`。embedding 与 rerank 可以使用不同主机和不同密钥。

SearXNG 的两条配置规则：

- 每个 `plugins:` 条目只接受 `active`。其他键会传给 `PluginCfg(**settings)` 并触发 `TypeError`。插件选项必须写在顶层块中。
- 用户写出的 `plugins:` 块会替换 SearXNG 的默认插件列表。必须列出所有需要的插件，包括内置插件；或者保留默认列表，只合并需要的内容。

## 协同

SearXNG 用集合保存插件，因此 `post_search` 的顺序不稳定。两个插件都会改写 `positions`，所以只能有一个插件执行重排序。

把 `rerankers.coordinator` 设为 `bm25_reranker` 或 `sem_reranker`。被指定的插件计算自己的信号；如果另一个包已安装并启用，它还导入对方的评分函数；然后用 RRF 融合引擎顺序、BM25 和向量分数，只改写一次 `positions`。另一个插件直接返回，不发起网络请求，也不改写位置。

如果指定的协调者未启用，当前运行的插件接管协调并写警告。如果两个插件都启用但未配置协调者，`sem_reranker` 负责协调。如果另一个包未安装，协调者只使用自己的信号。

当 BM25 作为协调者时，可以给 sem 调用加门控：

```yaml
rerankers:
  coordinator: bm25_reranker
  escalate:
    enabled: true
    min_coverage: 0.5
    min_margin: 0.1
    min_scored: 2
    top_k: 10
```

先跑 BM25。词面覆盖率低、首名分差小、或有分数结果太少时，门控打开。只有这时才调用 `sem_reranker`。`enabled: false`（默认值）时，sem 只要启用就在每次请求中调用。

## 开发

```bash
make test
make lint
make build
make vendor
```

在仓库根目录运行 `pytest`，共享的 `conftest.py` 才能为两个包加载。

## 许可证

[AGPL-3.0](LICENSE)。
