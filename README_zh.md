[![GitHub release](https://img.shields.io/github/v/release/Oaklight/searxng-bm25-reranker?color=green)](https://github.com/Oaklight/searxng-bm25-reranker/releases/latest)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-≥3.10-blue.svg)](https://www.python.org/)
[![SearXNG](https://img.shields.io/badge/SearXNG-plugin-orange.svg)](https://github.com/searxng/searxng)
[![Docker](https://img.shields.io/docker/pulls/oaklight/searxng?label=Docker%20pulls)](https://hub.docker.com/r/oaklight/searxng)

[English](README.md) | **中文**

# SearXNG BM25 Reranker

一个 SearXNG 外部插件，使用 BM25 文本相关性评分对搜索结果进行重排序，提升搜索质量。

## 特性

- **BM25F 多字段加权** — 标题权重 2.0，正文权重 1.0，优先匹配标题命中的结果
- **RRF 融合排序** — 将 BM25 得分与引擎原始排名通过 Reciprocal Rank Fusion 融合，而非完全替代
- **CJK 分词支持** — 内置零依赖的中日韩文字分词器（unigram + bigram），无需 jieba
- **零外部依赖** — 核心 BM25 引擎来自 [zerodep/sparse_search](https://github.com/Oaklight/zerodep)，纯标准库实现
- **即插即用** — 标准 SearXNG 外部插件，Docker 镜像或 pip install 均可

## 工作原理

```
搜索请求 → 各引擎返回结果 → [post_search hook]
                                    ↓
                         构建临时 BM25F 索引（title + content）
                                    ↓
                         用原始查询做 BM25 检索，得到相关性得分
                                    ↓
                         RRF 融合（引擎排名 + BM25 排名，k=60）
                                    ↓
                         改写 positions 影响最终评分 → 重排后的结果
```

插件在 `post_search` 阶段介入，此时所有引擎结果已收集但尚未计算最终分数。通过改写每条结果的 `positions` 列表，影响 SearXNG 内置的 `calculate_score()` 公式（`weight / position`），实现非侵入式重排。

## 安装

### 方式一：预构建 Docker 镜像（推荐）

使用 Docker Hub 上的预构建镜像，跟踪 `searxng/searxng:latest` 并预装插件：

```yaml
services:
  searxng:
    image: oaklight/searxng:latest
    restart: always
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=https://search.example.com
```

### 方式二：Docker Compose 内联构建

在 compose 阶段从官方 SearXNG 镜像构建并安装插件：

```yaml
services:
  searxng:
    build:
      dockerfile_inline: |
        FROM searxng/searxng:latest
        RUN /usr/local/searxng/.venv/bin/python3 -m ensurepip && \
            /usr/local/searxng/.venv/bin/python3 -m pip install --no-cache-dir searxng-bm25-reranker && \
            /usr/local/searxng/.venv/bin/python3 -m pip uninstall -y pip
    restart: always
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=https://search.example.com
```

### 注册插件

以上两种方式均需在 `settings.yml` 中注册插件：

```yaml
plugins:
  searxng_bm25_reranker.SXNGPlugin:
    active: true
```

然后启动或重启容器：

```bash
docker compose up -d
```

## 配置

插件开箱即用，无需额外配置。以下参数均可在 `settings.yml` 中以插件 id 为键的顶层配置块按实例覆盖，取值须为不小于 1 的数字，`rrf_k` 还须为整数。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `title_weight` | `2.0` | 标题字段 BM25F 权重 |
| `content_weight` | `1.0` | 正文字段 BM25F 权重 |
| `rrf_k` | `20` | RRF 融合常数 |
| `engine_weight` | `1.0` | 引擎原始排名在融合中的权重 |
| `bm25_weight` | `1.0` | BM25 排名在融合中的权重 |

固定且不暴露：Okapi BM25，`k1=1.5`、`b=0.75`、`delta=0`。

`rrf_k` 默认为 20 而非 Cormack 等人的 60，因为 60 是在候选量达数千条的检索任务上调优的，
而 SearXNG 单页结果只有约 30 条。在 k=60 时第 1 名与第 30 名的差距被压缩到 1.48 倍，
相关性信号被削弱。取值远高于 20 会进一步压平排名差异，最终退化为引擎原始顺序。

```yaml
bm25_reranker:
  rrf_k: 10
  bm25_weight: 2.0
```

该配置块位于顶层、以插件 id 为键，这是 SearXNG 对插件私有选项的约定（内置的 hostnames 插件即如此）。
`PluginCfg` 只接受 `active`，把这些键放在 `plugins:` 下会让 SearXNG 启动失败，而不是回退到默认值。

调高 `bm25_weight` 可让文本相关性更强地压过引擎原始排序；调高 `engine_weight` 则相反。

### URL 优先级

`url_priority` 会对 URL 匹配正则的结果乘上一个系数，用来在不改排序代码的前提下
提升或压低某个站点。系数小于 `1.0` 是降权，大于 `1.0` 是加权，`0.0` 直接压到底部。
多条规则同时命中时会连乘。

该配置块放在 `settings.yml` 的**顶层**，以插件 id 作为键名。它不能放在 `plugins:`
下面，那一层只接受 `active`：

```yaml
bm25_reranker:
  url_priority:
    '(.*\.)?arxiv\.org$': 1.5
    '(.*\.)?github\.com$': 1.2
    '(.*\.)?pinterest\.com$': 0.1
    '(.*\.)?facebook\.com$': 0.0

plugins:
  searxng_bm25_reranker.SXNGPlugin:
    active: true
```

正则非法、系数不是数字或为负数时，该条目会被跳过并在日志里给出 warning。
YAML 的布尔值同样算"不是数字"，写 `false` 并不能关掉一条规则，只会被跳过。
写错配置只会退化成"没有这条系数"，不会让搜索失败。

正则使用 `re.search` 匹配，未锚定的规则会命中 URL 的任意位置，包括 query string。
想表达"整个域名"，请像上面的例子那样用 `$` 锚定。该配置在插件加载时读取，
修改后需要重启。

## 项目结构

```
src/searxng_bm25_reranker/
├── __init__.py          # SXNGPlugin 类，post_search 重排逻辑
├── _tokenizer.py        # CJK-aware 分词器（unigram + bigram）
└── _vendor/
    └── sparse_search.py # 来自 zerodep 的 BM25 引擎（vendored）

tests/
├── conftest.py          # SearXNG API stub（SearXNG 不是依赖）
└── test_url_priority.py
```

## 致谢

- [SearXNG](https://github.com/searxng/searxng) — 隐私友好的元搜索引擎
- [zerodep/sparse_search](https://github.com/Oaklight/zerodep) — 零依赖 BM25 全文搜索引擎

## 许可证

[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0)
