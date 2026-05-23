# MEMS&NEMS RAG

一个面向论文问答的多模态 RAG 项目。使用由 LMStudio 或者其他部署方案本地部署的问答模型，以及基于 HuggingFace transformer 部署的Embedding 和 Reranker 模型。系统会读取 MinerU 从 PDF 解析出的 `*_content_list.json` 中间文件，抽取论文文本和带 caption 的图片，使用本地 `qwen3_vl_embedding` 生成文本/图片向量，使用本地 `qwen3_vl_reranker` 对检索结果重排，最后调用 OpenAI 兼容接口生成带来源引用的回答。

## 功能

- 解析 MinerU `*_content_list.json`，过滤页眉、页脚、页码等噪声。
- 将论文正文切分后写入 ChromaDB 向量库。
- 对图片 caption 和图片内容做多模态向量化，并在回答时把相关图片传给 LLM。
- 使用 `qwen3_vl_reranker` 对召回的文本和图片重新排序。
- 提供网页聊天界面，支持历史会话、流式回答、更新/重构向量库。
- 保留命令行交互模式。

## 项目结构

```text
.
├── rag_backend.py                 # Web 服务、RAG 流程、向量库构建和问答入口
├── docuement_parser.py            # MinerU content_list JSON 清洗与分块
├── script/
│   ├── qwen3_vl_embedding.py      # qwen3_vl_embedding 加载与向量生成
│   └── qwen3_vl_reranker.py       # qwen3_vl_reranker 加载与重排打分
├── frontend/
│   ├── index.html                 # Web 问答界面
│   └── login.html                 # 可选访问验证界面
├── json/                          # 放置 MinerU 生成的 *_content_list.json
├── images/                        # 放置论文对应图片
└── README.md
```

## 环境准备

建议使用 Python 3.10+。项目没有固定的 `requirements.txt`，核心依赖包括：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

python -m pip install \
  chromadb \
  langchain-text-splitters \
  numpy \
  openai \
  pillow \
  python-dotenv \
  qwen-vl-utils \
  scipy \
  transformers
```

PyTorch 请按你的 CUDA 环境安装对应版本。当前代码在加载模型时使用了 `flash_attention_2` 和 `bfloat16`，推荐使用支持 FlashAttention 2 的 NVIDIA GPU 环境。如果只有一张 GPU，可以把 Embedding 和 Reranker 都配置到 `cuda:0`。

## 下载模型

本项目不会自动下载模型。请自行从 ModelScope 或 Hugging Face 下载以下两个模型，并放到项目目录下，例如：

```text
models/
├── qwen3_vl_embedding/
└── qwen3_vl_reranker/
```

示例命令：

```bash
# Hugging Face，替换为实际 repo id
huggingface-cli download <qwen3_vl_embedding_repo_id> --local-dir ./models/qwen3_vl_embedding
huggingface-cli download <qwen3_vl_reranker_repo_id> --local-dir ./models/qwen3_vl_reranker

# ModelScope，替换为实际 model id
modelscope download --model <qwen3_vl_embedding_model_id> --local_dir ./models/qwen3_vl_embedding
modelscope download --model <qwen3_vl_reranker_model_id> --local_dir ./models/qwen3_vl_reranker
```

下载后的目录需要能被 Transformers 的 `from_pretrained()` 直接加载，通常应包含 `config.json`、tokenizer/processor 相关文件和权重文件。

## 准备论文数据

1. 使用 MinerU 将论文 PDF 转为 Markdown 解析中间文件。
2. 将生成的 `*_content_list.json` 放入项目的 `json/` 文件夹。
3. 将对应论文图片放入项目的 `images/` 文件夹。
4. 确认 JSON 中每个图片条目的 `img_path` 能在本项目运行目录下正确访问。

推荐布局：

```text
json/
├── paper_a_content_list.json
└── paper_b_content_list.json

images/
├── paper_a/
│   ├── image_0.png
│   └── image_1.png
└── paper_b/
    └── image_0.png
```

注意：代码会直接读取 MinerU JSON 里的 `img_path` 字段。如果 JSON 里写的是 `images/paper_a/image_0.png`，请从项目根目录运行服务；如果路径不一致，需要先调整 JSON 中的图片路径或移动图片到对应位置。

## 配置环境变量

项目启动时会优先读取 `RAG_ENV_FILE` 指定的 env 文件；未指定时读取项目根目录的 `.env`。请在 `.env` 中配置：

```bash
# 本地数据与持久化路径
RAG_JSON_DIR=./json
RAG_IMAGE_DIR=./images
RAG_CHROMA_PATH=./chroma_db
RAG_HISTORY_PATH=./rag_history.json
RAG_INDEX_STATE_PATH=./rag_index_state.json

# 本地模型路径
RAG_EMBEDDING_MODEL_PATH=./models/qwen3_vl_embedding
RAG_RERANKER_MODEL_PATH=./models/qwen3_vl_reranker
RAG_EMBEDDING_DEVICE=cuda:0
RAG_RERANKER_DEVICE=cuda:1

# OpenAI 兼容 LLM 接口
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
RAG_LLM_MODEL=your_multimodal_llm_model
# 可选：不设置则复用 RAG_LLM_MODEL
# RAG_QUESTION_REWRITE_MODEL=your_question_rewrite_model
RAG_QUESTION_REWRITE_TIMEOUT_SECONDS=30

# Web 服务
RAG_WEB_HOST=127.0.0.1
RAG_WEB_PORT=7860
RAG_WEB_API_KEY=your_web_login_key
RAG_WEB_SESSION_HOURS=24
RAG_MAX_CONTEXT_MESSAGES=8
```

说明：

- `RAG_JSON_DIR`、`RAG_IMAGE_DIR`、`RAG_CHROMA_PATH`、`RAG_HISTORY_PATH` 是必需配置。
- `RAG_EMBEDDING_MODEL_PATH` 和 `RAG_RERANKER_MODEL_PATH` 必须指向已经下载好的本地模型目录。
- 如果只在本机使用，`RAG_WEB_HOST=127.0.0.1` 即可；需要局域网访问时可用启动参数 `--lan`。
- `RAG_WEB_API_KEY` 用于网页登录验证；不设置则网页不启用访问验证。
- `RAG_QUESTION_REWRITE_MODEL` 用于确认前的问题改写；不设置时默认复用 `RAG_LLM_MODEL`。`RAG_QUESTION_REWRITE_TIMEOUT_SECONDS` 控制改写请求超时时间。
- 代码也兼容旧变量名 `API_KEY` / `API_URL`，但推荐使用 `OPENAI_API_KEY` / `OPENAI_BASE_URL`。
- 如果希望真实 RAG 初始化失败时仍进入前端 Demo 模式，可设置 `RAG_FALLBACK_DEMO=1`。

## 启动与使用

启动网页服务：

```bash
python rag_backend.py
```

默认访问：

```text
http://127.0.0.1:7860
```

首次使用时，在网页左侧点击“更新向量库”或“重构向量库”：

- 更新向量库：只处理新增或发生变化的 JSON，并清理已删除文档的旧向量。
- 重构向量库：清空现有 ChromaDB collection 后重新处理全部 JSON。

允许局域网访问：

```bash
python rag_backend.py --lan
```

指定端口：

```bash
python rag_backend.py --port 8000
```

测试前端 Demo 模式：

```bash
python rag_backend.py --demo
```

命令行交互模式：

```bash
python rag_backend.py --cli
```

在 CLI 菜单中：

```text
1. 从 JSON 文件夹构建/更新向量数据库
2. 进行问答查询
3. 退出系统
```

## 检索与回答流程

1. `docuement_parser.py` 读取 `*_content_list.json`，抽取正文与带 caption 的图片。
2. 文本按段落语义切分为 chunk。
3. `qwen3_vl_embedding` 分别生成文本 chunk 和图片记录的向量。
4. 向量写入 ChromaDB 的 `text_collection` 和 `image_collection`。
5. 用户提问时，系统先用问题向量召回候选文本和图片。
6. `qwen3_vl_reranker` 对候选结果重排。
7. 系统把重排后的文本、图片 caption 和图片内容发给 LLM。
8. LLM 按提示生成 Markdown 回答，并显式引用来源文档名或图片页码。

## 常见问题

**启动时报 `expected str, bytes or os.PathLike object, not NoneType`**

通常是 `.env` 里缺少 `RAG_JSON_DIR`、`RAG_IMAGE_DIR`、`RAG_CHROMA_PATH` 或 `RAG_HISTORY_PATH`。这些路径变量在模块加载时就会读取，需要先配置好。

**图片处理失败或回答里没有图片**

检查 MinerU JSON 中的 `img_path` 是否能从项目根目录访问。项目不会自动把 `RAG_IMAGE_DIR` 拼到 `img_path` 前面。

**模型加载失败**

确认 `RAG_EMBEDDING_MODEL_PATH` 和 `RAG_RERANKER_MODEL_PATH` 指向本地模型目录，并且目录能被 `transformers.from_pretrained()` 加载。还需要确认 PyTorch、CUDA、FlashAttention 2 与显卡环境匹配。

**只有一张 GPU**

把两个设备都设为同一张卡：

```bash
RAG_EMBEDDING_DEVICE=cuda:0
RAG_RERANKER_DEVICE=cuda:0
```

**想清空并重新生成全部向量**

在网页点击“重构向量库”，或删除 `RAG_CHROMA_PATH` 指向的向量库目录后重新启动并更新。
