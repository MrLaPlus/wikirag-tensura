# WikiRAG 🔮

[English version](README_EN.md)

## WikiRAG Tensura v2.1.0 — ภาษาไทย

WikiRAG คือระบบถามตอบฐานความรู้ Tensura แบบ local-first ใช้ RAG ค้นข้อมูลจาก Tensura Wiki แล้วตอบพร้อม Citation รองรับ OpenRouter, Ollama, LM Studio, Gemini, OpenAI และ Claude

ฟีเจอร์ v2.1.0 เพิ่มระบบตรวจคำตอบรอบที่สองแบบเปิด/ปิดได้ ค่าเริ่มต้นคือปิด ระบบจะเก็บคำตอบแรกไว้เสมอ และตรวจตัวเลข ชื่อบุคคล ระดับสกิล ความสัมพันธ์ Citation และข้อมูลที่ไม่มีหลักฐานได้แยกกัน โดยเลือกระดับ เร็ว / สมดุล / ละเอียด ได้

ดูรายการเปลี่ยนแปลงได้ที่ [CHANGELOG.md](CHANGELOG.md) และคู่มือภาษาอังกฤษที่ [README_EN.md](README_EN.md)

WikiRAG is a local-first Retrieval-Augmented Generation (RAG) platform for fandom and knowledge wikis. This repository contains the Tensura knowledge-base configuration, CLI tools, FastAPI web application, multilingual retrieval, entity browsing, a knowledge graph, and pluggable LLM providers.

**Current release: v2.1.0** · See [CHANGELOG.md](CHANGELOG.md)

v2.1.0 adds optional second-pass answer verification. It is disabled by default. When enabled, the first answer is preserved and the system can verify citations, names, numbers, skill ranks, relationships, and unsupported claims using a configurable fast, balanced, or detailed pass.

## Overview

```text
MediaWiki API → Crawl → Wikitext/Infobox parsing → Section chunking
→ BGE-M3 ONNX INT8 embeddings → LanceDB → Retrieval → LLM answer + citations
```

The interface supports Thai and English questions. Answers are grounded in retrieved wiki passages and include source links plus the required CC BY-SA attribution.

## Requirements

- Python 3.10 or newer (Python 3.11 recommended)
- Approximately 2–4 GB available RAM during model initialization
- Internet access for initial package/model downloads, unless dependencies are already local
- Optional LLM runtime/API: OpenRouter, Ollama, LM Studio, Google Gemini, OpenAI, or Anthropic Claude

## Installation

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
# .venv\Scripts\activate.bat

pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## Embedding model

WikiRAG uses the lightweight BGE-M3 ONNX INT8 model, not the full multi-GB PyTorch model:

- Model: [gpahal/bge-m3-onnx-int8](https://huggingface.co/gpahal/bge-m3-onnx-int8)
- Local file: `models/bge-m3-onnx/model_int8.onnx`
- Current file size: approximately 543 MB
- Runtime: ONNX Runtime
- Vector dimension: 1024
- Base model: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)

The model file is intentionally kept outside normal Git commits because it is larger than GitHub's 100 MB regular-file limit. Download it from Hugging Face and place it at the path above. The tokenizer must also be available in the local Hugging Face cache. Startup does not silently download the full BGE-M3 model.

## Configuration

Create a local environment file:

```bash
copy .env.example .env
```

For the default Tensura configuration, set OpenRouter:

```env
DEFAULT_LLM_PROVIDER=openrouter
DEFAULT_LLM_MODEL=minimax/minimax-m3:free
OPENROUTER_API_KEY=your_key_here
```

Never commit `.env` or any real API key. Project settings are in [`projects/tensura.yaml`](projects/tensura.yaml). The default embedding configuration uses `onnx` and `int8`; the large cross-encoder reranker is disabled by default to protect RAM.

## Ingestion workflow

Run these commands from the repository root.

```bash
# 1. Crawl namespace-0 wiki articles
wikirag crawl --project tensura

# 2. Parse Wikitext, Infoboxes, and sections
wikirag parse --project tensura

# 3. Create embeddings and index into LanceDB
wikirag embed --project tensura

# 4. View ingestion statistics
wikirag stats --project tensura

# 5. Fetch only later wiki changes
wikirag sync --project tensura --incremental
```

The crawler uses checkpoints and can be resumed safely after interruption. Review statistics after a sync because generated data is stored locally.

## CLI questions

```bash
# OpenRouter (Tensura project default)
wikirag query "ริมุรุ เทมเพสต์ มีสกิลและความสามารถอะไรบ้าง" --project tensura

# Ollama
wikirag query "Who is Rimuru Tempest?" --project tensura --llm ollama:llama3.1:8b

# Gemini
wikirag query "Explain Rimuru's relationship with Veldora" --project tensura --llm gemini:gemini-2.5-flash

# OpenAI
wikirag query "What is Rimuru's species?" --project tensura --llm openai:gpt-4o-mini

# Anthropic Claude
wikirag query "Summarize Veldora's role" --project tensura --llm anthropic:claude-3-5-sonnet-20241022
```

Set the relevant API key in `.env` before using a cloud provider. ChatGPT consumer accounts are not the same as the OpenAI API; the OpenAI provider requires an OpenAI API key.

## LM Studio

LM Studio exposes an OpenAI-compatible local endpoint. Start its local server, then configure:

```text
Provider: openai
Model: the model loaded in LM Studio
API Base URL: http://localhost:1234/v1
API Key: lm-studio
```

The web Settings panel includes an API Base URL field for OpenAI-compatible endpoints.

## Web application

```bash
python -m wikirag serve --host 127.0.0.1 --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

The web application includes streaming chat with stop/cancel, edit/copy/regenerate actions, persistent local chat history, non-secret settings, provider selection, retrieved-source transparency, entity browsing, an auto-refreshing knowledge graph, and an ingestion/embedding status dashboard.

Chat history is stored locally at `data/tensura/chat_history.db` and should not be committed to a public repository.

## Evaluation and tests

The evaluation dataset is in `eval/golden_qa.json`:

```bash
python -m pytest -q
wikirag eval --project tensura --golden eval/golden_qa.json
```

## Docker

The included Docker setup runs the FastAPI service with an Ollama container:

```bash
docker compose up --build
```

The default Tensura project uses OpenRouter, so configure its API key or change the project LLM provider when using Docker.

## Architecture

```text
Raw MediaWiki pages
  → MediaWikiConnector with checkpointing and incremental sync
  → WikitextParser + InfoboxExtractor
  → SectionAwareChunker with contextual headers
  → gpahal/bge-m3-onnx-int8 via ONNX Runtime
  → LanceDB vector store
  → QueryPreprocessor and RetrievalPipeline
  → GroundedAnswerGenerator
  → OpenRouter/Ollama/LM Studio/Gemini/OpenAI/Anthropic
```

## Repository and data policy

Generated data should normally stay local and should not be committed:

```text
data/tensura/raw/
data/tensura/parsed/
data/tensura/embeddings/
data/tensura/vectordb/
data/tensura/chat_history.db
data/tensura/graph.db
```

For a public repository, publish source code, project configuration, documentation, tests, and small synthetic/sample fixtures. Do not publish API keys, personal chat history, private logs, or large generated indexes.

## License and attribution

The WikiRAG application code is intended to use the MIT license. Add the appropriate `LICENSE` file before publishing.

Content derived from the [Tensura Wiki](https://tensura.fandom.com) is distributed under the **Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)** license. Generated answers include attribution automatically. Include a clear `NOTICE.md` when redistributing derived content.

The embedding model is [gpahal/bge-m3-onnx-int8](https://huggingface.co/gpahal/bge-m3-onnx-int8), whose model card identifies an MIT license. The base model is [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3). Preserve the applicable model license notices when distributing model files.
