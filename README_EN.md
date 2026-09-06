# WikiRAG 🔮

**Current release: v2.1.0** · See [CHANGELOG.md](CHANGELOG.md)

WikiRAG is a local-first Retrieval-Augmented Generation platform for the Tensura knowledge base. It provides a FastAPI web app, multilingual retrieval, entity exploration, a knowledge graph, persistent chat history, and pluggable LLM providers including OpenRouter, Ollama, LM Studio, Gemini, OpenAI, and Claude.

## Highlights

- BGE-M3 ONNX INT8 embeddings (approximately 543 MB, 1024 dimensions)
- Streaming grounded chat with citations, edit, retry, regenerate, cancel, export, and fallback models
- Optional second-pass answer verification, disabled by default
- Verification modes: off, verify only, suggest a revised answer, or auto-revise
- Verification strictness: fast, balanced, or detailed
- Independent checks for numbers, names, skill ranks, relationships, citations, and unsupported claims
- Entity browsing with categories, filters, rank sorting, pagination, exports, aliases, details, and relationships
- Interactive graph with filters, hover highlighting, labels toggle, clustering by type, zoom, pan, PNG/JSON export, and node limits
- Admin progress, live logs, cooperative cancellation, model status, and backup/restore

## Quick start

```bash
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
python -m wikirag serve --host 127.0.0.1 --port 8000
```

Open http://localhost:8000. Configure the LLM provider in the web Settings panel. Never commit `.env`, API keys, chat history, crawled data, vector databases, or model files.

## Embedding model

Download [gpahal/bge-m3-onnx-int8](https://huggingface.co/gpahal/bge-m3-onnx-int8) and place the model at `models/bge-m3-onnx/model_int8.onnx`. The full multi-GB BGE-M3 model is not required. See [README.md](README.md) for the complete Thai setup and ingestion guide.

## License and attribution

Application code: MIT. Derived Tensura Wiki content: CC BY-SA 3.0. See [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).
## Optional API protection

For deployments outside the local machine, enable API protection in `.env`:

```env
WIKIRAG_PROTECT_API=1
WIKIRAG_API_TOKEN=replace-with-a-long-random-token
WIKIRAG_CORS_ORIGINS=https://your-frontend.example
```

Clients must send the token in the `X-WikiRAG-API-Token` header. This is a deployment safeguard, not a full user-login system; keep the server behind HTTPS and never commit `.env`.
