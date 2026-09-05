# WikiRAG 🔮

[English version](README_EN.md)

WikiRAG คือแพลตฟอร์ม Retrieval-Augmented Generation (RAG) แบบ local-first สำหรับเว็บวิกิและฐานความรู้ ภายใน repository นี้มีการตั้งค่าสำหรับฐานความรู้ Tensura, เครื่องมือ CLI, เว็บแอป FastAPI, ระบบค้นหาหลายภาษา, หน้าสำรวจตัวละคร, โครงข่ายความสัมพันธ์ และระบบ LLM ที่เปลี่ยนผู้ให้บริการได้

## ภาพรวมการทำงาน

```text
MediaWiki API → Crawl → แยกวิเคราะห์ Wikitext/Infobox → แบ่ง Chunk ตาม Section
→ สร้าง Embedding ด้วย BGE-M3 ONNX INT8 → LanceDB → ค้นคืนข้อมูล → LLM ตอบพร้อมแหล่งอ้างอิง
```

หน้าเว็บรองรับคำถามภาษาไทยและภาษาอังกฤษ คำตอบจะอ้างอิงจากเนื้อหาวิกิที่ค้นพบ พร้อมลิงก์แหล่งข้อมูลและข้อความแสดงที่มาภายใต้สัญญาอนุญาต CC BY-SA

## สิ่งที่ต้องมี

- Python 3.10 ขึ้นไป (แนะนำ Python 3.11)
- RAM ว่างประมาณ 2–4 GB ในช่วงเริ่มต้นโมเดล
- อินเทอร์เน็ตสำหรับดาวน์โหลดแพ็กเกจและโมเดลครั้งแรก เว้นแต่มีไฟล์อยู่ในเครื่องแล้ว
- ผู้ให้บริการ LLM แบบเลือกใช้ได้: OpenRouter, Ollama, LM Studio, Google Gemini, OpenAI หรือ Anthropic Claude

## ติดตั้ง

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
# .venv\Scripts\activate.bat

pip install -e .
```

สำหรับการพัฒนาและการทดสอบ:

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## โมเดล Embedding

WikiRAG ใช้โมเดล BGE-M3 ONNX INT8 ขนาดเล็ก แทนโมเดล PyTorch ขนาดหลาย GB:

- โมเดล: [gpahal/bge-m3-onnx-int8](https://huggingface.co/gpahal/bge-m3-onnx-int8)
- ตำแหน่งไฟล์ในเครื่อง: `models/bge-m3-onnx/model_int8.onnx`
- ขนาดไฟล์ปัจจุบัน: ประมาณ 543 MB
- Runtime: ONNX Runtime
- จำนวนมิติของเวกเตอร์: 1024
- โมเดลต้นฉบับ: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)

ไฟล์โมเดลจะไม่ถูกรวมใน Git เนื่องจากมีขนาดเกินข้อจำกัดไฟล์ปกติ 100 MB ของ GitHub ให้ดาวน์โหลดจาก Hugging Face แล้ววางไว้ตามตำแหน่งด้านบน และตรวจสอบให้ tokenizer มีอยู่ใน Hugging Face cache ด้วย ระบบจะไม่ดาวน์โหลดโมเดล BGE-M3 ขนาดใหญ่แบบเงียบ ๆ ตอนเริ่มทำงาน

## การตั้งค่า

สร้างไฟล์ environment ในเครื่อง:

```bash
copy .env.example .env
```

การตั้งค่าเริ่มต้นของโปรเจกต์ Tensura ใช้ OpenRouter:

```env
DEFAULT_LLM_PROVIDER=openrouter
DEFAULT_LLM_MODEL=minimax/minimax-m3:free
OPENROUTER_API_KEY=your_key_here
```

ห้าม commit ไฟล์ `.env` หรือ API key จริง การตั้งค่าโปรเจกต์อยู่ที่ [`projects/tensura.yaml`](projects/tensura.yaml) โดยค่า Embedding เริ่มต้นคือ `onnx` และ `int8` ส่วน Reranker ขนาดใหญ่จะปิดไว้เป็นค่าเริ่มต้นเพื่อป้องกันการใช้ RAM สูงเกินไป

## ขั้นตอนนำเข้าข้อมูล

ให้รันคำสั่งจากโฟลเดอร์หลักของ repository:

```bash
# 1. Crawl บทความใน namespace-0
wikirag crawl --project tensura

# 2. แยกวิเคราะห์ Wikitext, Infobox และ Section
wikirag parse --project tensura

# 3. สร้าง Embedding และจัดเก็บลง LanceDB
wikirag embed --project tensura

# 4. ดูสถิติการนำเข้าข้อมูล
wikirag stats --project tensura

# 5. ดึงเฉพาะการเปลี่ยนแปลงล่าสุดจากวิกิ
wikirag sync --project tensura --incremental
```

Crawler มีระบบ checkpoint และสามารถทำงานต่อหลังหยุดกลางคันได้อย่างปลอดภัย ควรตรวจสอบสถิติทุกครั้งหลัง sync เนื่องจากข้อมูลที่สร้างขึ้นจะถูกเก็บไว้ในเครื่อง

## ถามตอบผ่าน CLI

```bash
# OpenRouter (ค่าเริ่มต้นของ Tensura)
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

กำหนด API key ของผู้ให้บริการที่ต้องการใน `.env` ก่อนใช้งาน ผู้ใช้ ChatGPT และ OpenAI API เป็นคนละระบบกัน โดย provider ของ OpenAI ต้องใช้ OpenAI API key

## LM Studio

LM Studio เปิด endpoint ภายในเครื่องที่เข้ากันได้กับ OpenAI ให้เปิด Local Server ใน LM Studio ก่อน แล้วตั้งค่าดังนี้:

```text
Provider: openai
Model: ชื่อโมเดลที่โหลดอยู่ใน LM Studio
API Base URL: http://localhost:1234/v1
API Key: lm-studio
```

ในหน้า Settings ของเว็บมีช่อง API Base URL สำหรับ endpoint ที่เข้ากันได้กับ OpenAI

## เว็บแอปพลิเคชัน

```bash
python -m wikirag serve --host 127.0.0.1 --port 8000
```

เปิดเว็บที่ [http://localhost:8000](http://localhost:8000)

เว็บแอปรองรับการแชทแบบ streaming, ปุ่มหยุด/ยกเลิก, แก้ไข/คัดลอก/สร้างคำตอบใหม่, ประวัติแชทถาวร, การตั้งค่าที่ไม่ใช่ความลับ, การเลือก provider, การดูแหล่งข้อมูลที่ค้นพบ, สำรวจตัวละคร, โครงข่ายความสัมพันธ์ที่อัปเดตอัตโนมัติ และแดชบอร์ดสถานะการนำเข้า/สร้าง Embedding

ประวัติแชทจะถูกเก็บไว้ในเครื่องที่ `data/tensura/chat_history.db` และไม่ควร commit ขึ้น repository สาธารณะ

## การประเมินผลและการทดสอบ

ชุดข้อมูลประเมินผลอยู่ที่ `eval/golden_qa.json`:

```bash
python -m pytest -q
wikirag eval --project tensura --golden eval/golden_qa.json
```

## Docker

ไฟล์ Docker ที่ให้มาจะรัน FastAPI service พร้อม container ของ Ollama:

```bash
docker compose up --build
```

โปรเจกต์ Tensura ใช้ OpenRouter เป็นค่าเริ่มต้น ดังนั้นให้กำหนด API key หรือเปลี่ยน provider ของ LLM ก่อนใช้งานผ่าน Docker

## สถาปัตยกรรม

```text
หน้า MediaWiki ดิบ
  → MediaWikiConnector พร้อม checkpoint และ incremental sync
  → WikitextParser + InfoboxExtractor
  → SectionAwareChunker พร้อม contextual headers
  → gpahal/bge-m3-onnx-int8 ผ่าน ONNX Runtime
  → LanceDB vector store
  → QueryPreprocessor และ RetrievalPipeline
  → GroundedAnswerGenerator
  → OpenRouter/Ollama/LM Studio/Gemini/OpenAI/Anthropic
```

## นโยบาย Repository และข้อมูล

ข้อมูลที่สร้างขึ้นควรเก็บไว้ในเครื่องและไม่ควร commit:

```text
data/tensura/raw/
data/tensura/parsed/
data/tensura/embeddings/
data/tensura/vectordb/
data/tensura/chat_history.db
data/tensura/graph.db
```

สำหรับ repository สาธารณะ ควรเผยแพร่เฉพาะ source code, การตั้งค่าโปรเจกต์, เอกสาร, tests และ fixture ตัวอย่างขนาดเล็ก ห้ามเผยแพร่ API key, ประวัติแชทส่วนตัว, log ส่วนตัว หรือดัชนีข้อมูลขนาดใหญ่

## License และการแสดงที่มา

โค้ดของ WikiRAG ตั้งใจให้ใช้สัญญาอนุญาต MIT ควรเพิ่มไฟล์ `LICENSE` ที่เหมาะสมก่อนเผยแพร่

เนื้อหาที่ดัดแปลงจาก [Tensura Wiki](https://tensura.fandom.com) เผยแพร่ภายใต้ **Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)** คำตอบที่สร้างขึ้นจะแสดงที่มาโดยอัตโนมัติ หากแจกจ่ายเนื้อหาที่ดัดแปลงควรเพิ่มไฟล์ `NOTICE.md` ที่ระบุที่มาอย่างชัดเจน

โมเดล Embedding คือ [gpahal/bge-m3-onnx-int8](https://huggingface.co/gpahal/bge-m3-onnx-int8) ซึ่ง model card ระบุสัญญาอนุญาต MIT ส่วนโมเดลต้นฉบับคือ [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) หากแจกจ่ายไฟล์โมเดล ต้องเก็บประกาศและเงื่อนไข license ที่เกี่ยวข้องไว้ด้วย
