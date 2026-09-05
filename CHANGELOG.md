# WikiRAG Tensura v2.0.1

## ภาษาไทย

รุ่น v2.0.1 ปรับปรุงประสบการณ์ใช้งานเว็บ การสำรวจฐานความรู้ และความเสถียรของการเรียก LLM

### เพิ่มและปรับปรุง

- หน้า Entities แสดงจำนวนรายการแยกตามหมวด
- แบ่งการแสดงผลเป็นชุดและโหลดเพิ่มทีละ 60 รายการ
- กรองตาม Species, Rank, Status และ Affiliation
- แสดงเฉพาะรายการที่ไม่มีข้อมูล Rank ได้
- เรียงชื่อ A–Z, Z–A, ตามประเภท และตามระดับ
- แสดง Rank ของอาวุธ อุปกรณ์ และ Entity ที่มีข้อมูล
- Export รายการ Entities เป็น JSON หรือ CSV
- Graph กรองตามประเภท Entity และประเภทความสัมพันธ์
- ค้นหา Entity ใน Graph และโฟกัสไปยังผลการค้นหา
- คลิก node เพื่อเปิดรายละเอียด และคลิกเส้นเพื่อดูความสัมพันธ์
- ลาก node, เลื่อนพื้นที่, Zoom และ Reset มุมมอง Graph
- Graph หยุดการจัดตำแหน่งเมื่อกราฟนิ่ง และไม่รีเซ็ตเมื่อข้อมูลไม่เปลี่ยน
- Chat เลือกโมเดลจากรายการที่เตรียมไว้ได้
- เพิ่ม Retry อัตโนมัติสำหรับ OpenRouter 429/502/503/504 พร้อมปุ่มเปิด/ปิด
- รองรับ Fallback Model เมื่อโมเดลหลักใช้งานไม่ได้
- แสดงเวลาและจำนวน token โดยประมาณของคำตอบ
- Export แชทเป็น Markdown และล้างประวัติแชททั้งหมด
- เพิ่มปุ่มทดสอบการเชื่อมต่อ Provider/API
- Import และ Export การตั้งค่าโดยไม่รวม API key
- จดจำภาษา สี และหน้าล่าสุดหลังรีเฟรช
- รองรับ LM Studio ผ่าน OpenAI-compatible API อย่างถูกต้อง
- คงระบบ Deduplicate, Embedding cache, Crawl checkpoint/resume และการกรอง secret จาก settings/logs

### หมายเหตุ

- ไม่รวม API key, ข้อมูลที่ Crawl แล้ว, ฐานข้อมูลเวกเตอร์ หรือโมเดลขนาดใหญ่
- โมเดล Embedding BGE-M3 ONNX INT8 ต้องดาวน์โหลดแยกต่างหากตาม README
- การนับ token ในหน้าเว็บเป็นค่าประมาณสำหรับการตอบแบบ streaming

## English

Version 2.0.1 improves the web experience, knowledge-base exploration, and LLM reliability.

### Added and improved

- Per-category counts in the Entities browser
- Batched rendering with a Load More flow of 60 items
- Species, Rank, Status, and Affiliation filters
- A filter for entities with missing rank data
- Name A–Z, Z–A, type, and rank sorting
- Rank display for weapons, equipment, and other ranked entities
- JSON and CSV export for filtered Entities results
- Graph filters for entity types and relationship types
- Graph search with focus on matching entities
- Click nodes for details and click edges for relationship details
- Drag nodes, pan the canvas, zoom, and reset the Graph view
- Stable Graph layout that does not reset when refreshed data is unchanged
- Model presets in Chat settings
- Automatic retry for OpenRouter 429/502/503/504 responses with an on/off toggle
- Fallback model support when the primary model fails
- Response duration and estimated token count in the Chat header
- Markdown chat export and clear-all chat history actions
- Provider/API connectivity test button
- Settings import and export without API keys
- Persistent language, theme, and last active page preferences
- Correct LM Studio support through the OpenAI-compatible API
- Existing deduplication, embedding cache, crawl checkpoint/resume, and secret filtering retained

### Notes

- API keys, crawled data, vector databases, and large model files are not included
- The BGE-M3 ONNX INT8 embedding model must be downloaded separately as documented in the README
- Token counts shown in the web UI are estimates for streaming responses
