# WikiRAG Tensura v2.1.0

## ภาษาไทย

### เพิ่มใหม่

- เพิ่มระบบตรวจสอบคำตอบรอบที่สอง โดยค่าเริ่มต้นปิดไว้
- เก็บคำตอบแรกไว้และแสดงผลตรวจสอบแยกจากคำตอบเดิม
- เพิ่มโหมด ปิด, ตรวจสอบอย่างเดียว, เสนอคำตอบใหม่ และแก้ไขอัตโนมัติ
- เพิ่มระดับความเข้มงวด เร็ว, สมดุล และละเอียด
- เปิด/ปิดการตรวจตัวเลข, ชื่อบุคคล, ระดับสกิล, ความสัมพันธ์, Citation และข้อมูลไม่มีหลักฐานแยกกัน
- ตรวจสอบจาก Sources ของ RAG เท่านั้น และไม่แก้ไขคำตอบต้นฉบับ
- เพิ่มแท็บผลตรวจสอบและคำตอบที่ปรับปรุงแล้วในหน้า Chat
- เพิ่ม API `POST /api/chat/verify`

### ปรับปรุงจาก v2.0.1

- เพิ่มรายละเอียดสถานะโมเดลและ Embedding ในหน้า Admin
- เพิ่ม Live Log, ปุ่มหยุดงาน, Backup และ Restore ล่าสุด
- เพิ่มการตรวจสอบความปลอดภัยของไฟล์ Backup ก่อน Restore
- เพิ่มการแสดงโมเดลสำรองเมื่อมีการสลับโมเดล
- เพิ่ม Citation ที่คลิกเปิด Entity ได้
- ปรับ Graph ให้จัดกลุ่มโหนดตามประเภทและส่งออก PNG/JSON ได้
- อัปเดต README ภาษาไทย/อังกฤษ และเอกสาร License/Notice

## English

### Added

- Optional second-pass answer verification, disabled by default
- Preserves the original answer and shows verification separately
- Modes for off, verify only, suggest a revised answer, and automatic revision
- Fast, balanced, and detailed verification strictness levels
- Independent checks for numbers, names, skill ranks, relationships, citations, and unsupported claims
- Verification uses only retrieved RAG sources and never overwrites the original answer
- Verification results and revised answers are displayed in the Chat UI
- Added `POST /api/chat/verify`

### Improved from v2.0.1

- Expanded Admin model and embedding status
- Added live logs, cooperative stop, backup, and latest-backup restore
- Added safe archive validation before restore
- Shows which fallback model was used
- Citations can open the related Entity
- Graph nodes are clustered by type and can be exported as PNG/JSON
- Updated Thai/English README and license/notice documentation
