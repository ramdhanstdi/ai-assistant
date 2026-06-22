# MODULE MAP

Tabel: file → tanggung jawab → dipanggil oleh siapa → status.

| File | Kelas / fungsi utama | Tanggung jawab | Dipanggil oleh | Status |
|------|----------------------|----------------|----------------|--------|
| `main.py` | `main()`, `load_memory()`, `save_memory()`, `summarize_now()` | Orchestrator loop: STT→memori→RAG→LLM→TTS. Manajemen `memory.json` & recursive summarization. | Entry point (`python main.py`) | **AKTIF** |
| `modules/stt_engine.py` | `STTManager.listen_and_transcribe()` | Rekam mic (SpeechRecognition, endpoint via silence) → transkrip faster-whisper (`cahya/faster-whisper-medium-id`, CPU/int8, lang=id). | `main.py` | **AKTIF** |
| `modules/llm_client.py` | `LLMClient.stream_response()` | Streaming chat completion ke LM Studio (OpenAI-compat). Yield potongan kalimat per tanda baca. Model dari `config.yaml → llm.model` (Fase 2; sebelumnya hardcoded). | `main.py` | **AKTIF** |
| `modules/tts_factory.py` | `get_tts_manager()` | Pilih engine TTS sesuai `config.yaml → tts.engine` (`mms`/`f5_indo`/`edge`). Lazy-import. | `main.py` | **AKTIF** (Fase 2) |
| `modules/tts_mms.py` | `MMSTTSManager.stream_tts()`, `speak_chunk()`, `pick_device()` | MMS-TTS (VITS) Bahasa Indonesia, lokal/offline. Device auto (Intel Arc xpu / CPU) → sounddevice. | `tts_factory.py` | **AKTIF / DEFAULT** (Fase 2) |
| `modules/tts_f5.py` | `F5TTSManager.stream_tts()`, `speak_chunk()` | F5-TTS finetune Indonesia, **voice cloning** dari sampel `voices/`. Best-effort (ckpt/vocab belum diverifikasi). | `tts_factory.py` (engine `f5_indo`) | Tersedia, perlu setup |
| `modules/tts_engine.py` | `EdgeTTSManager.stream_tts()`, `_generate_and_play()`, `speak_chunk()` | Sintesis Edge TTS → `temp_chunk.mp3` → pygame playback (blocking per chunk via `asyncio.run`). | `tts_factory.py` (engine `edge`), `test_stt.py` | **LEGACY / fallback** |
| `modules/memory_engine.py` | `VectorDBManager.save_fact()`, `search_context()` | Memori jangka panjang RAG. ChromaDB persisten (`data/vectordb`), koleksi `user_profile`, embedding `all-MiniLM-L6-v2`. | `main.py` | **AKTIF** |
| `modules/memory_rag.py` | `MemoryManager` | Sistem memori alternatif (short+long term), koleksi `personal_memory`. Embedding default ChromaDB. | — (tidak ada) | **DEAD / duplikat** |
| `modules/router.py` | `Router.identify_intent()` | Klasifikasi intent regex: `reset` / `vision` / `chat`. | — (tidak ada) | **DEAD** |
| `modules/vision_engine.py` | `VisionManager.capture_frame()`, `analyze_image()` | Webcam capture (OpenCV) + image-to-text Moondream2. | — (tidak ada) | **DEAD** |
| `test_stt.py` | `main()` | **Demo TTS** (bukan STT) — simulasi generator kalimat → Edge TTS. | Manual | Util / salah nama |
| `config.yaml` | — | Konfigurasi global (llm/stt/vision/tts/memory). | Semua modul `__init__` | **AKTIF** |
| `voices/README.md` | — | Panduan format sampel suara untuk voice cloning F5. | Manual | Doc (Fase 2) |
| `requirements.txt` | — | Daftar dependensi (**diperbarui Fase 2 Langkah 0**). | Setup | AKTIF |

## Catatan keterhubungan
- Orchestrator (`main.py`) merangkai: `stt_engine`, `llm_client`, `tts_factory` (→ salah satu `tts_mms`/`tts_f5`/`tts_engine`), `memory_engine`.
- Setiap modul membaca `config.yaml` sendiri-sendiri di `__init__` (tidak ada config object bersama).
- TTS kini di balik **factory**: `main.py` tak lagi import engine TTS langsung, cukup `get_tts_manager()`. Engine ditukar lewat `config.yaml → tts.engine` tanpa ubah kode (pola AudioSink yang akan diformalkan di Langkah 1).
- `memory_engine` dan `memory_rag` adalah dua implementasi memori yang **berbeda dan tidak kompatibel** (koleksi & API beda). Hanya `memory_engine` yang dipakai.
