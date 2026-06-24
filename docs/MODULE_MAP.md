# MODULE MAP

Status: **Tahap A selesai (Langkah 1–7).** File → tanggung jawab → dipanggil oleh → status.

| File | Inti | Tanggung jawab | Dipanggil oleh | Status |
|------|------|----------------|----------------|--------|
| `main.py` | `main()` | Perakit tipis: load modul → `LocalIO` → `Orchestrator` → `Session` → `run()`. Berisi `DEFAULT_SYSTEM_PROMPT` + guard env (KMP, UTF-8). | entry point | **AKTIF** |
| `modules/orchestrator.py` | `Orchestrator`, `Session` | Otak: loop, memori (load/save/compress/summarize), tool-loop, streaming TTS, profil, episodic. | `main.py` | **AKTIF** |
| `modules/io_contracts.py` | `AudioSource`/`AudioSink`/`FeedbackSink` | Kontrak I/O (ABC) pemisah otak ↔ perangkat. | `local_io`, orchestrator | **AKTIF** |
| `modules/local_io.py` | `LocalMicSource`/`LocalSpeakerSink`/`LocalFeedbackSink`/`LocalIO` | Implementasi kontrak utk mic/speaker PC (bungkus STT/TTS). | `main.py` | **AKTIF** |
| `modules/stt_engine.py` | `STTManager.capture()/transcribe()` | Rekam mic + transkrip faster-whisper. `transcribe(audio)` reusable utk RobotIO. | `local_io`, orchestrator | **AKTIF** |
| `modules/llm_client.py` | `LLMClient.stream_response()` | Streaming LM Studio + tool-calling + `disable_thinking` + override `max_tokens`. | orchestrator | **AKTIF** |
| `modules/tts_factory.py` | `get_tts_manager()` | Pilih engine TTS via `config.tts.engine`. | `main.py` | **AKTIF** |
| `modules/tts_mms.py` | `MMSTTSManager` | MMS-TTS Indonesia lokal (default). | `tts_factory` | **AKTIF / DEFAULT** |
| `modules/tts_f5.py` | `F5TTSManager` | F5-TTS voice cloning (fix KMP/pyarrow/soundfile utk Windows). | `tts_factory` (`f5_indo`) | Tersedia (lambat CPU) |
| `modules/tts_engine.py` | `EdgeTTSManager` | Edge TTS (cloud). | `tts_factory` (`edge`) | **LEGACY** |
| `modules/tool_registry.py` | `ToolRegistry`, `build_default_registry` | Lapisan aksi: `get_waktu`, `cari_memori`, `ingat_profil`. | orchestrator | **AKTIF** |
| `modules/memory_engine.py` | `VectorDBManager` | RAG jangka panjang (ChromaDB + embedding lokal). | orchestrator | **AKTIF** |
| `modules/profile_store.py` | `ProfileStore` | Profil terstruktur (`data/profile.json`), selalu disuntik. | orchestrator, tools | **AKTIF** |
| `modules/episodic_log.py` | `EpisodicLogger` | Log episodik JSONL (`data/episodes.jsonl`), extensible. | orchestrator | **AKTIF** |
| `modules/model_paths.py` | `resolve()` | Folder model lokal → fallback repo-id. | stt/mms/memory | **AKTIF** |
| `reset_memory.py` | `main()` | Factory reset: hapus memory.json + data/(profile, vectordb, episodes). | manual | **AKTIF** |
| `config.yaml` | — | Konfigurasi global. | semua `__init__` | **AKTIF** |
| `modules/router.py` | `Router` | Intent regex. | — | **DEAD** |
| `modules/vision_engine.py` | `VisionManager` | Moondream2 + webcam. | — | **DEAD** (Tahap B) |
| `modules/memory_rag.py` | `MemoryManager` | Memori alternatif (duplikat). | — | **DEAD** |
| `test_stt.py` | `main()` | Demo TTS (salah nama). | manual | Util |

## Catatan keterhubungan
- Aliran utama: `main.py` → `Orchestrator.run(Session(LocalIO, messages))`. Otak tak menyentuh
  mic/TTS langsung — semua lewat `session.io` (kontrak). RobotIO Tahap B = colok implementasi baru.
- Tiga modul **DEAD** (`router`, `vision_engine`, `memory_rag`) belum dipakai; `vision_engine`
  kandidat dihidupkan di Tahap B (vision via ESP32-CAM).
