# INTEGRATION POINTS

> **STATUS: Tahap A (local-first) SELESAI.** Banyak "titik integrasi" di bawah kini sudah
> diimplementasi. Untuk gambaran arsitektur terkini lihat `ARCHITECTURE.md` & `MODULE_MAP.md`.
> Yang tersisa = **Tahap B (butuh hardware ESP32):**
> - `RobotIO` (WebSocket + HTTP audio server) sebagai `AudioSource`/`AudioSink` kedua.
> - `FeedbackSink` → kirim `state` ke ESP32 (wajah/cue) — kontraknya sudah ada (`LocalFeedbackSink` print).
> - Sensor (presence/imu/distance) → konteks LLM + perilaku reaktif.
> - Vision (ESP32-CAM) → hidupkan `vision_engine` / VLM.
>
> Sudah selesai di Tahap A: abstraksi I/O (`io_contracts`+`local_io`), Orchestrator, tool-calling
> (`tool_registry`), streaming TTS, memori (kerja/RAG/profil), episodic logging, feedback state (print).
>
> Bagian historis di bawah (rencana Fase 1/2) dipertahankan sebagai konteks.

Titik tempat kode Fase 2 (abstraksi I/O, WebSocket/HTTP server, tool-calling, feedback/state)
akan menyambung. Mengacu ke target arsitektur di `PLANNING_PC.md` dan protokol
`PLANNING_FIRMWARE_ESP32.md`.

> Belum ada perubahan kode. Ini peta "di sini nanti dicolok".

## 1. Abstraksi I/O (`AudioSource` / `AudioSink` / `FeedbackSink`)

Kontrak yang perlu diekstrak agar otak tidak peduli sumber suara.

| Kontrak | Implementasi lokal SAAT INI | Lokasi yang harus dibungkus |
|---------|------------------------------|------------------------------|
| `AudioSource` (hasilkan teks/audio user) | Mic PC | `STTManager.listen_and_transcribe()` — `stt_engine.py:30`. **Masalah:** capture mic & transkrip tergabung. Untuk RobotIO perlu dipisah: *capture* (mic/PCM-WebSocket) vs *transcribe* (faster-whisper menerima `np.float32`). Pemisahan alami ada di `stt_engine.py:50-59` (raw bytes → numpy → `model.transcribe`). |
| `AudioSink` (putar audio respons) | Speaker PC | **Fase 2:** sudah di balik `modules/tts_factory.py` `get_tts_manager()` — `main.py` panggil ini, bukan engine langsung. Engine aktif (`mms`/`f5_indo`/`edge`) dipilih via `config.yaml`. Ini cikal-bakal `AudioSink`; Langkah 1 tinggal memformalkan jadi interface. Untuk RobotIO: tambah engine/sink baru yang "serahkan audio ke HTTP server + kirim URL ke ESP32". |
| `FeedbackSink` (state/wajah) | Belum ada (print ke terminal) | Titik transisi di loop `main.py` (lihat §4). |

**Rekomendasi titik refactor:** ekstrak loop `main.py` `main()` (`main.py:99-206`) menjadi
`Orchestrator.run(session)` di mana `session` membawa `AudioSource`, `AudioSink`, `FeedbackSink`.
`LocalIO` = implementasi pertama (bungkus yang sudah jalan).

## 2. STT — adaptasi stream/chunk

- Fungsi transkrip inti: `STTManager.transcribe(audio_np)` di `stt_engine.py` (atau
  `OVSTTManager.transcribe` di `stt_ov.py`). Sudah menerima numpy array float32 16 kHz, jadi
  siap dipakai RobotIO (PCM dari WebSocket → numpy → transcribe). Backend dipilih lewat
  `stt_factory.get_stt_manager()` — RobotIO tidak perlu peduli CPU atau Arc.
- Yang mic-spesifik: `BaseSTT.capture()` di `stt_base.py` (`sr.Microphone`, `listen`). Ini hanya
  untuk LocalIO; RobotIO tidak memakainya.
- VAD: saat ini hanya `vad_filter=True` di whisper + silence-endpoint SpeechRecognition. Untuk
  RobotIO, endpoint datang dari sinyal `audio_end` ESP32 (lihat firmware Bagian 4).

## 3. LLM — tool-calling (otak)

- Titik tunggal: `llm_client.py:26` `stream_response(messages, model=...)`.
- Untuk tool-calling: tambah jalur yang mengirim `tools=[...]` dan menangani `tool_calls` di delta
  (saat ini delta hanya diambil `content` — `llm_client.py:69-72`). Perlu jalur non-stream atau
  parsing `tool_calls` streaming.
- ✅ Model **sudah dipindah ke config** (`llm.model`) di Fase 2 Langkah 0 — tinggal swap di `config.yaml`.
- System prompt: `main.py:14-24` (`DEFAULT_SYSTEM_PROMPT`). Tempat menaruh kepribadian + daftar tool.
- Streaming per-kalimat sudah tersedia (`llm_client.py` yield) tapi **belum dipakai untuk TTS
  overlap** (lihat §5).

## 4. Feedback / State (modul K)

Sisipkan emit `state` di transisi nyata di `main.py`:

| State | Titik sisip di `main.py` |
|-------|--------------------------|
| `listening` | sebelum `stt.listen_and_transcribe()` — `main.py:105` |
| `thinking` | setelah dapat teks, sebelum/saat LLM — `main.py:171` |
| `speaking` | sebelum `tts.stream_tts(...)` — `main.py:194` |
| `idle` | akhir iterasi loop — `main.py:199` |
| `confused` | saat `user_text` kosong (`main.py:107`) atau exception (`main.py:205`) |

Untuk LocalIO `FeedbackSink` bisa no-op/print; untuk RobotIO kirim `state` via WebSocket.

## 5. Streaming TTS per kalimat (latency)

- **Saat ini:** `main.py:177-184` mengumpulkan SEMUA chunk LLM → `full_response_text` → satu
  panggilan TTS (`main.py:194`). Tidak ada overlap.
- **Integrasi:** sambungkan generator `llm.stream_response()` langsung ke `tts.stream_tts()`
  (TTS sudah menerima iterator kalimat — `tts_engine.py:69`). Hati-hati: butuh menampung
  `full_response_text` juga untuk disimpan ke memori.

## 6. Memori & RAG

- Write fakta: `main.py:131-132` → `vectordb.save_fact()`. Heuristik keyword → ganti dengan
  ekstraksi LLM "apa yang layak diingat?" (jalur tulis modul G).
- Read RAG: `main.py:163` → `vectordb.search_context()`. Sudah jadi titik injeksi konteks.
- Profil terstruktur (SQLite/JSON) & episodic log (JSONL) = modul baru, belum ada hook.

## 7. Transport baru (modul A & E) — file baru, bukan edit

- `RobotIO` WebSocket server (PC ⇄ ESP32): terima `wake`/PCM/`audio_end`, kirim
  `play_tts`/`play_music`/`action`/`display`/`state`. Protokol = `PLANNING_FIRMWARE_ESP32.md` §4.
- HTTP audio server lokal: sajikan mp3 TTS & stream musik untuk ESP32 (modul E).
- Node ESP32-CAM terpisah → endpoint kamera khusus (modul I).

Semua ini **konsumen** dari kontrak `AudioSource`/`AudioSink` — tidak menyentuh inti otak.
