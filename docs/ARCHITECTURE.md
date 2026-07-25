# ARCHITECTURE — AI Assistant Lokal (Jarvis local-first)

> Status: **Tahap A (local-first) selesai (Langkah 1–7).** Dokumen ini mencerminkan
> arsitektur saat ini. Tahap B (RobotIO/ESP32) belum ada — menunggu hardware.

## 1. Ringkasan

Voice assistant lokal: audio (mic PC) → STT → memori/RAG → LLM (+ tool-calling) → TTS → speaker PC.
**Otak terpisah dari sumber/tujuan suara** lewat kontrak I/O, sehingga robot (ESP32) nanti cukup
"dicolok" tanpa mengubah otak. Semua model lokal/offline kecuali LLM yang dilayani **LM Studio**.

Entry point: [`main.py`](../main.py) — perakit tipis. Dijalankan `python main.py`.

## 2. Diagram alur (per giliran)

```mermaid
flowchart TD
    MIC["🎙️ Mic PC"] --> SRC["LocalMicSource.read()\n(AudioSource)"]
    SRC --> STT["STTManager.transcribe()\nfaster-whisper (CPU/int8)"]
    STT --> TXT{teks kosong?}
    TXT -- ya --> IDLE["state: idle → ulang"]
    TXT -- tidak --> STOP{stop-word?\n(berhenti/keluar/..)}
    STOP -- ya --> SHUT["_shutdown: ringkas → memory.json\n+ filler 'rangkum dulu' → pamit"]
    STOP -- tidak --> TURN["_handle_turn"]

    subgraph TURN_DETAIL["_handle_turn (Orchestrator)"]
        FACT["fakta→vectordb + ekstraksi nama→profil"]
        APPEND["append user; kompresi bila >12 (filler)"]
        BUILD["working = messages + PROFIL + RAG"]
        LOOP["TOOL-LOOP: stream LLM (with tools)"]
        TOOLS{tool_calls?}
        EXEC["eksekusi tool → hasil ke working"]
        ANSWER["content → TTS streaming per kalimat\n(producer-consumer) + filler 'Hmm' paralel"]
        LOG["save memory.json + episodic log"]
    end

    TURN --> FACT --> APPEND --> BUILD --> LOOP --> TOOLS
    TOOLS -- ya --> EXEC --> LOOP
    TOOLS -- tidak --> ANSWER --> LOG
    ANSWER --> SINK["LocalSpeakerSink.speak_stream()\n(AudioSink)"] --> SPK["🔊 Speaker PC"]
    LLMSRV["LM Studio :1234\n(model dari config, disable_thinking)"] -. HTTP stream .- LOOP
```

## 3. Kontrak I/O (inti pemisahan otak ↔ perangkat)

[`modules/io_contracts.py`](../modules/io_contracts.py):
- `AudioSource.read() -> np.ndarray|None` — satu giliran audio user.
- `AudioSink.speak(text)` / `speak_stream(iter)` — keluarkan suara respons.
- `FeedbackSink.state(name)` — cue state (`listening/thinking/speaking/idle/confused`).

Implementasi lokal [`modules/local_io.py`](../modules/local_io.py): `LocalMicSource` (bungkus
`STTManager.capture`), `LocalSpeakerSink` (bungkus TTS), `LocalFeedbackSink` (print). Bundel `LocalIO`.
RobotIO (Tahap B) = implementasi lain dari kontrak yang sama.

## 4. Otak: Orchestrator + Session

[`modules/orchestrator.py`](../modules/orchestrator.py):
- `Session(io, messages)` — state satu sesi.
- `Orchestrator.run(session)` — loop: dengar → transcribe → stop? → `_handle_turn`.
- `_handle_turn` — fakta→RAG, profil, kompresi memori, **tool-loop streaming**, episodic log.
- Memori: `load_messages`/`save_messages`/`_compress_memory`/`_summarize_now`.
- `_stream_round` — producer (LLM→antrian) + consumer (TTS `speak_stream`) = streaming per kalimat.

## 5. Komponen

| Lapisan | Modul | Catatan |
|---------|-------|---------|
| STT | `stt_factory.py` → `stt_engine.py` (faster-whisper/CPU) / `stt_ov.py` (OpenVINO/Arc) | `capture()` (mic, di `stt_base.py`) + `transcribe(audio)` (reusable RobotIO); backend dipilih via `config.stt.backend`. |
| LLM | `llm_client.py` | streaming + `tools`/`tool_calls_out`; `disable_thinking` (Qwen); override `max_tokens` per panggilan. |
| TTS | `tts_factory.py` → `tts_mms` (default) / `tts_f5` (cloning) / `tts_engine` (Edge, legacy) | dipilih via `config.tts.engine`. |
| Tools | `tool_registry.py` | `get_waktu`, `cari_memori`, `ingat_profil`. Tool-loop di orchestrator (maks 3 hop). |
| Memori kerja | `messages[]` + `memory.json` | ringkasan + 10 turn terakhir; recursive summarization (>12). |
| Memori RAG | `memory_engine.py` (ChromaDB) | episodik/semantik per-relevansi. |
| Profil | `profile_store.py` (`data/profile.json`) | fakta stabil, **selalu disuntik**; tool `ingat_profil` + ekstraksi nama deterministik. |
| Episodic | `episodic_log.py` (`data/episodes.jsonl`) | log tiap giliran (ts, user, assistant, tools, latency). Extensible. |
| Path model | `model_paths.py` | folder lokal → fallback repo-id. |
| Reset | `reset_memory.py` | factory reset semua memori. |

## 6. Latensi (Langkah 4 + filler)

- **Streaming TTS per kalimat**: kalimat pertama diucapkan tanpa menunggu jawaban utuh; generasi
  LLM overlap dengan pemutaran (producer-consumer).
- **Filler paralel**: "Hmm" saat berpikir, "aku ingat itu" saat kompresi, "rangkum dulu" saat keluar.
- **Summary cepat**: kompresi/exit dipaksa 256 token + tanpa-thinking (~2× lebih cepat).

## 7. Konfigurasi & data

- `config.yaml` — LLM (url/model/disable_thinking), STT, TTS engine, vision, memory.
- Gitignored (data pribadi): `memory.json`, `data/` (vectordb, profile.json, episodes.jsonl),
  `voices/*.wav`, `models/` (bobot besar).
- Eksternal: **LM Studio** di `localhost:1234`. Engine TTS `mms`/`f5_indo` 100% offline.

## 8. Hardware mapping

Sejak migrasi ke torch `+xpu` + OpenVINO (branch `feature/intel-arc-xpu-openvino`), pemetaan
device **diputuskan berdasarkan pengukuran**, bukan asumsi "GPU selalu lebih cepat":

| Beban | Device | Alasan / angka (Arc B580 + Ryzen 5 8500G) |
|-------|--------|-------------------------------------------|
| LLM | Arc, via LM Studio `:1234` | di luar proses Python (keputusan tetap; lihat §9) |
| STT `openvino` (**default baru**) | Arc (`GPU`) | **435 ms vs 3603 ms** untuk audio 3.9 s (8× lebih cepat, teks identik); startup 12 s → ~2.5 s berkat `CACHE_DIR` |
| STT `faster_whisper` (fallback) | CPU | CTranslate2 tidak punya backend Intel GPU; 3603 ms (1.1× realtime) |
| TTS `mms` (VITS) | **CPU** | 780 ms/kalimat vs **2900 ms di XPU** — model kecil, ratusan op berurutan, ongkos dispatch kernel (flat ~2.8 s) menang atas komputasi |
| TTS `f5_indo` | Arc (`auto`→`xpu`) | model besar; di CPU ~90 dtk/kalimat (tidak praktis) |
| Embedding RAG | CPU (sentence-transformers) | beban sangat kecil; kandidat `TextEmbeddingPipeline` nanti bila perlu |

Prasyarat: torch build `+xpu` (bukan PyPI biasa) + driver Intel Graphics terbaru. IPEX **tidak**
diperlukan lagi. Cek dengan `python scripts/check_hardware.py`; ukur ulang dengan
`scripts/bench_tts.py` / `scripts/bench_stt.py` kalau ganti hardware atau driver.

- Tahap B (nanti): ESP32 sebagai `AudioSource`/`Sink` kedua; sensor; ESP32-CAM untuk vision
  (kandidat: `openvino_genai.VLMPipeline` di Arc, bukan menambah stack torch vision).

## 9. Kenapa LLM tetap di LM Studio (bukan `openvino_genai.LLMPipeline`)

`LLMPipeline` bisa menjalankan LLM langsung di Arc dalam proses ini, tapi ia tidak punya API
`tools` gaya OpenAI — chat template berisi tool schema dan parsing `tool_calls` harus ditulis
sendiri, padahal tool-loop di `orchestrator.py` sudah bergantung pada format itu. Ditambah VRAM
12 GB yang harus dibagi dengan Whisper + TTS. Jadi LM Studio dipertahankan; bila suatu saat
ingin lepas, jalur paling murah adalah server OpenAI-compatible (mis. OpenVINO Model Server),
bukan mengganti kontrak `llm_client`.
