# AI Assistant (Voice & Vision Local System)

Sebuah asisten AI lokal modular yang dirancang untuk berinteraksi menggunakan Bahasa Indonesia. Sistem ini mengintegrasikan berbagai model AI secara efisien pada resource lokal (CPU & GPU) mulai dari pemrosesan suara, penglihatan (vision), hingga memori jangka panjang.

> **Status:** Tahap A (local-first) selesai — otak penuh berjalan dengan mic/speaker PC.
> Tahap B (robot ESP32 via WebSocket) menyusul saat hardware siap. Detail arsitektur:
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) & [docs/MODULE_MAP.md](docs/MODULE_MAP.md).

## 🌟 Fitur Utama

- **🧠 LLM (LM Studio):** inferensi lokal via LM Studio (`:1234`), model-agnostik (ganti di `config.yaml`). Dukungan **tool-calling** & auto-matikan "thinking" (Qwen) agar cepat.
- **🗣️ STT (Faster-Whisper):** `cahya/faster-whisper-medium-id`, CPU/int8. `capture()` & `transcribe()` terpisah (siap untuk sumber audio lain).
- **🔊 TTS lokal (swappable):** `mms` (MMS-TTS Indonesia, default, 100% offline), `f5_indo` (voice cloning), `edge` (legacy cloud). Dipilih di `config.yaml → tts.engine`.
- **🎭 Orchestrator + kontrak I/O:** otak (STT→memori→LLM→TTS) terpisah dari sumber/tujuan suara lewat `AudioSource`/`AudioSink`/`FeedbackSink` — robot tinggal "dicolok" nanti.
- **⚡ Streaming & filler:** jawaban diucapkan **per kalimat** sambil LLM lanjut generate; filler suara ("Hmm") menutup jeda.
- **🛠️ Lapisan aksi (tools):** `get_waktu`, `cari_memori`, `ingat_profil` (plug-in, mudah ditambah).
- **📚 Memori berlapis:** kerja (ringkasan + recursive summarization), **RAG** (ChromaDB), **profil terstruktur** (selalu diingat), dan **episodic log** (JSONL tiap giliran).
- **🧹 Reset:** `python reset_memory.py` untuk factory reset semua memori.

## 🛠️ Prasyarat & Instalasi

Pastikan **Python 3.10+** (atau versi lebih baru) telah terinstal di sistem Anda.

1. **Clone repository ini:**

   ```bash
   git clone https://github.com/ramdhanstdi/ai-assistant.git
   cd ai-assistant
   ```

2. **Buat dan Aktifkan Virtual Environment (Disarankan):**

   ```bash
   python -m venv venv
   # Di Windows:
   venv\Scripts\activate
   # Di Linux/Mac:
   source venv/bin/activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   _(Catatan: Anda mungkin perlu menginstal PyTorch versi spesifik tergantung pada hardware yang Anda gunakan seperti CUDA/ROCm/DirectML)._

4. **Siapkan LM Studio:**
   - Unduh dan jalankan [LM Studio](https://lmstudio.ai/).
   - Muat model bahasa (LLM) pilihan Anda.
   - Nyalakan **Local Inference Server** pada port `1234`.

## 🚀 Cara Menjalankan

Setelah semua konfigurasi siap, jalankan _core application_:

```bash
python main.py
```

Bicaralah secara langsung ke mikrofon. Asisten akan menanggapi ucapan Anda secara singkat, padat, dan santai menggunakan gaya bahasa Indonesia sehari-hari!

## ⚙️ Konfigurasi (`config.yaml`)

Anda dapat mengatur berbagai parameter sistem secara fleksibel di dalam file `config.yaml`. Beberapa pengaturan penting meliputi:

- LLM: `api_base_url`, `model` (sesuai yang dimuat di LM Studio), `disable_thinking`.
- STT: `model`/`local_dir`, `device`, `compute_type`, `cpu_threads`, `beam_size`.
- TTS: `engine` (`mms` / `f5_indo` / `edge`) + setelan tiap engine.
- Memory: `persist_directory` (ChromaDB), `embedding_model`.

## 📂 Struktur Direktori Utama

- `main.py` - Perakit tipis: rakit modul → `LocalIO` → `Orchestrator` → `run()`.
- `config.yaml` - Konfigurasi global. · `reset_memory.py` - Factory reset memori.
- `modules/`:
  - `orchestrator.py`: otak (loop, memori, tool-loop, streaming, episodic).
  - `io_contracts.py` / `local_io.py`: kontrak I/O & implementasi mic/speaker PC.
  - `stt_engine.py` (Faster-Whisper) · `llm_client.py` (LM Studio + tools).
  - `tts_factory.py` → `tts_mms.py` / `tts_f5.py` / `tts_engine.py` (Edge).
  - `tool_registry.py`: lapisan aksi (tools).
  - `memory_engine.py` (RAG) · `profile_store.py` (profil) · `episodic_log.py` (log).
  - `model_paths.py`: resolver model lokal → repo-id.
  - _Dead code (Tahap B / lama):_ `vision_engine.py`, `router.py`, `memory_rag.py`.
- `docs/` - Dokumentasi arsitektur. · `models/`, `voices/` - aset lokal (lihat README masing-masing).

---

_Didesain khusus untuk privasi tinggi, operasi lokal, dan respon percakapan natural._
