# AI Assistant (Voice & Vision Local System)

Sebuah asisten AI lokal modular yang dirancang untuk berinteraksi menggunakan Bahasa Indonesia. Sistem ini mengintegrasikan berbagai model AI secara efisien pada resource lokal (CPU & GPU) mulai dari pemrosesan suara, penglihatan (vision), hingga memori jangka panjang.

> **Status:** Tahap A (local-first) selesai — otak penuh berjalan dengan mic/speaker PC.
> Akselerasi Intel Arc B580 aktif (torch `+xpu` + OpenVINO GenAI); pemetaan device diputuskan
> dari hasil ukur, lihat [docs/ARCHITECTURE.md §8](docs/ARCHITECTURE.md).
> Tahap B (robot ESP32 via WebSocket) menyusul saat hardware siap. Detail arsitektur:
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) & [docs/MODULE_MAP.md](docs/MODULE_MAP.md).

## 🌟 Fitur Utama

- **🧠 LLM (LM Studio):** inferensi lokal via LM Studio (`:1234`), model-agnostik (ganti di `config.yaml`). Dukungan **tool-calling** & auto-matikan "thinking" (Qwen) agar cepat.
- **🗣️ STT (swappable):** `faster_whisper` (CTranslate2, CPU/int8, default) atau `openvino` (Whisper di Intel Arc). Dipilih di `config.yaml → stt.backend`. `capture()` & `transcribe()` terpisah (siap untuk sumber audio lain).
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

3. **Install PyTorch untuk hardware-mu — SEBELUM requirements:**

   Target repo ini adalah **Intel Arc B580 (XPU)**, dan build PyTorch dari PyPI itu CPU-only
   (tidak akan pernah melihat Arc), jadi torch dipasang dari index Intel:

   ```bash
   pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
       --index-url https://download.pytorch.org/whl/xpu
   ```

   Yang dibutuhkan hanya **driver Intel Graphics terbaru** — Intel Extension for PyTorch
   (IPEX) tidak diperlukan lagi karena dukungan XPU sudah menyatu di torch inti.
   Pemakai NVIDIA/AMD cukup ganti perintah ini dengan build CUDA/ROCm yang sesuai.

4. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Verifikasi hardware:**

   ```bash
   python scripts/check_hardware.py
   ```

   Menampilkan apakah torch melihat Arc (`xpu`), apakah OpenVINO melihat device `GPU`, dan
   apakah versi numpy masih dalam batas OpenVINO. Jalankan ulang tiap kali ganti driver/torch.

6. **Siapkan LM Studio:**
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
- STT: `backend` (`faster_whisper` / `openvino`), `model`/`local_dir`, `device`, `compute_type`,
  `cpu_threads`, `beam_size`, dan blok `openvino` (`model_dir`, `device`).
- TTS: `engine` (`mms` / `f5_indo` / `edge`) + setelan tiap engine (termasuk `device`).
- Memory: `persist_directory` (ChromaDB), `embedding_model`.

### Pemetaan hardware (hasil ukur di Arc B580 + Ryzen 5 8500G)

| Beban | Device terbaik | Angka (audio/kalimat uji yang sama) |
|-------|----------------|-------------------------------------|
| STT Whisper medium | **Arc (OpenVINO `GPU`)** | **435 ms (9.0× realtime)** vs 3603 ms faster-whisper CPU int8 (1.1×) — **8× lebih cepat**, teks identik |
| TTS MMS (VITS) | **CPU** | 780 ms/kalimat (5.6× realtime) vs 2900 ms di XPU |
| TTS F5 (cloning) | Arc (`auto`→`xpu`) | belum diukur (paket `f5-tts` opsional) |
| LLM | Arc via LM Studio | di luar proses Python |

Startup STT di GPU: ~12 s pada run pertama (kompilasi kernel), lalu **~2.5 s** karena hasil
kompilasi di-cache ke `data/ov_cache` (`stt.openvino.cache_dir`).

MMS-TTS **sengaja** dibiarkan di CPU: modelnya kecil dengan ratusan op berurutan, sehingga
ongkos dispatch kernel GPU (flat ~2.8 s per kalimat, tak peduli panjangnya) mengalahkan
komputasinya. Ukur sendiri dengan `python scripts/bench_tts.py` dan `python scripts/bench_stt.py`.

## 📂 Struktur Direktori Utama

- `main.py` - Perakit tipis: rakit modul → `LocalIO` → `Orchestrator` → `run()`.
- `config.yaml` - Konfigurasi global. · `reset_memory.py` - Factory reset memori.
- `modules/`:
  - `orchestrator.py`: otak (loop, memori, tool-loop, streaming, episodic).
  - `io_contracts.py` / `local_io.py`: kontrak I/O & implementasi mic/speaker PC.
  - `stt_factory.py` → `stt_engine.py` (Faster-Whisper/CPU) / `stt_ov.py` (OpenVINO/Arc),
    bagian mic bersama di `stt_base.py` · `llm_client.py` (LM Studio + tools).
  - `tts_factory.py` → `tts_mms.py` / `tts_f5.py` / `tts_engine.py` (Edge).
  - `tool_registry.py`: lapisan aksi (tools).
  - `memory_engine.py` (RAG) · `profile_store.py` (profil) · `episodic_log.py` (log).
  - `model_paths.py`: resolver model lokal → repo-id.
  - _Dead code (Tahap B / lama):_ `vision_engine.py`, `router.py`, `memory_rag.py`.
- `scripts/` - `check_hardware.py` (deteksi Arc/OpenVINO) · `bench_tts.py` · `bench_stt.py`.
- `docs/` - Dokumentasi arsitektur. · `models/`, `voices/` - aset lokal (lihat README masing-masing).

---

_Didesain khusus untuk privasi tinggi, operasi lokal, dan respon percakapan natural._
