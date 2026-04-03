# AI Assistant (Voice & Vision Local System)

Sebuah asisten AI lokal modular yang dirancang untuk berinteraksi menggunakan Bahasa Indonesia. Sistem ini mengintegrasikan berbagai model AI secara efisien pada resource lokal (CPU & GPU) mulai dari pemrosesan suara, penglihatan (vision), hingga memori jangka panjang.

## 🌟 Fitur Utama

- **🧠 Pemrosesan Bahasa Alami (LLM):** Mengandalkan koneksi ke **LM Studio** lokal (dioptimalkan untuk GPU seperti Intel Arc via port 1234) untuk melakukan inferensi yang cepat dan aman tanpa harus mengirim data ke cloud.
- **🗣️ Speech-to-Text (STT):** Menggunakan **Faster-Whisper** (`base` model, `int8` compute) yang berjalan sangat ringan di CPU untuk transkripsi suara ke teks secara real-time dengan akurasi tinggi pada Bahasa Indonesia.
- **🔊 Text-to-Speech (TTS):** Menggunakan **Edge TTS** dari Microsoft untuk menghasilkan suara Bahasa Indonesia yang natural (`id-ID-GadisNeural`).
- **👁️ Vision Processing (Image to Text):** Terintegrasi dengan **Moondream2** (`vikhyatk/moondream2`) yang siap menjelaskan gambar dan melihat lingkungan (saat ini berjalan di CPU/iGPU).
- **📚 Memori Jangka Panjang (RAG):** Menggunakan **ChromaDB** sebagai vektor database untuk menyimpan fakta tentang pengguna. Sistem akan menarik konteks secara otomatis pada percakapan selanjutnya.
- **🔄 Sliding Window Memory:** Mampu mengingat 10 alur percakapan terakhir agar obrolan tetap relevan tanpa menghabiskan batasan *context window* pada LLM.

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
   *(Catatan: Anda mungkin perlu menginstal PyTorch versi spesifik tergantung pada hardware yang Anda gunakan seperti CUDA/ROCm/DirectML).*

4. **Siapkan LM Studio:**
   - Unduh dan jalankan [LM Studio](https://lmstudio.ai/).
   - Muat model bahasa (LLM) pilihan Anda.
   - Nyalakan **Local Inference Server** pada port `1234`.

## 🚀 Cara Menjalankan

Setelah semua konfigurasi siap, jalankan *core application*:

```bash
python main.py
```

Bicaralah secara langsung ke mikrofon. Asisten akan menanggapi ucapan Anda secara singkat, padat, dan santai menggunakan gaya bahasa Indonesia sehari-hari!

## ⚙️ Konfigurasi (`config.yaml`)

Anda dapat mengatur berbagai parameter sistem secara fleksibel di dalam file `config.yaml`. Beberapa pengaturan penting meliputi:
- URL & API Key untuk LLM (Default: `http://localhost:1234/v1`).
- Pengaturan perangkat dan tipe komputasi (`cpu`, `int8`, dll) untuk modul STT dan Vision.
- Pemilihan suara *Edge TTS* (`id-ID-GadisNeural` atau `id-ID-ArdiNeural`).
- Direktori penyimpanan untuk *Vector Database*.

## 📂 Struktur Direktori Utama

- `main.py` - Script utama pengendali alur (Router STT -> RAG -> LLM -> TTS).
- `config.yaml` - File konfigurasi global.
- `requirements.txt` - Daftar paket dan dependensi yang digunakan proyek.
- `modules/` - Direktori arsitektur micro-services:
  - `llm_client.py`: Integrasi streaming API Language Model.
  - `memory_engine.py`: ChromaDB Vector store untuk memori persisten.
  - `memory_rag.py`: (Opsional) Layer RAG untuk pemrosesan teks.
  - `router.py`: Routing *intent* atau tugas.
  - `stt_engine.py`: Transkripsi suara (Faster-Whisper).
  - `tts_engine.py`: Pembangkitan suara natural (Edge TTS).
  - `vision_engine.py`: Pemahaman gambar (Moondream2).
- `test_stt.py` - Script untuk mengetes fungsi mikrofon & rekaman.

---
*Didesain khusus untuk privasi tinggi, operasi lokal, dan respon percakapan natural.*
