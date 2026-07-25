# RISKS — bagian rapuh, kode mati, dependensi usang

Ditemukan saat Fase Discovery. Diurut kira-kira dari paling berdampak.

> **Pembaruan Fase 2 (branch `feature/orchestrator-local-first`):** sebagian risiko sudah
> ditangani saat Langkah 0 + swap TTS lokal. Item yang beres ditandai **✅ RESOLVED** dengan
> tanggal/cara perbaikannya; temuan asli tetap dicatat sebagai konteks historis.

> **Pembaruan Tahap A selesai (Langkah 1–7):** tambahan yang sudah beres —
> - ✅ Loop dipindah dari `main.py` ke `Orchestrator` (tak lagi monolitik); `main.py` tipis.
> - ✅ Streaming LLM→TTS per kalimat (latensi); filler paralel saat berpikir/kompresi/exit.
> - ✅ Model LLM/STT/embedding lewat config + resolver folder lokal.
> - ✅ Kompatibilitas template Qwen: ringkasan dikemas 1 pesan `user`; tool-call omit `content=""`.
> - ✅ Memori: profil terstruktur + episodic log + `reset_memory.py`.
> - ⚠️ Masih: keyword fact-save kasar; RAG top-2 tanpa ambang; `id=time.time()` bisa tabrakan;
>   pin versi sebagian masih lama; vision/router/memory_rag masih dead code.

## A. Dependensi (`requirements.txt`) tidak sinkron dengan kode — ✅ RESOLVED (Fase 2, Langkah 0)
`requirements.txt` sudah dirombak agar sesuai import nyata: ditambah `PyAudio`, `numpy`,
`sentence-transformers`, `soundfile`, `scipy`, `sounddevice`, `transformers`, `torch`; `edge-tts`
& `pygame` jadi opsional (engine `edge`); dependensi vision dijadikan opsional (komentar).

Temuan asli (historis):
- **HILANG (dipakai tapi tidak terdaftar):** `edge-tts`, `pygame`, `sentence-transformers`,
  `numpy`, `PyAudio`.
- **TERDAFTAR tapi tidak dipakai:** `piper-tts`; deps vision berat untuk `vision_engine.py` (dead code).
- **Versi pin tua:** `chromadb==0.4.24`, `requests==2.31.0`, `SpeechRecognition==3.10.1`.
  ⚠️ **Masih perlu dicek**: pin tua belum di-update (potensi konflik), sengaja dibiarkan agar
  perilaku stabil dulu.

## B. Konfigurasi diabaikan / hardcoded — ✅ sebagian RESOLVED (Fase 2, Langkah 0)
- ✅ `stt.model` kini dibaca dari `config.yaml` (`stt_engine.py`), tak lagi hardcode.
- ✅ LLM model kini dari `config.yaml` (`llm.model`), default arg `stream_response` jadi `None`.
- ⚠️ `config.yaml: vision.*` masih ada tapi belum dirangkai (vision masih dead code → §C).

## C. Kode mati / membingungkan
- `modules/router.py`, `modules/vision_engine.py`, `modules/memory_rag.py` — tidak dipanggil
  di mana pun. `memory_rag.MemoryManager` (koleksi `personal_memory`) bertabrakan secara konsep
  dengan `memory_engine.VectorDBManager` (koleksi `user_profile`) yang aktif.
- `test_stt.py` salah nama: isinya tes **TTS**, bukan STT. Tidak ada test STT/LLM/memori nyata.
- README menyebut Moondream2/vision & router "terintegrasi" — **belum** di `main.py`.

## D. Latensi & arsitektur loop
- **Tidak ada overlap STT→LLM→TTS.** `main.py:177-194` menunggu LLM selesai penuh sebelum TTS
  mulai, padahal `stream_response` & `stream_tts` sudah mendukung per-kalimat. Latensi terasa.
- `adjust_for_ambient_noise(duration=0.5)` dipanggil **tiap iterasi** (`stt_engine.py:39`) →
  +0.5s tetap tiap giliran.
- `EdgeTTSManager` membuat **event loop baru** lewat `asyncio.run` per chunk (`tts_engine.py:80`).
  Boros & bisa bentrok jika dipindah ke konteks async (server WebSocket nanti).
- Loop sepenuhnya **blocking & single-session** — tidak bisa melayani PC + robot bersamaan
  tanpa refactor ke model sesi/async (lihat INTEGRATION_POINTS §1).

## E. Robustness / korektness
- `DEFAULT_SYSTEM_PROMPT` memakai `role: "assistant"` untuk system prompt (`main.py:15`), bukan
  `role: "system"`. Tidak standar; beberapa model mengabaikannya.
- Deteksi stop-word & fakta = **substring match kasar** (`main.py:113`, `131`). "saya suka" akan
  ke-trigger di kalimat apa pun; "keluar" bisa false-positive.
- `temp_chunk.mp3` dipakai bersama untuk semua chunk; di Windows file-lock kadang gagal dihapus
  (`tts_engine.py:62-64`). Bila TTS dipakai konkuren (multi-sesi) → race condition nama file tetap.
- `save_memory` menelan semua exception diam-diam (`main.py:56-57`) → kegagalan tulis tak terlihat.
- `id=str(time.time())` untuk fakta (`main.py:132`) bisa **tabrakan** bila dua fakta < 1ms; juga
  meng-`upsert` id berbeda terus → tidak ada dedup fakta serupa.
- RAG selalu mengambil top-2 tanpa ambang skor → bisa menyuntik konteks tidak relevan.

## F. Ketergantungan eksternal & klaim "lokal" — ✅ sebagian RESOLVED (Fase 2, swap TTS)
- ✅ **TTS kini default lokal**: engine `mms` (MMS-TTS Bahasa Indonesia) jadi default di
  `config.yaml → tts.engine`, 100% offline. Edge TTS (cloud) turun jadi opsi `edge` (legacy).
  Voice cloning lokal tersedia via engine `f5_indo`. Lihat `modules/tts_factory.py`.
  ✅ **Catatan IPEX sudah tidak berlaku** (migrasi Arc, 2026-07-25): dukungan XPU menyatu di
  torch inti — cukup build `+xpu` + driver Intel. Ternyata MMS-TTS **lebih cepat di CPU**
  (780 ms vs 2900 ms per kalimat), jadi `tts.mms.device` sengaja dipatok `cpu`; lihat
  ARCHITECTURE §8 untuk angka & alasannya.
- ⚠️ LLM tetap bergantung **LM Studio hidup** di `:1234`. Bila mati, `stream_response` hanya
  print error dan loop lanjut tanpa suara. (Belum ditangani.)

## G. Keamanan / rahasia — ✅ sebagian RESOLVED (Fase 2, Langkah 0)
- `config.yaml: llm.api_key = "lm-studio"` — placeholder lokal, bukan rahasia nyata (aman).
- ✅ `memory.json` + sampel suara `voices/*.wav` kini sudah di-`.gitignore` (data pribadi).
- `.env` sudah di-gitignore. `data/` gitignored.

## H. Dependensi hardware (temuan migrasi Intel Arc, 2026-07-25)
- ⚠️ **`numpy` terkunci `<2.5`** oleh `openvino 2026.2.x`. Upgrade numpy tanpa cek akan
  mematikan impor OpenVINO (STT backend `openvino` + `scripts/check_hardware.py`).
- ⚠️ **`optimum-intel` tidak boleh masuk venv runtime**: menuntut `transformers<5.1`,
  `safetensors<0.8.0`, `requests>=2.33` — bentrok mati dengan `transformers` 5.x
  (`safetensors>=0.8.0`) dan pin `requests==2.31.0`. Konversi model dilakukan di
  `venv-convert/` (lihat `models/README.md` §1b).
- ⚠️ **`torch` dipatok 2.11.0** karena `torchaudio` (dipakai engine `f5_indo`) rilis
  terakhirnya 2.11.0. Kalau `f5_indo` dibuang, torch boleh naik ke 2.13+.
- ✅ **Lockfile**: `requirements.lock.txt` = kondisi setelah migrasi yang sudah diverifikasi
  jalan; `requirements.lock.pre-xpu.txt` = titik rollback sebelum migrasi (torch CPU + numpy 2.5.1).
- ⚠️ **Ekspor Whisper→OpenVINO punya 2 jebakan** yang menghasilkan model gagal-pakai:
  task harus `automatic-speech-recognition-with-past` (kalau tidak: `beam_idx ... not found`),
  dan `generation_config.json` hasil ekspor harus ditambal `lang_to_id` lewat
  `scripts/patch_ov_whisper_config.py` (kalau tidak: `'lang_to_id' map must be provided`).
  Detail di `models/README.md` §1b.
- ⚠️ **Plugin GPU OpenVINO mencetak ~17 baris `onednn_verbose,...error,ocl,...` ke STDOUT**
  saat init (probing OpenCL antara Arc & iGPU Radeon; setelahnya jalan normal). Dibungkam
  dengan `ONEDNN_VERBOSE=0` di `main.py`; `scripts/check_hardware.py` sengaja tidak
  membungkamnya agar tetap kelihatan saat mendiagnosa.
- ⚠️ **`data/ov_cache` ~760 MB** (cache kernel GPU). Gitignored, aman dihapus — hanya membuat
  start pertama kembali ~12 detik.
- ⚠️ **`faster_whisper` masih menyentuh internet saat load**: folder lokal
  `models/stt/faster-whisper-medium-id/` tidak punya `tokenizer.json`, sehingga
  faster-whisper mengunduh tokenizer `openai/whisper-tiny` dari HF. Klaim "100% offline"
  belum sepenuhnya benar untuk STT (backend `openvino` tidak punya masalah ini).
- ⚠️ **Engine `f5_indo` belum diuji ulang di torch 2.11+xpu**: paket `f5-tts` belum
  terpasang di venv (berat: gradio/wandb/bitsandbytes/torchcodec), dan patch
  `torchaudio.load` di `tts_f5.py` perlu diverifikasi lagi di torchaudio 2.11.

## Rekomendasi prioritas
1. ✅ Perbaiki `requirements.txt`. — **selesai Fase 2 Langkah 0**.
2. ✅ Pindahkan model LLM & STT ke `config.yaml`. — **selesai**.
3. ✅ Tambah `memory.json` ke `.gitignore`. — **selesai**.
4. ⬜ Putuskan nasib kode mati (router/vision/memory_rag): pakai atau hapus saat refactor.
5. ⬜ Sambungkan streaming per-kalimat LLM→TTS untuk latensi (Langkah 4).
6. ⬜ Update pin versi tua (`chromadb`, dll) setelah pipeline stabil.
