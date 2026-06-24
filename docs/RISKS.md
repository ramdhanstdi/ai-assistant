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
  ⚠️ Catatan baru: engine `mms`/`f5_indo` butuh `torch`+`transformers` (berat); akselerasi
  Intel Arc butuh IPEX terpasang terpisah, jika tidak fallback ke CPU (lebih lambat).
- ⚠️ LLM tetap bergantung **LM Studio hidup** di `:1234`. Bila mati, `stream_response` hanya
  print error dan loop lanjut tanpa suara. (Belum ditangani.)

## G. Keamanan / rahasia — ✅ sebagian RESOLVED (Fase 2, Langkah 0)
- `config.yaml: llm.api_key = "lm-studio"` — placeholder lokal, bukan rahasia nyata (aman).
- ✅ `memory.json` + sampel suara `voices/*.wav` kini sudah di-`.gitignore` (data pribadi).
- `.env` sudah di-gitignore. `data/` gitignored.

## Rekomendasi prioritas
1. ✅ Perbaiki `requirements.txt`. — **selesai Fase 2 Langkah 0**.
2. ✅ Pindahkan model LLM & STT ke `config.yaml`. — **selesai**.
3. ✅ Tambah `memory.json` ke `.gitignore`. — **selesai**.
4. ⬜ Putuskan nasib kode mati (router/vision/memory_rag): pakai atau hapus saat refactor.
5. ⬜ Sambungkan streaming per-kalimat LLM→TTS untuk latensi (Langkah 4).
6. ⬜ Update pin versi tua (`chromadb`, dll) setelah pipeline stabil.
