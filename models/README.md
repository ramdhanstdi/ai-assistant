# models/ — Tempat menaruh model hasil download manual

Kalau kamu download model manual dari browser (Hugging Face), **taruh file-nya di folder yang
sesuai di bawah**. Kode otomatis memakai folder lokal bila berisi file; kalau folder kosong,
kode fallback ke auto-download dari Hugging Face (lihat `modules/model_paths.py`).

> Cara download per file di browser: buka halaman repo → tab **Files and versions** → klik file →
> tombol **download** (ikon panah). Atau pakai URL langsung:
> `https://huggingface.co/<REPO>/resolve/main/<NAMA_FILE>`
>
> Folder ini (`models/`) di-gitignore (file model besar), jadi tidak ikut ter-commit.

---

## 1. STT — `cahya/faster-whisper-medium-id`  **(WAJIB)**
Taruh di: **`models/stt/faster-whisper-medium-id/`**
Repo: https://huggingface.co/cahya/faster-whisper-medium-id

| File | Ukuran |
|------|--------|
| `config.json` | 7 KB |
| `model.bin` | **1.43 GB** |
| `preprocessor_config.json` | 185 KB |
| `tokenizer_config.json` | <1 KB |
| `vocabulary.json` | 1 MB |

Config terkait: `stt.local_dir`.

---

## 2. TTS default — `facebook/mms-tts-ind`  **(WAJIB untuk engine `mms`)**
Taruh di: **`models/tts/mms-tts-ind/`**
Repo: https://huggingface.co/facebook/mms-tts-ind

| File | Ukuran |
|------|--------|
| `config.json` | 1.6 KB |
| `model.safetensors` | 145 MB |
| `tokenizer_config.json` | <1 KB |
| `special_tokens_map.json` | <1 KB |
| `vocab.json` | <1 KB |

> `pytorch_model.bin` (145 MB) **tidak perlu** kalau sudah ambil `model.safetensors`.
Config terkait: `tts.mms.local_dir`.

---

## 3. Embedding RAG — `sentence-transformers/all-MiniLM-L6-v2`  **(WAJIB)**
Taruh di: **`models/embedding/all-MiniLM-L6-v2/`** (pertahankan subfolder `1_Pooling/`)
Repo: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

| File | Ukuran |
|------|--------|
| `config.json` | <1 KB |
| `config_sentence_transformers.json` | <1 KB |
| `model.safetensors` | 91 MB |
| `modules.json` | <1 KB |
| `sentence_bert_config.json` | <1 KB |
| `special_tokens_map.json` | <1 KB |
| `tokenizer.json` | 466 KB |
| `tokenizer_config.json` | <1 KB |
| `vocab.txt` | 232 KB |
| `1_Pooling/config.json` | <1 KB |

> File opsional yang **tidak perlu**: `pytorch_model.bin`, `rust_model.ot`, `tf_model.h5`,
> `onnx/`, `openvino/`, `data_config.json`, `train_script.py`.
Config terkait: `memory.embedding_local_dir`.

---

## 4. TTS voice cloning — `Eempostor/F5-TTS-INDO-FINETUNE`  *(OPSIONAL, hanya untuk engine `f5_indo`)*
Taruh di: **`models/tts/f5-indo/`**
Repo: https://huggingface.co/Eempostor/F5-TTS-INDO-FINETUNE

| File | Ukuran | Catatan |
|------|--------|---------|
| `f5_tts_ind.pt` | **1.26 GB** | checkpoint utama |

> Repo ini **tidak** menyertakan `vocab.txt` → biarkan `tts.f5_indo.vocab_file` kosong
> (pakai vocab base F5 bawaan paket `f5-tts`). Butuh `pip install f5-tts` dulu.
Config terkait: `tts.f5_indo.ckpt_file`.

---

## 5. Vision — `vikhyatk/moondream2` *(OPSIONAL, belum dirangkai ke main.py)*
Taruh di: **`models/vision/moondream2/`** — **revision `2024-05-08`**
Repo: https://huggingface.co/vikhyatk/moondream2/tree/2024-05-08

Butuh semua file di revision tersebut (termasuk file kode `*.py` karena `trust_remote_code=True`,
config, dan bobot `*.safetensors`). Lewati saja kalau belum memakai fitur vision.

---

## Ringkasan: folder → config
| Model | Folder | Kunci config | Status |
|-------|--------|--------------|--------|
| STT whisper | `models/stt/faster-whisper-medium-id/` | `stt.local_dir` | wajib |
| TTS MMS | `models/tts/mms-tts-ind/` | `tts.mms.local_dir` | wajib (engine mms) |
| Embedding | `models/embedding/all-MiniLM-L6-v2/` | `memory.embedding_local_dir` | wajib |
| F5 cloning | `models/tts/f5-indo/` | `tts.f5_indo.ckpt_file` | opsional |
| Vision | `models/vision/moondream2/` | `vision.local_dir` | opsional |

Kalau sebuah folder dibiarkan kosong, model itu otomatis di-download saat dibutuhkan.
LLM **tidak** ada di sini — itu dimuat oleh **LM Studio**, bukan oleh kode ini.
