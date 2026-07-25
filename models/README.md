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

## 1b. STT di Intel Arc — Whisper → OpenVINO IR  *(OPSIONAL, untuk `stt.backend: "openvino"`)*
Hasil konversi ditaruh di: **`models/stt/whisper-medium-id-ov/`**

Model CTranslate2 di §1 **tidak bisa** dipakai OpenVINO, dan model OpenVINO tidak bisa
di-download jadi — harus **dikonversi sekali** dari repo Hugging Face aslinya
(https://huggingface.co/cahya/whisper-medium-id, `pytorch_model.bin` ~3 GB; hasil IR int8
≈ 800 MB).

Konversi WAJIB di **venv terpisah**, bukan sekadar demi kerapian: `optimum-intel` menuntut
`transformers<5.1`, `safetensors<0.8.0`, dan `requests>=2.33`, sementara venv runtime memakai
`transformers` 5.x + `safetensors>=0.8.0` + `requests==2.31.0` — bentrok dan tidak ada solusi
yang memuaskan keduanya. Venv `venv-convert/` sudah di-gitignore dan boleh dihapus setelah
konversi selesai.

```bash
python -m venv venv-convert
venv-convert\Scripts\python.exe -m pip install "optimum-intel[openvino]"

venv-convert\Scripts\optimum-cli.exe export openvino ^
    --model cahya/whisper-medium-id ^
    --task automatic-speech-recognition-with-past ^
    --weight-format int8 ^
    models/stt/whisper-medium-id-ov

REM Lengkapi generation_config.json (lang_to_id) -- lihat penjelasan di bawah
venv\Scripts\python.exe scripts/patch_ov_whisper_config.py
```

**Dua jebakan yang WAJIB diperhatikan** (dua-duanya sudah pernah menghasilkan model yang
gagal jalan di sini):

1. **`-with-past` itu bukan hiasan.** Config `cahya/whisper-medium-id` berisi
   `use_cache: false` (sisa setelan training), dan optimum menentukan KV-cache dari nama task
   (`use_cache=task.endswith("with-past")`). Dengan `--task automatic-speech-recognition`
   biasa, dekoder diekspor tanpa KV-cache/stateful dan openvino_genai menolak:
   `Port for tensor name beam_idx was not found`.
2. **`generation_config.json` hasil ekspor tidak lengkap.** Repo finetune tidak menyertakan
   file itu, jadi optimum menurunkannya dari `config.json` saja — tanpa `lang_to_id`, sehingga
   pipeline gagal: `Check '!lang_to_id.empty()' failed`. `scripts/patch_ov_whisper_config.py`
   menambalnya dari `openai/whisper-medium` (peta token milik tokenizer Whisper multilingual,
   bukan hasil finetune, dan vocab-nya identik: 51865 token, `sot=50258`). File asli disimpan
   sebagai `generation_config.optimum.json`.

> **Kalau unduhan 3 GB-nya menggantung** (ukuran file `.incomplete` di
> `~/.cache/huggingface/hub` berhenti bergerak): itu backend transfer **xet**. Matikan dan
> ulangi — unduhan HTTP klasik bisa resume:
> ```bash
> set HF_HUB_DISABLE_XET=1
> set HF_HUB_DOWNLOAD_TIMEOUT=60
> ```
> Hapus dulu file `*.incomplete` sisa xet (tidak bisa dipakai jalur klasik), lalu jalankan
> `optimum-cli` lagi.

Hasilnya harus berisi `openvino_encoder_model.xml/.bin`, `openvino_decoder_model.xml/.bin`,
`generation_config.json`, dan file tokenizer. Modul `modules/stt_ov.py` memeriksa keberadaan
`openvino_encoder_model.xml`; bila tidak ada, `stt_factory` otomatis kembali ke backend
`faster_whisper` (CPU) dengan peringatan.

Setelah itu aktifkan di `config.yaml`:

```yaml
stt:
  backend: "openvino"
  openvino:
    model_dir: "models/stt/whisper-medium-id-ov"
    device: "GPU"
```

Bandingkan kecepatannya dengan `python scripts/bench_stt.py`.
Config terkait: `stt.backend`, `stt.openvino.*`.

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
| STT whisper (CPU) | `models/stt/faster-whisper-medium-id/` | `stt.local_dir` | wajib |
| STT whisper (Arc, IR) | `models/stt/whisper-medium-id-ov/` | `stt.openvino.model_dir` | opsional (hasil konversi §1b) |
| TTS MMS | `models/tts/mms-tts-ind/` | `tts.mms.local_dir` | wajib (engine mms) |
| Embedding | `models/embedding/all-MiniLM-L6-v2/` | `memory.embedding_local_dir` | wajib |
| F5 cloning | `models/tts/f5-indo/` | `tts.f5_indo.ckpt_file` | opsional |
| Vision | `models/vision/moondream2/` | `vision.local_dir` | opsional |

Kalau sebuah folder dibiarkan kosong, model itu otomatis di-download saat dibutuhkan.
LLM **tidak** ada di sini — itu dimuat oleh **LM Studio**, bukan oleh kode ini.
