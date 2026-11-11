"""
Tambal generation_config.json hasil 'optimum-cli export openvino' untuk Whisper.

    venv\\Scripts\\python.exe scripts/patch_ov_whisper_config.py [ir_dir] [base_repo]

MASALAH YANG DIPERBAIKI
Repo finetune (mis. cahya/whisper-medium-id) tidak menyertakan generation_config.json,
jadi optimum menurunkannya dari config.json saja. Hasilnya kehilangan peta token khusus
Whisper -- terutama 'lang_to_id' -- dan openvino_genai menolak jalan:

    Check '!lang_to_id.empty()' failed ... 'lang_to_id' map must be provided

Peta itu milik tokenizer Whisper multilingual, bukan hasil finetune, jadi aman diambil
dari repo Whisper resmi dengan vocab yang sama (51865 token, sot=50258).

Yang dilakukan: base (openai/whisper-medium) + nilai dari file hasil ekspor ditumpuk di
atasnya, lalu ditulis ulang. File asli disimpan sebagai generation_config.optimum.json.
Idempoten -- aman dijalankan ulang.
"""
import json
import os
import shutil
import sys

# Kunci ini murni artefak ekspor, jangan dibawa ke hasil merge:
# _from_model_config/transformers_version = metadata; use_cache=False bukan urusan generasi
# (dekoder IR sudah stateful) dan bisa menyesatkan.
_BUANG = {"_from_model_config", "transformers_version", "use_cache"}

# Bukti bahwa base cocok dengan model: token awal & panjang maksimum harus sama.
_WAJIB_SAMA = ["decoder_start_token_id", "max_length"]


def main():
    ir_dir = sys.argv[1] if len(sys.argv) > 1 else "models/stt/whisper-medium-id-ov"
    base_repo = sys.argv[2] if len(sys.argv) > 2 else "openai/whisper-medium"

    target = os.path.join(ir_dir, "generation_config.json")
    backup = os.path.join(ir_dir, "generation_config.optimum.json")
    if not os.path.exists(target) and not os.path.exists(backup):
        print(f"❌ '{target}' tidak ada. Jalankan ekspor optimum-cli dulu (models/README.md §1b).")
        return 1

    # Sumber merge: pakai backup bila sudah pernah ditambal (idempoten).
    src = backup if os.path.exists(backup) else target
    with open(src, "r", encoding="utf-8") as f:
        exported = json.load(f)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # backend xet gampang menggantung
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(base_repo, "generation_config.json"), "r", encoding="utf-8") as f:
        base = json.load(f)

    if "lang_to_id" not in base:
        print(f"❌ '{base_repo}' tidak punya lang_to_id — pilih repo Whisper multilingual lain.")
        return 1

    beda = [k for k in _WAJIB_SAMA if k in exported and base.get(k) != exported[k]]
    if beda:
        print(f"⚠️ Base '{base_repo}' beda di {beda} — kemungkinan ukuran/vocab model tidak cocok.")
        print("   Nilai dari model hasil ekspor yang dipakai, tapi periksa ulang hasilnya.")

    merged = {**base, **{k: v for k, v in exported.items() if k not in _BUANG}}

    if not os.path.exists(backup):
        shutil.copy2(target, backup)
        print(f"📦 Cadangan file asli -> {backup}")

    with open(target, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    lang = f"<|{'id'}|>"
    print(f"✅ {target} ditambal: {len(merged['lang_to_id'])} bahasa, "
          f"{lang} = {merged['lang_to_id'].get(lang)}, task_to_id = {merged.get('task_to_id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
