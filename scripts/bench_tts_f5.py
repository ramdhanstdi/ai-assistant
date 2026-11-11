"""
Benchmark F5-TTS (voice cloning) di CPU vs Intel Arc B580 (xpu).

    venv\\Scripts\\python.exe scripts/bench_tts_f5.py [--devices cpu,xpu]

Berbeda dengan MMS-TTS (model kecil, menang di CPU), F5 adalah model besar dengan
banyak langkah sampling — inilah kandidat utama pemakai GPU. Skrip ini mengukurnya,
bukan menebak.

Memakai setelan nyata dari config.yaml (tts.f5_indo): checkpoint, ref_audio, ref_text.
Tidak memutar suara dan tidak menulis wav — durasi dihitung dari array hasil.
CATATAN: di CPU satu kalimat bisa puluhan detik, jadi jumlah kalimat dibuat sedikit.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from modules.tts_f5 import apply_windows_patches  # noqa: E402
from modules.tts_mms import pick_device  # noqa: E402

KALIMAT = [
    "Halo, aku siap bantu kamu hari ini.",
    "Oke, aku sudah ingat kalau kamu suka kopi susu.",
]


def load_conf():
    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = (yaml.safe_load(f) or {}).get("tts", {}).get("f5_indo", {})
    ref_audio = conf.get("ref_audio", "voices/reference.wav")
    ref_text_path = conf.get("ref_text", "voices/reference.txt")
    ref_text = ""
    if os.path.exists(ref_text_path):
        with open(ref_text_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
    return conf, ref_audio, ref_text


def bench(device: str, conf, ref_audio, ref_text):
    print(f"\n--- device: {device} ---")
    from f5_tts.api import F5TTS

    try:
        t0 = time.perf_counter()
        model = F5TTS(
            model=conf.get("model", "F5TTS_Base"),
            ckpt_file=conf.get("ckpt_file", ""),
            vocab_file=conf.get("vocab_file", ""),
            device=device,
        )
        print(f"  load model: {time.perf_counter() - t0:.1f}s")
    except Exception as e:
        print(f"  GAGAL memuat di '{device}': {type(e).__name__}: {e}")
        return

    def synth(text: str):
        # file_wave=None -> tidak menulis file; kembalikan (wav, sr, spec).
        wav, sr, _ = model.infer(
            ref_file=ref_audio, ref_text=ref_text, gen_text=text, file_wave=None, show_info=lambda *a, **k: None,
        )
        return wav, sr

    try:
        t0 = time.perf_counter()
        synth("Halo.")
        print(f"  warmup    : {time.perf_counter() - t0:.1f}s (kompilasi kernel / alokasi)")
    except Exception as e:
        print(f"  GAGAL sintesis di '{device}': {type(e).__name__}: {e}")
        return

    total_s, total_audio = 0.0, 0.0
    for text in KALIMAT:
        t0 = time.perf_counter()
        wav, sr = synth(text)
        dt = time.perf_counter() - t0
        audio_s = len(wav) / sr
        total_s += dt
        total_audio += audio_s
        print(f"  {dt:6.1f}s | audio {audio_s:4.1f}s | {audio_s / dt:5.2f}x realtime | {text[:38]}...")

    print(f"  RATA-RATA : {total_s / len(KALIMAT):.1f}s/kalimat ({total_audio / total_s:.2f}x realtime)")


def main():
    devices = ["cpu", "xpu"]
    for a in sys.argv[1:]:
        if a.startswith("--devices"):
            devices = a.split("=", 1)[-1].split(",") if "=" in a else devices

    conf, ref_audio, ref_text = load_conf()
    if not os.path.exists(ref_audio):
        print(f"Sampel suara '{ref_audio}' tidak ada (lihat voices/README.md).")
        return
    ckpt = conf.get("ckpt_file", "")
    if ckpt and not os.path.exists(ckpt):
        print(f"Checkpoint '{ckpt}' tidak ada (lihat models/README.md §4).")
        return
    print(f"ref_audio : {ref_audio}\nref_text  : \"{ref_text[:60]}...\"\ncheckpoint: {ckpt or '(base F5)'}")

    apply_windows_patches()  # WAJIB sebelum import f5_tts di Windows

    if "xpu" in devices and pick_device("auto") != "xpu":
        print("\n(xpu tidak tersedia -> hanya CPU. Jalankan scripts/check_hardware.py)")
        devices = [d for d in devices if d != "xpu"]

    for d in devices:
        bench(d, conf, ref_audio, ref_text)

    print("\nCatatan: faktor realtime < 1 berarti sintesis lebih lambat dari durasi bicara,")
    print("artinya streaming per kalimat akan tersendat menunggu suara.")


if __name__ == "__main__":
    main()
