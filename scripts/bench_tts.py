"""
Benchmark MMS-TTS: CPU (Ryzen 8500G) vs Intel Arc B580 (xpu).

    venv\\Scripts\\python.exe scripts/bench_tts.py

Mengukur waktu sintesis per kalimat dan faktor realtime (berapa kali lebih cepat dari
durasi audio yang dihasilkan). Tidak memutar suara -- murni angka, aman dijalankan
sambil headset lepas. Warmup dibuang dari hitungan (kompilasi kernel & alokasi VRAM).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml  # noqa: E402

from modules.model_paths import resolve  # noqa: E402

# Panjang kalimat dibuat mirip jawaban asisten sehari-hari (1-3 kalimat pendek).
KALIMAT = [
    "Halo, aku siap bantu kamu hari ini.",
    "Cuacanya cerah, kayaknya enak buat jalan-jalan sore.",
    "Oke, aku sudah ingat kalau kamu tinggal di Bandung dan suka kopi susu.",
]


def load_model(device: str):
    import torch
    from transformers import VitsModel, AutoTokenizer

    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = (yaml.safe_load(f) or {}).get("tts", {}).get("mms", {})
    model_id = resolve(conf.get("local_dir"), conf.get("model_id", "facebook/mms-tts-ind"))

    model = VitsModel.from_pretrained(model_id).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    return torch, model, tok, model.config.sampling_rate


def bench(device: str):
    print(f"\n--- device: {device} ---")
    try:
        torch, model, tok, sr = load_model(device)
    except Exception as e:
        print(f"  GAGAL memuat model di '{device}': {type(e).__name__}: {e}")
        return

    def synth(text: str):
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            wav = model(**inputs).waveform
        if device.startswith("xpu"):
            torch.xpu.synchronize()
        return wav.squeeze(0).detach().to("cpu").float().numpy()

    # Warmup: 2 kali, hasilnya dibuang.
    try:
        for _ in range(2):
            synth("Halo.")
    except Exception as e:
        print(f"  GAGAL eksekusi di '{device}': {type(e).__name__}: {e}")
        return

    total_synth = 0.0
    total_audio = 0.0
    for text in KALIMAT:
        t0 = time.perf_counter()
        wav = synth(text)
        dt = time.perf_counter() - t0
        audio_s = len(wav) / sr
        total_synth += dt
        total_audio += audio_s
        print(f"  {dt * 1000:7.0f} ms  | audio {audio_s:4.1f}s | {audio_s / dt:5.1f}x realtime | {text[:40]}...")

    print(f"  RATA-RATA: {total_synth / len(KALIMAT) * 1000:.0f} ms/kalimat "
          f"({total_audio / total_synth:.1f}x realtime)")


def main():
    devices = ["cpu"]
    try:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            devices.append("xpu")
        else:
            print("(xpu tidak tersedia -> hanya benchmark CPU. Jalankan scripts/check_hardware.py)")
    except ImportError:
        print("torch belum terpasang.")
        return

    for d in devices:
        bench(d)
    print("\nCatatan: faktor realtime > 1 berarti sintesis lebih cepat dari durasi bicara,")
    print("yaitu syarat streaming TTS per kalimat tidak tersendat.")


if __name__ == "__main__":
    main()
