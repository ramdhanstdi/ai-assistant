"""
Benchmark STT: faster-whisper (CPU/CTranslate2) vs OpenVINO Whisper (Arc B580 & CPU).

    venv\\Scripts\\python.exe scripts/bench_stt.py [file.wav]

Default file uji: f5_out_test.wav (ada di repo). File apa pun boleh -- otomatis
di-resample ke 16 kHz mono float32, format yang diminta semua backend.

Yang dilaporkan per backend: waktu transkripsi, faktor realtime, dan teks hasilnya
(supaya kecepatan tidak ditukar dengan akurasi tanpa sadar). Warmup dibuang.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Bungkam semburan 'onednn_verbose,...error,ocl' dari plugin GPU OpenVINO (probing OpenCL
# yang gagal lalu tetap jalan) supaya tabel hasil benchmark terbaca.
os.environ.setdefault("ONEDNN_VERBOSE", "0")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

RUNS = 3


def load_audio(path: str):
    """WAV apa pun -> (float32 mono 16 kHz, durasi detik)."""
    import soundfile as sf
    from scipy.signal import resample_poly

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)  # mono
    if sr != 16000:
        from math import gcd
        g = gcd(16000, sr)
        data = resample_poly(data, 16000 // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(data, dtype=np.float32), len(data) / 16000.0


def report(label: str, fn, audio, audio_s: float):
    try:
        text = fn(audio)  # warmup (kompilasi kernel / alokasi)
    except Exception as e:
        print(f"  {label:26s} GAGAL: {type(e).__name__}: {e}")
        return
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        text = fn(audio)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f"  {label:26s} {avg * 1000:7.0f} ms | {audio_s / avg:5.1f}x realtime")
    print(f"  {'':26s} -> \"{text}\"")


def bench_faster_whisper(conf, audio, audio_s):
    from modules.stt_engine import STTManager
    print("\n--- faster-whisper (CTranslate2) ---")
    stt = STTManager()
    report(f"cpu/{conf.get('compute_type', 'int8')}", stt.transcribe, audio, audio_s)


def bench_openvino(conf, audio, audio_s):
    ov_conf = conf.get("openvino", {})
    model_dir = ov_conf.get("model_dir", "models/stt/whisper-medium-id-ov")
    if not os.path.exists(os.path.join(model_dir, "openvino_encoder_model.xml")):
        print(f"\n--- OpenVINO: DILEWATI (IR belum ada di '{model_dir}', lihat models/README.md) ---")
        return

    import openvino as ov
    import openvino_genai as ov_genai

    print(f"\n--- OpenVINO Whisper ('{model_dir}') ---")
    language = f"<|{conf.get('language', 'id')}|>"
    available = ov.Core().available_devices
    for device in ["GPU", "CPU"]:
        if device not in available:
            print(f"  {device:26s} tidak tersedia di mesin ini")
            continue
        try:
            t0 = time.perf_counter()
            pipe = ov_genai.WhisperPipeline(model_dir, device)
            load_s = time.perf_counter() - t0
        except Exception as e:
            print(f"  {device:26s} GAGAL memuat: {type(e).__name__}: {e}")
            continue

        def run(a, _pipe=pipe):
            r = _pipe.generate(a, language=language, task="transcribe", return_timestamps=False)
            texts = getattr(r, "texts", None)
            return (" ".join(texts) if texts else str(r)).strip()

        report(f"{device} (load {load_s:.1f}s)", run, audio, audio_s)
        del pipe


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "f5_out_test.wav"
    if not os.path.exists(path):
        print(f"File audio '{path}' tidak ada. Berikan path wav sebagai argumen.")
        return

    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = (yaml.safe_load(f) or {}).get("stt", {})

    audio, audio_s = load_audio(path)
    print(f"File uji : {path} ({audio_s:.2f}s, 16 kHz mono, {RUNS} run + warmup)")

    bench_faster_whisper(conf, audio, audio_s)
    bench_openvino(conf, audio, audio_s)


if __name__ == "__main__":
    main()
