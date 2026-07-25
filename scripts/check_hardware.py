"""
Cek ketersediaan akselerator untuk AI Assistant lokal (Intel Arc B580 + Ryzen 8500G).

Jalankan kapan pun setelah mengubah versi torch / driver GPU / openvino:

    venv\\Scripts\\python.exe scripts/check_hardware.py

Yang diperiksa:
  1. torch  -> build XPU (Intel Arc) terpasang & GPU kebaca?   dipakai TTS (mms/f5_indo)
  2. OpenVINO -> device GPU kebaca?                            dipakai STT (WhisperPipeline)
  3. numpy  -> masih < 2.5 (batas openvino 2026.2.x)?
Semua section dibungkus try/except supaya satu kegagalan tidak menutupi hasil lainnya.
"""
import platform
import sys
import time

OK = "[ OK ]"
NO = "[FAIL]"
WARN = "[WARN]"


def _header(title: str):
    print()
    print("=" * 66)
    print(title)
    print("=" * 66)


def check_python():
    _header("1. PYTHON & OS")
    print(f"      python   : {sys.version.split()[0]} ({sys.executable})")
    print(f"      platform : {platform.platform()}")


def check_torch() -> bool:
    """True bila torch bisa memakai Intel Arc (xpu)."""
    _header("2. TORCH / INTEL ARC (XPU)  -- dipakai TTS mms & f5_indo")
    try:
        import torch
    except ImportError as e:
        print(f"{NO} torch belum terpasang: {e}")
        return False

    print(f"      versi torch : {torch.__version__}")
    # Build XPU selalu bersuffix '+xpu'. Build CPU/CUDA tidak akan pernah melihat Arc.
    if "+xpu" not in torch.__version__:
        print(f"{WARN} ini BUKAN build XPU. Install ulang:")
        print("       pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \\")
        print("           --index-url https://download.pytorch.org/whl/xpu")

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print(f"{NO} torch.xpu.is_available() = False -> TTS akan jalan di CPU.")
        print("       Cek: (a) build torch '+xpu', (b) driver Intel Graphics terbaru untuk Arc B580.")
        return False

    count = torch.xpu.device_count()
    print(f"{OK} torch.xpu.is_available() = True (device_count={count})")
    for i in range(count):
        try:
            p = torch.xpu.get_device_properties(i)
            vram = getattr(p, "total_memory", 0) / (1024 ** 3)
            print(f"      xpu:{i} -> {getattr(p, 'name', '?')} | VRAM {vram:.1f} GB")
        except Exception as e:
            print(f"      xpu:{i} -> properti tidak terbaca: {e}")

    # Smoke test: matmul kecil. Membuktikan runtime & driver benar-benar bisa eksekusi kernel,
    # bukan cuma "terdeteksi". Iterasi pertama sengaja dibuang (kompilasi kernel/warmup).
    try:
        a = torch.randn(1024, 1024, device="xpu")
        b = torch.randn(1024, 1024, device="xpu")
        (a @ b).sum().item()  # warmup + sinkronisasi implisit
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            c = a @ b
        torch.xpu.synchronize()
        dt = (time.perf_counter() - t0) / 10
        print(f"{OK} smoke test matmul 1024x1024 di xpu: {dt * 1000:.1f} ms/iterasi (hasil {c.shape})")
        return True
    except Exception as e:
        print(f"{NO} device terdeteksi tapi GAGAL eksekusi kernel: {type(e).__name__}: {e}")
        return False


def check_openvino() -> bool:
    """True bila OpenVINO melihat device GPU."""
    _header("3. OPENVINO  -- dipakai STT (WhisperPipeline) & VLM Tahap B")
    try:
        import openvino as ov
    except ImportError as e:
        print(f"{NO} openvino belum terpasang: {e}")
        return False

    print(f"      versi openvino : {ov.__version__}")
    try:
        import openvino_genai
        print(f"      openvino_genai : {openvino_genai.__version__}")
    except Exception as e:
        print(f"{WARN} openvino_genai tidak bisa diimpor: {type(e).__name__}: {e}")

    try:
        core = ov.Core()
        devices = core.available_devices
    except Exception as e:
        print(f"{NO} gagal query device: {type(e).__name__}: {e}")
        return False

    if not devices:
        print(f"{NO} tidak ada device OpenVINO sama sekali.")
        return False

    print(f"      device     : {', '.join(devices)}")
    for d in devices:
        try:
            print(f"      {d:10s} -> {core.get_property(d, 'FULL_DEVICE_NAME')}")
        except Exception:
            print(f"      {d:10s} -> (nama lengkap tidak tersedia)")

    has_gpu = any(d.startswith("GPU") for d in devices)
    if has_gpu:
        print(f"{OK} device GPU tersedia -> STT OpenVINO bisa pakai device 'GPU'.")
    else:
        print(f"{NO} tidak ada device 'GPU' (hanya CPU). STT OpenVINO akan jalan di CPU.")
    return has_gpu


def check_deps():
    _header("4. DEPENDENSI KRITIS")
    import importlib.metadata as md

    # numpy: openvino 2026.2.x mensyaratkan <2.5.0. Naik ke 2.5+ akan merusak openvino.
    for name, note in [
        ("numpy", "openvino 2026.2.x butuh <2.5.0"),
        ("transformers", "VitsModel untuk MMS-TTS"),
        ("faster-whisper", "STT backend fallback (CPU/CTranslate2)"),
        ("ctranslate2", "engine faster-whisper (tanpa dukungan Intel GPU)"),
        ("chromadb", "RAG"),
        ("sentence-transformers", "embedding RAG"),
        ("torchaudio", "hanya dibutuhkan engine TTS f5_indo"),
    ]:
        try:
            print(f"      {name:22s} {md.version(name):12s}  ({note})")
        except md.PackageNotFoundError:
            print(f"      {name:22s} {'-':12s}  (belum terpasang; {note})")

    try:
        from packaging.version import Version
        if Version(md.version("numpy")) >= Version("2.5.0"):
            print(f"{WARN} numpy >= 2.5.0 -> openvino bisa gagal impor. Turunkan: pip install 'numpy<2.5'")
    except Exception:
        pass


def main():
    check_python()
    torch_xpu = check_torch()
    ov_gpu = check_openvino()
    check_deps()

    _header("RINGKASAN")
    print("      -- kemampuan hardware --")
    print(f"      torch bisa pakai Arc     : {'YA' if torch_xpu else 'TIDAK'}")
    print(f"      OpenVINO bisa pakai GPU  : {'YA' if ov_gpu else 'TIDAK'}")

    # Kemampuan != pilihan. Device yang DIPAKAI ditentukan config.yaml, dan untuk MMS-TTS
    # pilihan sadarnya adalah CPU (lihat ARCHITECTURE §8: GPU justru lebih lambat di situ).
    print("      -- yang dipakai config.yaml --")
    try:
        import os
        import yaml
        cfg_path = "config.yaml"
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        stt = cfg.get("stt", {})
        tts = cfg.get("tts", {})
        engine = tts.get("engine", "mms")
        backend = stt.get("backend", "faster_whisper")
        stt_dev = stt.get("openvino", {}).get("device", "GPU") if backend in ("openvino", "ov") else stt.get("device", "cpu")
        print(f"      STT                      : backend '{backend}' di '{stt_dev}'")
        print(f"      TTS                      : engine '{engine}' di '{tts.get(engine, {}).get('device', '-')}'")
    except Exception as e:
        print(f"      (config.yaml tidak terbaca: {type(e).__name__}: {e})")
    print("      LLM                      : lewat LM Studio :1234 (di luar proses ini)")
    print()


if __name__ == "__main__":
    main()
