"""
STT factory — pilih backend transkripsi berdasarkan config.yaml -> stt.backend.

Pola sengaja dibuat sama dengan modules/tts_factory.py: semua backend berbagi interface
(capture / transcribe / listen_and_transcribe lewat BaseSTT), jadi pemanggil (main.py,
LocalIO, Orchestrator) tidak perlu tahu backend mana yang aktif.

    backend: "faster_whisper" -> STTManager    (CTranslate2, CPU Ryzen) [default]
    backend: "openvino"       -> OVSTTManager  (OpenVINO GenAI, Intel Arc 'GPU')
"""
import os

import yaml


def get_stt_manager(config_path="config.yaml"):
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    backend = (config.get("stt", {}).get("backend") or "faster_whisper").lower()

    if backend in ("faster_whisper", "faster-whisper", "fw"):
        from modules.stt_engine import STTManager
        return STTManager(config_path)

    if backend in ("openvino", "ov"):
        from modules.stt_ov import OVSTTManager
        try:
            return OVSTTManager(config_path)
        except Exception as e:
            # Model IR belum dikonversi / GPU bermasalah: jangan matikan asisten,
            # tapi katakan dengan lantang supaya tidak diam-diam jalan lambat di CPU.
            print(f"⚠️ Backend STT 'openvino' gagal dipakai: {type(e).__name__}: {e}")
            print("   -> fallback ke backend 'faster_whisper' (CPU).")
            from modules.stt_engine import STTManager
            return STTManager(config_path)

    raise ValueError(
        f"stt.backend '{backend}' tidak dikenal. Pilihan: 'faster_whisper', 'openvino'."
    )
