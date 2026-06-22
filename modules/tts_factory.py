"""
TTS factory — pilih engine TTS berdasarkan config.yaml -> tts.engine.

Semua engine berbagi interface yang sama (stream_tts / speak_chunk), jadi pemanggil
(main.py / orchestrator) tidak perlu tahu engine mana yang aktif.

    engine: "mms"     -> MMSTTSManager   (lokal, Bahasa Indonesia, tanpa cloning) [default]
    engine: "f5_indo" -> F5TTSManager    (lokal, Bahasa Indonesia, voice cloning)
    engine: "edge"    -> EdgeTTSManager  (cloud Microsoft, legacy/fallback)
"""
import os
import yaml


def get_tts_manager(config_path="config.yaml"):
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    engine = (config.get("tts", {}).get("engine") or "mms").lower()

    if engine == "mms":
        from modules.tts_mms import MMSTTSManager
        return MMSTTSManager(config_path)
    if engine == "f5_indo":
        from modules.tts_f5 import F5TTSManager
        return F5TTSManager(config_path)
    if engine == "edge":
        from modules.tts_engine import EdgeTTSManager
        return EdgeTTSManager(config_path)

    raise ValueError(
        f"tts.engine '{engine}' tidak dikenal. Pilihan: 'mms', 'f5_indo', 'edge'."
    )
