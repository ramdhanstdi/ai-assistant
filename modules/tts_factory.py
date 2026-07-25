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

    tts_conf = config.get("tts", {})
    engine = (tts_conf.get("engine") or "mms").lower()

    if engine == "mms":
        from modules.tts_mms import MMSTTSManager
        manager = MMSTTSManager(config_path)
    elif engine == "f5_indo":
        from modules.tts_f5 import F5TTSManager
        manager = F5TTSManager(config_path)
    elif engine == "edge":
        from modules.tts_engine import EdgeTTSManager
        manager = EdgeTTSManager(config_path)
    else:
        raise ValueError(
            f"tts.engine '{engine}' tidak dikenal. Pilihan: 'mms', 'f5_indo', 'edge'."
        )

    # Override manual (opsional) atas preferensi pengucapan bawaan engine. Default engine
    # sudah masuk akal (lihat modules/io_contracts.py), jadi kunci ini hanya untuk eksperimen
    # -- mis. memaksa engine cepat pun menunggu jawaban lengkap.
    for key, attr in (("stream_sentences", "stream_by_sentence"), ("filler", "wants_filler")):
        if key in tts_conf and tts_conf[key] is not None:
            setattr(manager, attr, bool(tts_conf[key]))
            print(f"🔊 Override config: {attr} = {bool(tts_conf[key])}")

    return manager
