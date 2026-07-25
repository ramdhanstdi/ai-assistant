"""
Bagian STT yang TIDAK tergantung engine: baca config + rekam mikrofon PC.

Dipisah supaya backend transkripsi bisa ditukar (faster-whisper di CPU vs OpenVINO di
Intel Arc) tanpa menduplikasi kode mic. Semua subclass menjaga kontrak publik yang sama:

    capture()               -> np.float32 16kHz mono | None   (dipakai LocalMicSource)
    transcribe(audio_np)    -> str                            (dipakai Orchestrator)
    listen_and_transcribe() -> str                            (kompatibilitas lama)

Bagian capture() ini mic-spesifik (hanya LocalIO). RobotIO (Tahap B) tidak memakainya:
dia mengirim PCM dari WebSocket langsung ke transcribe().
"""
import os

import numpy as np
import speech_recognition as sr
import yaml


class BaseSTT:
    """Kerangka STT: config + mic. Subclass wajib mengisi transcribe()."""

    def __init__(self, config_path="config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Konfigurasi file {config_path} tidak ditemukan!")

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file) or {}

        self.stt_conf = self.config.get("stt", {})
        # Bahasa dipaksa (default 'id') supaya tidak buang waktu deteksi bahasa.
        self.language = self.stt_conf.get("language", "id")

        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True

    def capture(self):
        """
        Merekam audio dari mikrofon default hingga ada keheningan.
        Mengembalikan numpy array float32 (16kHz mono) atau None bila tidak ada suara.
        Bagian "mic-specific" ini dipakai oleh LocalMicSource (lihat modules/local_io.py).
        """
        with sr.Microphone() as source:
            print("🎙️ Mendengarkan... (Silakan bicara, akan berhenti otomatis setelah ada jeda keheningan)")

            # Anda bisa menyesuaikan noise floor ambient untuk akurasi deteksi silence yang lebih baik
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

            # listen() akan merekam hingga mendeteksi keheningan/silence
            try:
                audio_data = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=15.0)
            except sr.WaitTimeoutError:
                return None

            print("⏳ Memproses file audio dengan Whisper...")

        # Ambil raw data 16kHz, 16-bit mono
        raw_data = audio_data.get_raw_data(convert_rate=16000, convert_width=2)

        # Ubah ke numpy array float32 yang diminta engine STT
        return np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

    def transcribe(self, audio_np) -> str:
        raise NotImplementedError("Subclass BaseSTT harus mengimplementasi transcribe().")

    def listen_and_transcribe(self) -> str:
        """Kompatibilitas: rekam mic lalu transkrip (capture + transcribe)."""
        audio_np = self.capture()
        if audio_np is None:
            return ""
        return self.transcribe(audio_np)
