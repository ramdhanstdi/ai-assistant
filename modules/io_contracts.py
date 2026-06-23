"""
Kontrak I/O — memisahkan "otak" (STT->LLM->memori->TTS) dari sumber & tujuan suara.

Otak tidak peduli audio datang dari mana atau keluar ke mana. Implementasi yang
bisa ditukar: LocalIO (mic/speaker PC, lihat modules/local_io.py) sekarang, dan
RobotIO (PCM via WebSocket ke ESP32) nanti di Tahap B — keduanya memenuhi kontrak
yang SAMA, jadi inti otak tidak berubah.
"""
from abc import ABC, abstractmethod
from typing import Iterator, Optional

import numpy as np


class AudioSource(ABC):
    """Penghasil input user: satu giliran bicara -> audio 16kHz mono float32."""

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """
        Ambil satu giliran audio user. Kembalikan numpy float32 (16kHz mono),
        atau None bila tidak ada suara / timeout (otak akan lanjut ke loop berikutnya).
        """
        raise NotImplementedError

    def close(self) -> None:
        """Bersihkan resource (opsional)."""
        pass


class AudioSink(ABC):
    """Penyalur suara respons (ke speaker PC, atau nanti di-stream ke robot)."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Ucapkan satu teks utuh (blocking sampai selesai diputar)."""
        raise NotImplementedError

    def speak_stream(self, text_chunks: Iterator[str]) -> None:
        """
        Ucapkan potongan-kalimat berurutan (untuk streaming/overlap di Langkah 4).
        Default: gabung lalu speak(). Implementasi boleh override agar benar-benar streaming.
        """
        for chunk in text_chunks:
            if chunk and chunk.strip():
                self.speak(chunk)

    def close(self) -> None:
        pass


class FeedbackSink(ABC):
    """
    Sinyal state pipeline (cue/wajah). Lokal cukup print/no-op; robot nanti
    memetakannya ke ekspresi + cue lokal di ESP32 (modul K).
    State umum: listening, thinking, speaking, idle, confused.
    """

    @abstractmethod
    def state(self, name: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass
