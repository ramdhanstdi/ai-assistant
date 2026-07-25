"""
LocalIO — implementasi kontrak I/O untuk PC (mic + speaker).

Membungkus kode yang SUDAH jalan tanpa menulis ulang:
- LocalMicSource  -> stt.capture() (rekam mic via SpeechRecognition; lihat BaseSTT)
- LocalSpeakerSink-> engine TTS dari get_tts_manager() (stream_tts ke speaker)
- LocalFeedbackSink-> cetak state ke terminal (cue robot belum ada di PC)

Nanti RobotIO menyediakan implementasi lain dari kontrak yang sama (Tahap B),
sehingga orchestrator/otak tidak perlu diubah.
"""
from typing import Iterator, Optional

import numpy as np

from modules.io_contracts import AudioSource, AudioSink, FeedbackSink


class LocalMicSource(AudioSource):
    """AudioSource dari mikrofon PC. Membungkus capture() dari backend STT mana pun."""

    def __init__(self, stt):
        self._stt = stt

    def read(self) -> Optional[np.ndarray]:
        return self._stt.capture()


class LocalSpeakerSink(AudioSink):
    """AudioSink ke speaker PC. Membungkus engine TTS (MMS/F5/Edge)."""

    def __init__(self, tts):
        self._tts = tts

    def speak(self, text: str) -> None:
        if text and text.strip():
            self._tts.stream_tts([text])

    def speak_stream(self, text_chunks: Iterator[str]) -> None:
        # Engine TTS sudah menerima iterator kalimat -> serahkan langsung (siap untuk Langkah 4).
        self._tts.stream_tts(text_chunks)


class LocalFeedbackSink(FeedbackSink):
    """
    FeedbackSink untuk PC: hanya cetak state (debug). Tidak ada wajah/cue di PC.
    Robot nanti mengirim state ini ke ESP32.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def state(self, name: str) -> None:
        if self.verbose:
            print(f"   [state: {name}]")


class LocalIO:
    """
    Bundel I/O lokal: source (mic) + sink (speaker) + feedback (print).
    Otak cukup menerima objek ini dan tidak peduli implementasinya.
    """

    def __init__(self, stt, tts, feedback_verbose: bool = True):
        self.source: AudioSource = LocalMicSource(stt)
        self.sink: AudioSink = LocalSpeakerSink(tts)
        self.feedback: FeedbackSink = LocalFeedbackSink(verbose=feedback_verbose)

    def close(self) -> None:
        self.source.close()
        self.sink.close()
        self.feedback.close()
