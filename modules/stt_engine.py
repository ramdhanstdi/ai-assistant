"""
STT backend faster-whisper (CTranslate2) — berjalan di CPU (Ryzen 8500G).

CATATAN HARDWARE: CTranslate2 tidak punya backend Intel GPU, jadi backend ini SELALU
CPU (atau CUDA di mesin NVIDIA). Untuk memakai Intel Arc, pilih backend 'openvino'
di config.yaml -> stt.backend (lihat modules/stt_ov.py).

Bagian mic (capture) ada di modules/stt_base.py, dibagi dengan backend lain.
"""
import os

from faster_whisper import WhisperModel

from modules.stt_base import BaseSTT


class STTManager(BaseSTT):
    def __init__(self, config_path="config.yaml"):
        super().__init__(config_path)

        stt_conf = self.stt_conf

        # Ambil nilai parameter dinamis dari config.yaml
        # 'model' kini dibaca dari config (sebelumnya hard-code). Fallback ke 'model_size' lama.
        # Pakai folder lokal (local_dir) bila berisi file hasil download manual, jika tidak repo-id.
        from modules.model_paths import resolve
        repo_id = stt_conf.get('model', stt_conf.get('model_size', 'cahya/faster-whisper-medium-id'))
        model_name = resolve(stt_conf.get('local_dir'), repo_id)
        device = stt_conf.get('device', 'cpu')
        compute_type = stt_conf.get('compute_type', 'int8')

        # Tuning kecepatan: cpu_threads (0 = semua core) & beam_size (1 = greedy, lebih cepat).
        cpu_threads = stt_conf.get('cpu_threads', 0) or os.cpu_count()
        self.beam_size = stt_conf.get('beam_size', 1)

        print(f"Loading Whisper model '{model_name}' on '{device}' with type '{compute_type}' "
              f"(cpu_threads={cpu_threads}, beam_size={self.beam_size})...")

        # Inisialisasi model faster-whisper dengan setting dinamis
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=cpu_threads)

    def transcribe(self, audio_np) -> str:
        """
        Transkripsi audio numpy (16kHz mono float32) menjadi teks.
        Ini bagian "otak" yang reusable: dipakai untuk audio dari mic (LocalIO)
        MAUPUN dari robot (RobotIO/PCM via WebSocket nanti). Sumber audio tidak penting.
        """
        if audio_np is None or len(audio_np) == 0:
            return ""

        # Masukkan numpy array ke transcribe() dengan bahasa dipaksa
        segments, info = self.model.transcribe(audio_np, language=self.language, task="transcribe", condition_on_previous_text=False, vad_filter=True, beam_size=self.beam_size)

        # Segment bertipe generator, sehingga kita loop dan gabungkan hasilnya
        texts = []
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            texts.append(segment.text)

        return " ".join(texts).strip()
