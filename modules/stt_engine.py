import yaml
import os
import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel

class STTManager:
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Konfigurasi file {config_path} tidak ditemukan!")
            
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        stt_conf = self.config.get('stt', {})

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

        # Ubah ke numpy array float32 yang diminta faster-whisper
        return np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

    def transcribe(self, audio_np) -> str:
        """
        Transkripsi audio numpy (16kHz mono float32) menjadi teks.
        Ini bagian "otak" yang reusable: dipakai untuk audio dari mic (LocalIO)
        MAUPUN dari robot (RobotIO/PCM via WebSocket nanti). Sumber audio tidak penting.
        """
        if audio_np is None or len(audio_np) == 0:
            return ""

        # Ambil setelan bahasa dari config
        forced_language = self.config.get('stt', {}).get('language', 'id')

        # Masukkan numpy array ke transcribe() dengan bahasa dipaksa
        segments, info = self.model.transcribe(audio_np, language=forced_language, task="transcribe", condition_on_previous_text=False, vad_filter=True, beam_size=self.beam_size)

        # Segment bertipe generator, sehingga kita loop dan gabungkan hasilnya
        texts = []
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            texts.append(segment.text)

        return " ".join(texts).strip()

    def listen_and_transcribe(self) -> str:
        """Kompatibilitas: rekam mic lalu transkrip (capture + transcribe)."""
        audio_np = self.capture()
        if audio_np is None:
            return ""
        return self.transcribe(audio_np)
