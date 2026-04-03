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
        model_size = stt_conf.get('model_size', 'base')
        device = stt_conf.get('device', 'cpu')
        compute_type = stt_conf.get('compute_type', 'int8')
        
        print(f"Loading Whisper model 'cahya/faster-whisper-medium-id' on '{device}' with type '{compute_type}'...")
        
        # Inisialisasi model faster-whisper dengan setting dinamis
        self.model = WhisperModel("cahya/faster-whisper-medium-id", device=device, compute_type=compute_type)
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True

    def listen_and_transcribe(self) -> str:
        """
        Merekam audio dari mikrofon default komputer hingga ada keheningan,
        lalu mentranskripsikan hasilnya menggunakan model faster-whisper.
        """
        with sr.Microphone() as source:
            print("🎙️ Mendengarkan... (Silakan bicara, akan berhenti otomatis setelah ada jeda keheningan)")
            
            # Anda bisa menyesuaikan noise floor ambient untuk akurasi deteksi silence yang lebih baik
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # listen() akan merekam hingga mendeteksi keheningan/silence
            try:
                audio_data = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=15.0)
            except sr.WaitTimeoutError:
                return ""
                
            print("⏳ Memproses file audio dengan Whisper...")

        # Ambil raw data 16kHz, 16-bit mono
        raw_data = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
        
        # Ubah ke numpy array float32 yang diminta faster-whisper
        audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Ambil setelan bahasa dari config
        forced_language = self.config.get('stt', {}).get('language', 'id')
        
        # Masukkan numpy array ke transcribe() dengan bahasa dipaksa
        segments, info = self.model.transcribe(audio_np, language=forced_language, task="transcribe", condition_on_previous_text=False, vad_filter=True)

        # Segment bertipe generator, sehingga kita loop dan gabungkan hasilnya
        texts = []
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            texts.append(segment.text)
            
        full_text = " ".join(texts).strip()
        
        return full_text
