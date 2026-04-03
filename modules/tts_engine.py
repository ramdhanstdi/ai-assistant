import yaml
import os
import asyncio
import edge_tts
import pygame
from typing import Iterator

class EdgeTTSManager:
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Konfigurasi file {config_path} tidak ditemukan!")
            
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        tts_conf = self.config.get('tts', {})
        
        # Ambil nilai parameter dari config.yaml
        self.voice = tts_conf.get('voice', 'id-ID-GadisNeural')
        self.rate = tts_conf.get('rate', '+0%')
        self.volume = tts_conf.get('volume', '+0%')
        
        # Inisialisasi library PyGame Mixer untuk memutar audio
        pygame.mixer.init()
        print(f"🔊 Edge TTS Model siap. Menggunakan suara: {self.voice}")

    async def _generate_and_play(self, text: str):
        """
        Mensintesis text menjadi audio menggunakan Edge TTS,
        menyimpannya ke temp file, lalu memutarnya secara sinkron
        menggunakan PyGame Mixer.
        """
        temp_filename = "temp_chunk.mp3"
        
        try:
            # 1. Konfigurasi objek sintesis Edge TTS
            communicate = edge_tts.Communicate(
                text=text, 
                voice=self.voice, 
                rate=self.rate, 
                volume=self.volume
            )
            
            # 2. Simpan audio stream ke file sementara
            await communicate.save(temp_filename)
            
            # 3. Load dan Play audio menggunakan pygame.mixer.music
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            
            print(f"🔊 Memutar audio: {text}")
            
            # 4. Tunggu / blocking secara asynchronous sampai durasi audio selesai diputar
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
                
            # 5. Hapus file sementara dengan melepaskan referensi filenya
            pygame.mixer.music.unload()
            try:
                os.remove(temp_filename)
            except OSError as e:
                # Terkadang library belum melepaskan lock file tepat pada waktunya
                print(f"⚠️ Gagal menghapus {temp_filename}: {e}")
                
        except Exception as e:
            print(f"❌ Error saat proses sintesis TTS: {e}")

    def stream_tts(self, text_generator: Iterator[str]):
        """
        Entry point utama yang dipanggil oleh loop utama.
        Menerima generator kalimat (chunks) dan memutarnya berurutan.
        """
        for text_chunk in text_generator:
            if not text_chunk or not text_chunk.strip():
                continue
            
            # Karena metode di atas async secara native, kita jalankan satu-satu 
            # untuk setiap chunk agar tidak bertabrakan dengan menggunakan asyncio.run
            asyncio.run(self._generate_and_play(text_chunk))

    def speak_chunk(self, text: str):
        """
        Kompatibilitas wrapper untuk method TTS lama (jika main.py mengirimkannya string statis per-chunk 
        alih-alih generator). Ini membuat array berisi 1 elemen teks ke stream_tts().
        """
        self.stream_tts([text])
