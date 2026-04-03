import time
from modules.tts_engine import EdgeTTSManager

def main():
    print("Mempersiapkan Edge TTS Manager...")
    try:
        # Inisialisasi TTS (otomatis membaca config.yaml untuk suara id-ID-GadisNeural)
        tts = EdgeTTSManager()
        print("✅ Edge TTS Siap!\n")
        
        # Kita buat fungsi generator palsu untuk mensimulasikan LLM yang sedang mengetik
        def simulasi_llm_generator():
            kalimat_kalimat = [
                "Halo! ",
                "Ini adalah suara baru saya menggunakan teknologi dari Microsoft Edge. ",
                "Bagaimana menurutmu? ",
                "Terdengar jauh lebih natural dan tidak kaku lagi, bukan?"
            ]
            
            for kalimat in kalimat_kalimat:
                # Jeda sedikit seolah-olah AI sedang berpikir/mengetik
                time.sleep(0.5) 
                yield kalimat

        print("Memulai simulasi streaming suara...\n")
        
        # Jalankan generator dan kirim ke TTS
        tts.stream_tts(simulasi_llm_generator())
        
        print("\n✅ Tes TTS selesai!")

    except Exception as e:
        print(f"\n❌ Terjadi Error: {e}")

if __name__ == "__main__":
    main()