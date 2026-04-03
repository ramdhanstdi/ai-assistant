import sys
import time
from modules.stt_engine import STTManager
from modules.llm_client import LLMClient
from modules.tts_engine import EdgeTTSManager
from modules.memory_engine import VectorDBManager

def main():
    print("="*50)
    print("🚀 MEMULAI SISTEM AI ASSISTANT LOKAL")
    print("="*50)
    
    try:
        # 1. Inisialisasi Semua Modul
        print("[1/4] Memuat STT (Ryzen CPU)...")
        stt = STTManager()
        
        print("[2/4] Memuat LLM Client (Koneksi ke Intel Arc)...")
        llm = LLMClient()
        
        print("[3/4] Memuat Edge TTS (Microsoft API)...")
        tts = EdgeTTSManager()
        
        print("[4/4] Memuat Vector DB (Memori Jangka Panjang)...")
        vectordb = VectorDBManager()
        
        print("\n✅ SEMUA SISTEM SIAP!\n")
    except Exception as e:
        print(f"❌ Gagal memuat modul: {e}")
        sys.exit(1)

    # 2. Setup System Prompt & Memory (Konteks Awal)
    messages = [
        {"role": "assistant", "content": (
            "Kamu adalah teman ngobrol virtual yang asyik. "
            "ATURAN MUTLAK: Jawablah setiap ucapan dengan SANGAT SINGKAT, padat, "
            "dan gunakan bahasa Indonesia pergaulan sehari-hari yang kasual (aku/kamu, santai). "
            "Maksimal jawabanmu HANYA 2 atau 3 kalimat pendek saja. "
            "DILARANG KERAS memberikan penjelasan panjang, membuat daftar (bullet points), atau bertele-tele, "
            "KECUALI user secara eksplisit menggunakan kata 'jelaskan', 'ceritakan', atau meminta detail lebih lanjut."
        )}
    ]

    MAX_HISTORY_TURNS = 10

    # 3. Main Loop (Siklus Percakapan)
    while True:
        try:
            print("\n" + "-"*50)
            
            # TAHAP A: MENDENGAR (STT)
            user_text = stt.listen_and_transcribe()
            
            if not user_text:
                continue # Kalau ga kedengeran apa-apa, ulang loop-nya
                
            print(f"🧑 Anda: {user_text}")
            
            # Cek perintah berhenti
            if any(kata in user_text.lower() for kata in ['berhenti', 'keluar', 'matikan', 'tutup', 'exit']):
                print("🛑 Mematikan sistem. Sampai jumpa!")
                tts.speak_chunk("Yaudah, saya matikan sistemnya. Dah!") # Optional: Kasih ucapan perpisahan
                break

            # Cek dan simpan fakta ke memori jangka panjang
            if any(kata in user_text.lower() for kata in ['namaku', 'saya adalah', 'saya suka', 'aku tinggal']):
                vectordb.save_fact(text=user_text, id=str(time.time()))

            # Masukkan ucapan user ke memori
            messages.append({"role": "user", "content": user_text})
            
            # Sliding Window Memory Management
            if len(messages) > (MAX_HISTORY_TURNS + 1):
                messages = [messages[0]] + messages[-MAX_HISTORY_TURNS:]
                
            # Injeksi Konteks (Retrieval-Augmented Generation)
            context = vectordb.search_context(user_text)
            temp_messages = list(messages)
            if context:
                temp_messages.insert(-1, {
                    "role": "system",
                    "content": f"Informasi tambahan dari memori: {context}. Gunakan informasi ini HANYA jika relevan dengan pertanyaan user."
                })

            print(f"🤖 AI: ", end="", flush=True)
            
            full_response_parts = []
            
            # Loop ini HANYA menampilkan teks ke layar (belum dikirim ke suara)
            # Menggunakan temp_messages agar system prompt sementara tidak merusak history memori chat
            for chunk in llm.stream_response(temp_messages):
                full_response_parts.append(chunk)
                print(chunk, end=" ", flush=True) 
            
            print() # Tambahkan enter setelah LLM selesai mengetik
            
            # Gabungkan jadi satu teks utuh
            full_response_text = " ".join(full_response_parts).strip()
            
            # Simpan memori obrolan
            messages.append({"role": "assistant", "content": full_response_text})

            # ==========================================================
            # TAHAP C: BERBICARA (TTS) - Baca teks utuh dengan intonasi natural
            # ==========================================================
            # Kita bungkus full_response_text ke dalam list [] 
            # agar dibaca sebagai 1 chunk raksasa oleh Edge TTS
            tts.stream_tts([full_response_text])
            
            print() # Enter setelah selesai ngomong

        except KeyboardInterrupt:
            # Kalau user tekan CTRL+C
            print("\n🛑 Sistem dihentikan paksa.")
            break
        except Exception as e:
            print(f"\n❌ Terjadi Error di Main Loop: {e}")

if __name__ == "__main__":
    main()