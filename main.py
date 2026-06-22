import sys
import time
import json
import asyncio
import os

# Pastikan output emoji/Unicode tidak meng-crash terminal Windows lawas (cp1252).
# Tanpa ini, print berisi emoji (🔊 🎙️ ✅ dst) bisa melempar UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from modules.stt_engine import STTManager
from modules.llm_client import LLMClient
from modules.tts_factory import get_tts_manager
from modules.memory_engine import VectorDBManager

MEMORY_FILE = 'memory.json'

DEFAULT_SYSTEM_PROMPT = {
    "role": "assistant",
    "content": (
        "Kamu adalah teman ngobrol virtual yang asyik dan cerdas. "
        "ATURAN MUTLAK: Jawablah setiap ucapan dengan SANGAT SINGKAT, padat, "
        "dan gunakan bahasa Indonesia pergaulan sehari-hari yang kasual (aku/kamu, santai). "
        "Maksimal jawabanmu HANYA 3 kalimat pendek saja. "
        "DILARANG KERAS memberikan penjelasan panjang, membuat daftar (bullet points), atau bertele-tele, "
        "KECUALI user secara eksplisit menggunakan kata 'jelaskan', 'ceritakan', atau meminta detail lebih lanjut."
    )
}

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                summary = data.get("summary", "")
                history = data.get("history", [])
                
                messages = [DEFAULT_SYSTEM_PROMPT]
                if summary:
                    messages.append({"role": "system", "content": f"Context: {summary}"})
                messages.extend(history)
                return messages
        except Exception as e:
            print(f"❌ Gagal memuat file memori: {e}")
    return [DEFAULT_SYSTEM_PROMPT]

def save_memory(messages):
    summary = ""
    history = []
    
    for m in messages:
        if m.get("role") == "system" and m.get("content", "").startswith("Context:"):
            summary = m["content"].replace("Context: ", "").strip()
        elif not (m.get("role") == "assistant" and "Kamu adalah teman ngobrol virtual" in m.get("content", "")):
            history.append(m)
            
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({"summary": summary, "history": history[-10:]}, f, indent=4)
    except Exception as e:
        pass

async def summarize_now(messages, llm):
    summary_prompt = "Summarize the entire conversation so far in 3-5 concise sentences. Focus on user preferences, facts mentioned, and current topics. Output ONLY the summary."
    temp_msgs = list(messages)
    temp_msgs.append({"role": "system", "content": summary_prompt})
    
    parts = []
    for chunk in llm.stream_response(temp_msgs):
        parts.append(chunk)
        
    return "".join(parts).strip()

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
        
        print("[3/4] Memuat TTS (engine sesuai config.yaml -> tts.engine)...")
        tts = get_tts_manager()
        
        print("[4/4] Memuat Vector DB (Memori Jangka Panjang)...")
        vectordb = VectorDBManager()
        
        print("\n✅ SEMUA SISTEM SIAP!\n")
    except Exception as e:
        print(f"❌ Gagal memuat modul: {e}")
        sys.exit(1)

    # 2. Setup System Prompt & Memory (Konteks Awal)
    messages = load_memory()

    MAX_HISTORY_TURNS = 12

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
                print("⏳ Generating summary before exit...")
                try:
                    summary_text = asyncio.run(summarize_now(messages, llm))
                    history_slice = messages[-5:]
                    # Cleanup default prompt dari history_slice bila ada di indeks terbawah
                    history_slice = [m for m in history_slice if not (m.get("role") == "assistant" and "Kamu adalah teman ngobrol" in m.get("content", ""))]
                    
                    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                        json.dump({"summary": summary_text, "history": history_slice}, f, indent=4)
                except Exception as e:
                    print(f"❌ Error saat menyimpan summary: {e}")
                    
                print("🛑 Mematikan sistem. Sampai jumpa!")
                tts.speak_chunk("Yaudah, saya matikan sistemnya. Dah!") # Optional: Kasih ucapan perpisahan
                break

            # Cek dan simpan fakta ke memori jangka panjang
            if any(kata in user_text.lower() for kata in ['namaku', 'saya adalah', 'saya suka', 'aku tinggal']):
                vectordb.save_fact(text=user_text, id=str(time.time()))

            # Masukkan ucapan user ke memori
            messages.append({"role": "user", "content": user_text})
            
            # Recursive Summarization Memory Management
            if len(messages) > MAX_HISTORY_TURNS:
                # Ambil index 1 sampai len(messages) - 4
                middle_messages = messages[1:-3]
                
                # Buat temporary list untuk dikirim ke LLM
                summary_temp = list(middle_messages)
                summary_temp.append({
                    "role": "system", 
                    "content": "Rangkum percakapan ini menjadi 2 kalimat padat yang berisi fakta kunci tentang user dan topik bahasan. Jangan bertele-tele."
                })
                
                print("🔄 Mengkompresi memori (Recursive Summarization)...")
                summary_parts = []
                for chunk in llm.stream_response(summary_temp):
                    summary_parts.append(chunk)
                
                new_summary_text = "".join(summary_parts).strip()
                
                # Replace those middle messages
                messages = [messages[0]] + [{"role": "system", "content": f"Ringkasan percakapan sebelumnya: {new_summary_text}"}] + messages[-3:]
                
                # Ensure save_memory() is called after this compaction
                save_memory(messages)
                
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
            
            # Simpan ke memori persisten di setiap putaran
            save_memory(messages)
            
            print() # Enter setelah selesai ngomong

        except KeyboardInterrupt:
            # Kalau user tekan CTRL+C
            print("\n🛑 Sistem dihentikan paksa.")
            break
        except Exception as e:
            print(f"\n❌ Terjadi Error di Main Loop: {e}")

if __name__ == "__main__":
    main()