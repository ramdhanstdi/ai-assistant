import sys
import time
import json
import asyncio
import os
import threading

# Cegah crash (segfault) konflik OpenMP ganda di Windows saat torch/ctranslate2 + numba/librosa
# (dipakai F5-TTS) dimuat berbarengan. Harus di-set SEBELUM library berat diimpor.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
from modules.local_io import LocalIO

MEMORY_FILE = 'memory.json'

DEFAULT_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Kamu adalah teman ngobrol virtual yang asyik, cerdas, dan terdengar natural seperti manusia. "
        "Bicaralah dengan Bahasa Indonesia sehari-hari yang santai dan akrab (pakai aku/kamu), "
        "mengalir dan tidak kaku. "
        "PENTING — karena jawabanmu akan DIBACAKAN dengan suara, perhatikan penulisannya: "
        "1) Pakai tanda baca yang lengkap dan benar: titik di akhir kalimat, koma untuk jeda, "
        "serta tanda tanya atau tanda seru sesuai konteks, supaya intonasinya pas saat diucapkan. "
        "2) Pakai kapitalisasi yang benar: huruf besar di awal tiap kalimat dan pada nama orang, "
        "tempat, atau merek. "
        "3) Tulis kalimat yang utuh dan wajar; jangan huruf kecil semua dan jangan singkatan aneh. "
        "Tetap RINGKAS dan padat: cukup 1 sampai 3 kalimat pendek yang natural. "
        "Jangan bertele-tele dan jangan membuat daftar berpoin, KECUALI user secara eksplisit "
        "memakai kata 'jelaskan', 'ceritakan', atau meminta penjelasan lebih detail."
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
        elif not (m.get("role") == "system" and "Kamu adalah teman ngobrol virtual" in m.get("content", "")):
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

        # Bungkus mic & speaker di balik kontrak I/O. Otak memakai 'io', bukan stt/tts langsung,
        # jadi sumber/tujuan suara bisa ditukar (RobotIO nanti) tanpa mengubah loop ini.
        io = LocalIO(stt, tts)

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
            
            # TAHAP A: MENDENGAR — AudioSource (mic lokal) -> STT (otak)
            io.feedback.state("listening")
            audio = io.source.read()
            user_text = stt.transcribe(audio)
            if not user_text:
                io.feedback.state("idle")
                continue # Kalau ga kedengeran apa-apa, ulang loop-nya
                
            print(f"🧑 Anda: {user_text}")
            
            # Cek perintah berhenti
            if any(kata in user_text.lower() for kata in ['berhenti', 'keluar', 'matikan', 'tutup', 'exit']):
                print("⏳ Generating summary before exit...")
                try:
                    summary_text = asyncio.run(summarize_now(messages, llm))
                    history_slice = messages[-5:]
                    # Cleanup default prompt dari history_slice bila ada di indeks terbawah
                    history_slice = [m for m in history_slice if not (m.get("role") == "system" and "Kamu adalah teman ngobrol" in m.get("content", ""))]
                    
                    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                        json.dump({"summary": summary_text, "history": history_slice}, f, indent=4)
                except Exception as e:
                    print(f"❌ Error saat menyimpan summary: {e}")
                    
                print("🛑 Mematikan sistem. Sampai jumpa!")
                io.sink.speak("Yaudah, saya matikan sistemnya. ADIOS!") # Optional: Kasih ucapan perpisahan
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

            # FILLER PARALEL (latency masking): putar "Hmm" di thread terpisah BERSAMAAN
            # dengan LLM yang sedang berpikir. Filler memberi respons instan ke user
            # sementara jawaban asli digenerate. Di-join sebelum jawaban diputar (lihat bawah)
            # agar tidak bertabrakan di audio device.
            io.feedback.state("thinking")
            filler_thread = threading.Thread(target=io.sink.speak, args=("Hmmmmm,",), daemon=True)
            filler_thread.start()

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
            # Tunggu filler "Hmm" selesai diputar dulu agar tidak tabrakan dengan jawaban
            # (biasanya sudah selesai karena LLM lebih lama). Lalu putar jawaban asli.
            filler_thread.join()

            # Salurkan jawaban ke AudioSink sesi (speaker lokal sekarang; robot nanti).
            io.feedback.state("speaking")
            io.sink.speak(full_response_text)
            io.feedback.state("idle")

            # Simpan ke memori persisten di setiap putaran
            save_memory(messages)

            print() # Enter setelah selesai ngomong

        except KeyboardInterrupt:
            # Kalau user tekan CTRL+C
            print("\n🛑 Sistem dihentikan paksa.")
            break
        except Exception as e:
            print(f"\n❌ Terjadi Error di Main Loop: {e}")
            io.feedback.state("confused")

if __name__ == "__main__":
    main()