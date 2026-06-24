"""
Orchestrator — "otak" yang merangkai STT -> memori/RAG -> LLM -> TTS untuk satu Session.

Loop percakapan dipindah dari main.py ke sini. Otak TIDAK peduli sumber/tujuan suara:
semua lewat session.io (kontrak AudioSource/AudioSink/FeedbackSink). Jadi LocalIO (PC)
sekarang dan RobotIO (ESP32) nanti bisa ditukar tanpa mengubah orchestrator.
"""
import json
import os
import time
import types
import queue
import asyncio
import threading

from modules.tool_registry import build_default_registry


class Session:
    """State satu sesi percakapan: I/O (sumber/tujuan suara) + riwayat pesan."""

    def __init__(self, io, messages):
        self.io = io
        self.messages = messages


class Orchestrator:
    STOP_WORDS = ['berhenti', 'keluar', 'matikan', 'tutup', 'exit']
    FACT_WORDS = ['namaku', 'saya adalah', 'saya suka', 'aku tinggal']
    FILLER_TEXT = "Hmmmmm,"
    MAX_TOOL_HOPS = 3  # batas iterasi tool-calling agar tidak loop tak hingga

    def __init__(self, stt, llm, vectordb, system_prompt,
                 max_history_turns: int = 12, memory_file: str = "memory.json"):
        self.stt = stt
        self.llm = llm
        self.vectordb = vectordb
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.memory_file = memory_file
        # Lapisan aksi: registry tool. Context memberi tool akses ke komponen otak (vectordb).
        self.tools = build_default_registry(types.SimpleNamespace(vectordb=vectordb))

    # ===================== Memori persisten =====================
    def load_messages(self):
        """Muat riwayat dari memory_file (summary + history) + system prompt di depan."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    summary = data.get("summary", "")
                    history = data.get("history", [])

                    messages = [self.system_prompt]
                    if summary:
                        messages.append({"role": "system", "content": f"Context: {summary}"})
                    messages.extend(history)
                    return messages
            except Exception as e:
                print(f"❌ Gagal memuat file memori: {e}")
        return [self.system_prompt]

    def save_messages(self, messages):
        """Simpan ke memory_file. System prompt & ringkasan 'Context:' tidak ikut ke history."""
        summary = ""
        history = []
        for m in messages:
            if m.get("role") == "system" and m.get("content", "").startswith("Context:"):
                summary = m["content"].replace("Context: ", "").strip()
            elif not (m.get("role") == "system" and "Kamu adalah teman ngobrol virtual" in m.get("content", "")):
                history.append(m)
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump({"summary": summary, "history": history[-10:]}, f, indent=4)
        except Exception:
            pass

    async def _summarize_now(self, messages):
        summary_prompt = ("Summarize the entire conversation so far in 3-5 concise sentences. "
                          "Focus on user preferences, facts mentioned, and current topics. "
                          "Output ONLY the summary.")
        temp_msgs = list(messages)
        temp_msgs.append({"role": "system", "content": summary_prompt})
        parts = []
        for chunk in self.llm.stream_response(temp_msgs):
            parts.append(chunk)
        return "".join(parts).strip()

    # ===================== Loop utama =====================
    def run(self, session: Session):
        """Siklus percakapan: dengar -> (stop?) -> proses giliran. Berhenti via stop-word/Ctrl+C."""
        while True:
            try:
                print("\n" + "-" * 50)

                # TAHAP A: MENDENGAR — AudioSource (mic/robot) -> STT (otak)
                session.io.feedback.state("listening")
                audio = session.io.source.read()
                user_text = self.stt.transcribe(audio)
                if not user_text:
                    session.io.feedback.state("idle")
                    continue

                print(f"🧑 Anda: {user_text}")

                if any(kata in user_text.lower() for kata in self.STOP_WORDS):
                    self._shutdown(session)
                    break

                self._handle_turn(session, user_text)

            except KeyboardInterrupt:
                print("\n🛑 Sistem dihentikan paksa.")
                break
            except Exception as e:
                print(f"\n❌ Terjadi Error di Main Loop: {e}")
                session.io.feedback.state("confused")

    def _shutdown(self, session: Session):
        """Saat user minta berhenti: ringkas percakapan, simpan, pamit lewat suara."""
        print("⏳ Generating summary before exit...")
        try:
            summary_text = asyncio.run(self._summarize_now(session.messages))
            history_slice = session.messages[-5:]
            history_slice = [m for m in history_slice
                             if not (m.get("role") == "system" and "Kamu adalah teman ngobrol" in m.get("content", ""))]
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump({"summary": summary_text, "history": history_slice}, f, indent=4)
        except Exception as e:
            print(f"❌ Error saat menyimpan summary: {e}")

        print("🛑 Mematikan sistem. Sampai jumpa!")
        session.io.sink.speak("Yaudah, saya matikan sistemnya. ADIOS!")

    def _handle_turn(self, session: Session, user_text: str):
        """
        Proses satu giliran user: fakta -> memori -> RAG -> (tool-loop) LLM -> TTS streaming.

        TOOL-LOOP (Langkah 5): tiap ronde LLM dikirim daftar tool. Bila model memanggil tool,
        tool dieksekusi, hasilnya dimasukkan ke konteks, lalu ronde berikutnya menstream jawaban.
        Bila tidak ada panggilan tool, content ronde itu langsung jadi jawaban (di-stream & diucapkan,
        mempertahankan streaming per kalimat dari Langkah 4).
        """
        messages = session.messages

        # Simpan fakta ke memori jangka panjang (heuristik kata kunci)
        if any(kata in user_text.lower() for kata in self.FACT_WORDS):
            self.vectordb.save_fact(text=user_text, id=str(time.time()))

        messages.append({"role": "user", "content": user_text})

        # Recursive summarization bila riwayat terlalu panjang
        if len(messages) > self.max_history_turns:
            messages = self._compress_memory(messages)
            session.messages = messages

        # Konteks kerja untuk ronde-ronde LLM (RAG + tempat menaruh hasil tool). Tidak dipersist.
        working = self._build_with_rag(messages, user_text)

        session.io.feedback.state("thinking")
        # Filler "Hmm" paralel dengan ronde pertama; di-join sebelum suara apa pun diputar.
        filler_thread = threading.Thread(target=session.io.sink.speak, args=(self.FILLER_TEXT,), daemon=True)
        filler_thread.start()

        final_text = ""
        for hop in range(self.MAX_TOOL_HOPS + 1):
            # Ronde terakhir: paksa jawaban teks (tanpa tool) agar loop berhenti.
            tools = None if hop == self.MAX_TOOL_HOPS else self.tools.schemas()
            pre_speak = filler_thread.join if hop == 0 else None

            text, tool_calls = self._stream_round(session, working, tools=tools, pre_speak=pre_speak)

            if tool_calls and hop < self.MAX_TOOL_HOPS:
                # Catat niat memanggil tool + hasil eksekusinya ke konteks kerja.
                working.append({
                    "role": "assistant",
                    "content": text or "",
                    "tool_calls": [{"id": tc["id"], "type": "function",
                                    "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                                   for tc in tool_calls],
                })
                for tc in tool_calls:
                    try:
                        args = json.loads(tc["arguments"] or "{}")
                    except Exception:
                        args = {}
                    result = self.tools.execute(tc["name"], args)
                    print(f"   [tool: {tc['name']}({args}) -> {result[:80]}]")
                    working.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                continue  # ronde berikutnya: stream jawaban memakai hasil tool

            final_text = text
            break

        session.io.feedback.state("idle")
        messages.append({"role": "assistant", "content": final_text})
        self.save_messages(messages)
        print()

    def _stream_round(self, session: Session, working, tools=None, pre_speak=None):
        """
        Satu ronde LLM streaming: producer (LLM -> antrian) + consumer (TTS speak_stream).
        Kembalikan (teks_lengkap, tool_calls). Content (bila ada) langsung diucapkan per kalimat.
        pre_speak: callable yang dipanggil sebelum mulai mengucapkan (mis. join filler).
        """
        chunk_q: "queue.Queue" = queue.Queue()
        parts = []
        tool_calls = []

        def _produce():
            try:
                print("🤖 AI: ", end="", flush=True)
                for chunk in self.llm.stream_response(working, tools=tools, tool_calls_out=tool_calls):
                    parts.append(chunk)
                    chunk_q.put(chunk)
                print()
            finally:
                chunk_q.put(None)

        producer = threading.Thread(target=_produce, daemon=True)
        producer.start()

        if pre_speak:
            pre_speak()

        def _sentences():
            while True:
                chunk = chunk_q.get()
                if chunk is None:
                    return
                if chunk and chunk.strip():
                    yield chunk

        session.io.feedback.state("speaking")
        session.io.sink.speak_stream(_sentences())

        producer.join()
        return " ".join(parts).strip(), tool_calls

    def _compress_memory(self, messages):
        """Kompres bagian tengah riwayat jadi 2 kalimat ringkasan (recursive summarization)."""
        middle_messages = messages[1:-3]
        summary_temp = list(middle_messages)
        summary_temp.append({
            "role": "system",
            "content": "Rangkum percakapan ini menjadi 2 kalimat padat yang berisi fakta kunci tentang user dan topik bahasan. Jangan bertele-tele."
        })

        print("🔄 Mengkompresi memori (Recursive Summarization)...")
        summary_parts = []
        for chunk in self.llm.stream_response(summary_temp):
            summary_parts.append(chunk)
        new_summary_text = "".join(summary_parts).strip()

        messages = ([messages[0]]
                    + [{"role": "system", "content": f"Ringkasan percakapan sebelumnya: {new_summary_text}"}]
                    + messages[-3:])
        self.save_messages(messages)
        return messages

    def _build_with_rag(self, messages, user_text):
        """Kembalikan salinan messages dengan konteks RAG disisipkan (bila ada)."""
        context = self.vectordb.search_context(user_text)
        temp_messages = list(messages)
        if context:
            temp_messages.insert(-1, {
                "role": "system",
                "content": f"Informasi tambahan dari memori: {context}. Gunakan informasi ini HANYA jika relevan dengan pertanyaan user."
            })
        return temp_messages
