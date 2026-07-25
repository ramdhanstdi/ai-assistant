import sys
import os

# Cegah crash (segfault) konflik OpenMP ganda di Windows saat torch/ctranslate2 + numba/librosa
# (dipakai F5-TTS) dimuat berbarengan. Harus di-set SEBELUM library berat diimpor.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Plugin GPU OpenVINO (backend STT 'openvino') menyemburkan ~17 baris 'onednn_verbose,...
# error,ocl,...' ke STDOUT saat inisialisasi. Itu cuma probing OpenCL antar-GPU (Arc + iGPU
# Radeon) yang gagal lalu jalan normal -- bukan kegagalan, tapi terbaca seperti error.
# Diamkan di aplikasi; scripts/check_hardware.py sengaja TIDAK menyetel ini agar tetap terlihat
# saat mendiagnosa masalah GPU.
os.environ.setdefault("ONEDNN_VERBOSE", "0")

# Pastikan output emoji/Unicode tidak meng-crash terminal Windows lawas (cp1252).
# Tanpa ini, print berisi emoji (🔊 🎙️ ✅ dst) bisa melempar UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from modules.stt_factory import get_stt_manager
from modules.llm_client import LLMClient
from modules.tts_factory import get_tts_manager
from modules.memory_engine import VectorDBManager
from modules.local_io import LocalIO
from modules.orchestrator import Orchestrator, Session

# Kepribadian + aturan main asisten. Dipakai sebagai system prompt (role 'system').
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
        "memakai kata 'jelaskan', 'ceritakan', atau meminta penjelasan lebih detail. "
        "PENTING soal memori: bila user menyebut informasi pribadi yang stabil dan layak diingat "
        "(nama, kota/asal, pekerjaan, atau preferensi/kesukaan), SEGERA panggil tool 'ingat_profil' "
        "untuk menyimpannya (satu panggilan per fakta), baru lanjut menjawab dengan ramah."
    )
}


def main():
    print("=" * 50)
    print("🚀 MEMULAI SISTEM AI ASSISTANT LOKAL")
    print("=" * 50)

    try:
        # 1. Inisialisasi semua modul (komponen otak)
        print("[1/4] Memuat STT (backend sesuai config.yaml -> stt.backend)...")
        stt = get_stt_manager()

        print("[2/4] Memuat LLM Client (Koneksi ke Intel Arc)...")
        llm = LLMClient()

        print("[3/4] Memuat TTS (engine sesuai config.yaml -> tts.engine)...")
        tts = get_tts_manager()

        print("[4/4] Memuat Vector DB (Memori Jangka Panjang)...")
        vectordb = VectorDBManager()

        # Bungkus mic & speaker di balik kontrak I/O (LocalIO). Otak (orchestrator) memakai
        # 'io' lewat Session, bukan stt/tts langsung -> sumber/tujuan suara bisa ditukar
        # (RobotIO nanti) tanpa mengubah loop.
        io = LocalIO(stt, tts)

        print("\n✅ SEMUA SISTEM SIAP!\n")
    except Exception as e:
        print(f"❌ Gagal memuat modul: {e}")
        sys.exit(1)

    # 2. Rakit otak + sesi lokal, lalu jalankan loop percakapan.
    orchestrator = Orchestrator(stt, llm, vectordb, DEFAULT_SYSTEM_PROMPT)
    session = Session(io, orchestrator.load_messages())
    orchestrator.run(session)


if __name__ == "__main__":
    main()
