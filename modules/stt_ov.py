"""
STT backend OpenVINO GenAI — Whisper di Intel Arc B580 (device 'GPU').

Alasan keberadaan modul ini: faster-whisper memakai CTranslate2 yang TIDAK punya backend
Intel GPU, jadi STT selalu memakan CPU. OpenVINO bisa menjalankan Whisper yang sama di
Arc, membebaskan CPU untuk TTS/aplikasi lain.

Model harus dalam format OpenVINO IR (bukan CTranslate2, bukan .safetensors HF). Konversi
dilakukan SEKALI di venv terpisah — lihat models/README.md bagian "Whisper -> OpenVINO IR".
Ringkasnya:

    python -m venv venv-convert
    venv-convert\\Scripts\\python.exe -m pip install "optimum-intel[openvino]"
    venv-convert\\Scripts\\optimum-cli.exe export openvino ^
        --model cahya/whisper-medium-id --task automatic-speech-recognition ^
        --weight-format int8 models/stt/whisper-medium-id-ov

Venv terpisah itu WAJIB, bukan kerapian: optimum-intel menuntut transformers<5.1 &
safetensors<0.8.0 yang bertabrakan langsung dengan transformers 5.x + safetensors>=0.8.0
di venv runtime (konflik tak terpecahkan). Runtime hanya butuh openvino-genai.

Interface SAMA dengan STTManager (capture/transcribe/listen_and_transcribe) lewat BaseSTT,
jadi Orchestrator & LocalIO tidak perlu tahu backend mana yang aktif.
"""
import os
import time

from modules.stt_base import BaseSTT

# File penanda hasil ekspor optimum-cli. Dipakai untuk memberi pesan yang jelas
# ("belum dikonversi") alih-alih error kriptik dari dalam OpenVINO.
_MARKER = "openvino_encoder_model.xml"


class OVSTTManager(BaseSTT):
    def __init__(self, config_path="config.yaml"):
        super().__init__(config_path)

        ov_conf = self.stt_conf.get("openvino", {})
        self.model_dir = ov_conf.get("model_dir", "models/stt/whisper-medium-id-ov")
        pref_device = ov_conf.get("device", "GPU")
        # Kompilasi model ke kernel GPU ~12 detik tiap start. CACHE_DIR menyimpan hasil
        # kompilasinya ke disk sehingga start berikutnya jauh lebih cepat. Kosongkan untuk
        # menonaktifkan (mis. saat mengukur waktu kompilasi murni).
        self.cache_dir = ov_conf.get("cache_dir", "data/ov_cache")

        if not os.path.exists(os.path.join(self.model_dir, _MARKER)):
            raise FileNotFoundError(
                f"Model OpenVINO IR tidak ditemukan di '{self.model_dir}' (butuh {_MARKER}).\n"
                f"   Konversi dulu (lihat models/README.md) ATAU kembali ke backend CPU:\n"
                f"   config.yaml -> stt.backend: \"faster_whisper\""
            )

        import openvino_genai as ov_genai

        # Coba device pilihan; bila gagal (driver/VRAM/OpenCL) jatuh ke CPU. Asisten yang
        # mendengar lebih lambat tetap lebih baik daripada asisten yang tidak mau start.
        devices = [pref_device] if pref_device.upper() == "CPU" else [pref_device, "CPU"]
        props = {}
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            props["CACHE_DIR"] = self.cache_dir
        last_error = None
        for device in devices:
            try:
                print(f"🎧 Memuat Whisper OpenVINO '{self.model_dir}' di device '{device}'...")
                t0 = time.perf_counter()
                self.pipe = ov_genai.WhisperPipeline(self.model_dir, device, **props)
                self.device = device
                print(f"🎧 Model dimuat ({time.perf_counter() - t0:.1f}s), warmup...")
                self._warmup()
                print(f"🎧 STT OpenVINO siap di '{device}'.")
                return
            except Exception as e:
                last_error = e
                print(f"⚠️ Device '{device}' gagal untuk Whisper OpenVINO ({type(e).__name__}: {e}).")

        raise RuntimeError(f"Semua device gagal memuat Whisper OpenVINO. Terakhir: {last_error}")

    def _warmup(self):
        """
        Transkripsi 1 detik keheningan. Kompilasi model ke kernel GPU terjadi di panggilan
        pertama (mahal); dibayar di sini supaya ucapan pertama user tidak menunggu.
        """
        import numpy as np
        t0 = time.perf_counter()
        self._generate(np.zeros(16000, dtype=np.float32))
        print(f"🎧 Warmup selesai ({time.perf_counter() - t0:.1f}s).")

    def _generate(self, audio_np):
        # WhisperPipeline minta urutan float. numpy float32 diterima langsung; beberapa
        # build lebih rewel -> fallback ke list Python.
        kwargs = {
            "language": f"<|{self.language}|>",
            "task": "transcribe",
            "return_timestamps": False,
        }
        try:
            return self.pipe.generate(audio_np, **kwargs)
        except TypeError:
            return self.pipe.generate(audio_np.tolist(), **kwargs)

    def transcribe(self, audio_np) -> str:
        """Transkripsi audio numpy (16kHz mono float32) -> teks. Kontrak sama dengan STTManager."""
        if audio_np is None or len(audio_np) == 0:
            return ""

        t0 = time.perf_counter()
        result = self._generate(audio_np)
        elapsed = time.perf_counter() - t0

        texts = getattr(result, "texts", None)
        text = (" ".join(texts) if texts else str(result)).strip()

        audio_s = len(audio_np) / 16000.0
        print(f"[{self.device} {elapsed:.2f}s / audio {audio_s:.1f}s] {text}")
        return text
