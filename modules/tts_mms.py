"""
MMS-TTS Manager — Text-to-Speech lokal Bahasa Indonesia (Meta MMS, model VITS).

100% offline, tanpa cloud. Bisa diakselerasi di Intel Arc (xpu) bila torch build '+xpu'
terpasang (lihat README -> Prasyarat); jika tidak, otomatis jalan di CPU.

Interface sengaja dibuat SAMA dengan EdgeTTSManager (stream_tts / speak_chunk) agar
bisa di-swap tanpa mengubah pemanggil (lihat modules/tts_factory.py).

Catatan: MMS-TTS adalah single-speaker (TIDAK bisa clone suara). Untuk voice cloning
Bahasa Indonesia gunakan engine 'f5_indo' (lihat modules/tts_f5.py).
"""
import os
import time
import yaml
from typing import Iterator


def pick_device(pref: str) -> str:
    """Pilih device. 'auto' -> Intel Arc (xpu) bila ada, selain itu 'cpu'."""
    if pref and pref != "auto":
        return pref
    try:
        import torch
        # Dukungan XPU sudah menyatu di torch inti; yang dibutuhkan hanya build '+xpu'
        # (index download.pytorch.org/whl/xpu) + driver Intel Graphics terbaru.
        # IPEX terpisah TIDAK lagi diperlukan. Cek dengan scripts/check_hardware.py.
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    return "cpu"


class MMSTTSManager:
    def __init__(self, config_path="config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Konfigurasi file {config_path} tidak ditemukan!")

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file) or {}

        mms_conf = self.config.get("tts", {}).get("mms", {})
        # Pakai folder lokal bila berisi file hasil download manual, jika tidak repo-id.
        from modules.model_paths import resolve
        self.model_id = resolve(mms_conf.get("local_dir"), mms_conf.get("model_id", "facebook/mms-tts-ind"))
        self.device = pick_device(mms_conf.get("device", "auto"))

        # Lazy-import library berat hanya saat engine ini benar-benar dipakai.
        import torch
        from transformers import VitsModel, AutoTokenizer

        self._torch = torch
        print(f"🔊 Memuat MMS-TTS '{self.model_id}' di device '{self.device}'...")
        self.model = VitsModel.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.sample_rate = self.model.config.sampling_rate
        self.model.eval()
        self._place_model()
        print(f"🔊 MMS-TTS siap di '{self.device}' (sample rate {self.sample_rate} Hz).")

    def _place_model(self):
        """
        Pindahkan model ke device pilihan, lalu warmup.

        Kegagalan GPU (driver/DLL/VRAM) sengaja TIDAK mematikan startup: asisten yang
        bicara lebih lambat di CPU tetap jauh lebih baik daripada asisten yang tidak
        bisa bicara. Warmup dijalankan di sini karena sintesis pertama di XPU harus
        mengompilasi kernel + mengalokasi VRAM (mahal) -- biar ongkos itu tidak jatuh
        pada kalimat pertama user yang sedang ditunggu telinga.
        """
        for device in ([self.device, "cpu"] if self.device != "cpu" else ["cpu"]):
            try:
                self.model.to(device)
                self.device = device
                t0 = time.perf_counter()
                self._synthesize("Halo.")
                print(f"🔊 Warmup TTS di '{device}' selesai ({time.perf_counter() - t0:.2f}s).")
                return
            except Exception as e:
                print(f"⚠️ Device '{device}' gagal dipakai MMS-TTS ({type(e).__name__}: {e}).")
                if device != "cpu":
                    print("   -> fallback ke CPU (lebih lambat). Cek scripts/check_hardware.py.")

    def _synthesize(self, text: str):
        """Teks -> waveform numpy float32 (mono)."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            waveform = self.model(**inputs).waveform
        # Kernel XPU berjalan asinkron. Sinkronkan eksplisit supaya error kernel muncul
        # DI SINI (bukan menyusul di panggilan berikutnya) & pengukuran waktu jujur.
        if self.device.startswith("xpu"):
            self._torch.xpu.synchronize()
        # bentuk (1, n) -> (n,), pindah ke cpu/numpy untuk diputar
        return waveform.squeeze(0).detach().to("cpu").float().numpy()

    def _play(self, waveform, sample_rate: int):
        """Putar waveform ke speaker (blocking) via sounddevice."""
        import sounddevice as sd
        sd.play(waveform, sample_rate)
        sd.wait()

    def stream_tts(self, text_generator: Iterator[str]):
        """Entry point utama. Terima iterator kalimat dan putar berurutan."""
        for text_chunk in text_generator:
            if not text_chunk or not text_chunk.strip():
                continue
            try:
                wav = self._synthesize(text_chunk.strip())
                print(f"🔊 Memutar audio: {text_chunk.strip()}")
                self._play(wav, self.sample_rate)
            except Exception as e:
                print(f"❌ Error sintesis MMS-TTS: {e}")

    def speak_chunk(self, text: str):
        """Kompatibilitas dengan interface lama: bungkus satu string."""
        self.stream_tts([text])
