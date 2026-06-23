"""
F5-TTS Manager — Text-to-Speech lokal Bahasa Indonesia DENGAN voice cloning.

Memakai finetune Indonesia 'Eempostor/F5-TTS-INDO-FINETUNE'. Suara ditiru (zero-shot
clone) dari satu sampel audio referensi yang kamu taruh di folder voices/
(lihat voices/README.md untuk format yang benar).

Interface SAMA dengan EdgeTTSManager/MMSTTSManager (stream_tts / speak_chunk) supaya
bisa di-swap lewat modules/tts_factory.py tanpa mengubah pemanggil.

PRASYARAT (belum tentu terpasang default):
  pip install f5-tts soundfile
  + sampel suara di voices/reference.wav  (~5-12 dtk, mono, bersih)
  + transkrip persis di voices/reference.txt

CATATAN JUJUR: nama file checkpoint/vocab spesifik finetune ini belum diverifikasi.
Set 'ckpt_file' & 'vocab_file' di config bila loader default gagal (lihat model card repo).
"""
import os
import yaml
from typing import Iterator

from modules.tts_mms import pick_device


class F5TTSManager:
    def __init__(self, config_path="config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Konfigurasi file {config_path} tidak ditemukan!")

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file) or {}

        f5_conf = self.config.get("tts", {}).get("f5_indo", {})
        self.device = pick_device(f5_conf.get("device", "auto"))
        self.ref_audio = f5_conf.get("ref_audio", "voices/reference.wav")
        self.ref_text_path = f5_conf.get("ref_text", "voices/reference.txt")
        self.ckpt_file = f5_conf.get("ckpt_file", "")   # opsional: path checkpoint finetune
        self.vocab_file = f5_conf.get("vocab_file", "") # opsional: path vocab.txt finetune
        self.model_name = f5_conf.get("model", "F5TTS_Base")

        # Validasi sampel suara untuk cloning.
        if not os.path.exists(self.ref_audio):
            raise FileNotFoundError(
                f"Sampel suara untuk cloning tidak ditemukan: {self.ref_audio}\n"
                f"   Taruh rekaman suaramu di sana (lihat voices/README.md)."
            )
        self.ref_text = ""
        if os.path.exists(self.ref_text_path):
            with open(self.ref_text_path, "r", encoding="utf-8") as f:
                self.ref_text = f.read().strip()

        # Cegah segfault konflik OpenMP ganda (Windows) saat f5-tts memuat numba/librosa
        # bersama torch/ctranslate2. Set sebelum import f5_tts (defensif bila dipakai standalone).
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        # Pre-load pyarrow LEBIH DULU. f5-tts menarik 'datasets'->'pyarrow' (lewat trainer,
        # padahal cuma perlu untuk training). Bila pyarrow dimuat di tengah rantai import
        # bersama torch/torchaudio, native lib-nya bisa access-violation (segfault) di Windows.
        # Memuatnya duluan membuat native lib-nya bersih.
        try:
            import pyarrow.dataset  # noqa: F401
        except Exception:
            pass

        # torchaudio 2.10 mengarahkan torchaudio.load() ke torchcodec yang butuh FFmpeg 4-7
        # (tidak ada / versi salah di Windows). f5-tts hanya memakai torchaudio.load untuk
        # membaca audio referensi (utils_infer.py). Kita ganti dengan loader berbasis soundfile
        # (libsndfile, tanpa FFmpeg). Output f5 sendiri sudah pakai soundfile (sf.write).
        try:
            import torchaudio as _ta
            import torch as _torch
            import soundfile as _sf

            def _load_via_soundfile(filepath, *args, **kwargs):
                data, sr = _sf.read(str(filepath), dtype="float32", always_2d=True)  # (frame, channel)
                return _torch.from_numpy(data.T).contiguous(), sr  # (channel, frame)

            _ta.load = _load_via_soundfile
        except Exception:
            pass

        # Lazy-import: f5-tts berat & opsional. Beri pesan jelas bila belum terinstall.
        try:
            from f5_tts.api import F5TTS
        except ImportError as e:
            raise ImportError(
                "Engine TTS 'f5_indo' butuh paket 'f5-tts' yang belum terinstall.\n"
                "   Pasang dulu:  pip install f5-tts torchaudio==2.10.0\n"
                "   ATAU ganti ke engine lokal lain di config.yaml -> tts.engine: \"mms\"."
            ) from e

        print(f"🔊 Memuat F5-TTS (Indo, cloning) di device '{self.device}'...")
        self.model = F5TTS(
            model=self.model_name,
            ckpt_file=self.ckpt_file,
            vocab_file=self.vocab_file,
            device=self.device,
        )
        print(f"🔊 F5-TTS siap. Suara ditiru dari: {self.ref_audio}")

    def _play_file(self, wav_path: str):
        import soundfile as sf
        import sounddevice as sd
        data, sr = sf.read(wav_path, dtype="float32")
        sd.play(data, sr)
        sd.wait()

    def stream_tts(self, text_generator: Iterator[str]):
        tmp = "temp_f5.wav"
        for text_chunk in text_generator:
            if not text_chunk or not text_chunk.strip():
                continue
            try:
                self.model.infer(
                    ref_file=self.ref_audio,
                    ref_text=self.ref_text,
                    gen_text=text_chunk.strip(),
                    file_wave=tmp,
                )
                print(f"🔊 Memutar audio: {text_chunk.strip()}")
                self._play_file(tmp)
            except Exception as e:
                print(f"❌ Error sintesis F5-TTS: {e}")
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

    def speak_chunk(self, text: str):
        self.stream_tts([text])
