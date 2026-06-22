# voices/ — Sampel suara untuk Voice Cloning (F5-TTS)

Folder ini tempat menaruh **rekaman suara** yang ingin ditiru (clone) oleh TTS saat
engine `f5_indo` aktif (`config.yaml -> tts.engine: "f5_indo"`).

> Engine default saat ini `mms` (MMS-TTS, suara preset Bahasa Indonesia, **tanpa cloning**).
> Folder ini baru terpakai begitu kamu pindah ke `f5_indo`.

## Cara pakai

1. Rekam / siapkan **satu** file referensi suara, simpan sebagai:
   - `voices/reference.wav`
2. Tulis **transkrip persis** dari isi rekaman itu ke:
   - `voices/reference.txt`  (teks Bahasa Indonesia, apa adanya, tanpa tanda baca berlebih)
3. Set di `config.yaml`:
   ```yaml
   tts:
     engine: "f5_indo"
   ```
4. Jalankan seperti biasa (`python main.py`). Suara balasan akan meniru `reference.wav`.

Path bisa diubah di `config.yaml -> tts.f5_indo.ref_audio` / `ref_text` bila ingin
menyimpan beberapa profil suara (mis. `voices/ayah.wav`, `voices/ibu.wav`).

## Format referensi yang disarankan (konvensi F5-TTS)

| Parameter | Anjuran |
|-----------|---------|
| Durasi    | ~5–12 detik (jangan terlalu panjang) |
| Channel   | Mono |
| Sample rate | 24000 Hz (akan di-resample otomatis bila beda, tapi 24k paling aman) |
| Format    | WAV PCM |
| Kualitas  | Bersih, tanpa musik/noise/echo, satu pembicara saja |
| Isi       | Bicara natural, jelas; transkripnya harus **sama persis** dengan audio |

## Catatan

- Rekaman audiomu = **data pribadi**. File `*.wav` / `*.mp3` di folder ini sengaja
  di-`.gitignore` agar tidak ikut ter-commit. Hanya README ini yang dilacak git.
- Checkpoint/vocab spesifik finetune `Eempostor/F5-TTS-INDO-FINETUNE` belum diverifikasi;
  bila loader default F5 gagal, isi `tts.f5_indo.ckpt_file` & `vocab_file` di config sesuai
  file di repo model tersebut.
