"""
EpisodicLogger — catatan episodik (Langkah 7).

Mencatat SETIAP episode (sekarang: giliran obrolan) sebagai satu baris JSON
bertimestamp di data/episodes.jsonl. Skema sengaja EXTENSIBLE: log(**fields)
menerima field apa pun, jadi nanti bisa menampung episode non-teks (state sensor
robot, aksi/servo, hasil) tanpa mengubah format. Komponen pembelajaran masa depan
cukup membaca log ini (read_all) tanpa merombak sistem.

Logging dibuat best-effort: kegagalan menulis TIDAK boleh mengganggu percakapan.
File data/ di-gitignore -> data pribadi tak ter-commit.
"""
import json
import os
import threading
import datetime


class EpisodicLogger:
    def __init__(self, path: str = "data/episodes.jsonl"):
        self.path = path
        self._lock = threading.Lock()

    def log(self, **fields):
        """Tulis satu episode (timestamp ditambahkan otomatis)."""
        entry = {"ts": datetime.datetime.now().isoformat(timespec="seconds")}
        entry.update(fields)
        try:
            line = json.dumps(entry, ensure_ascii=False)
            with self._lock:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            pass  # jangan sampai logging mengganggu loop percakapan

    def read_all(self):
        """Baca semua episode (untuk analisis/pembelajaran nanti)."""
        out = []
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                out.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            except Exception:
                pass
        return out
