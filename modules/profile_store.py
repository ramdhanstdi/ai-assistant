"""
ProfileStore — memori profil terstruktur (Langkah 6).

Fakta stabil tentang user (nama, preferensi, konfigurasi, rutinitas) yang bertahan
lintas sesi. Disimpan sebagai key-value sederhana di JSON. Beda dengan RAG:
- Profil: kecil, stabil, SELALU disuntik ke konteks (cepat, tanpa retrieval).
- RAG (vector DB): episodik/semantik, diambil per-query berdasarkan relevansi.

File default data/profile.json (folder data/ di-gitignore -> data pribadi tak ter-commit).
"""
import json
import os
import threading


class ProfileStore:
    def __init__(self, path: str = "data/profile.json"):
        self.path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def set(self, key: str, value):
        """Simpan/timpa satu fakta profil."""
        with self._lock:
            self._data[str(key).strip().lower()] = value
            self._save()

    def get(self, key: str, default=None):
        return self._data.get(str(key).strip().lower(), default)

    def forget(self, key: str) -> bool:
        with self._lock:
            if str(key).strip().lower() in self._data:
                del self._data[str(key).strip().lower()]
                self._save()
                return True
        return False

    def all(self) -> dict:
        return dict(self._data)

    def as_text(self) -> str:
        """Ringkasan profil untuk disuntik ke konteks LLM. Kosong bila belum ada."""
        if not self._data:
            return ""
        return "; ".join(f"{k}: {v}" for k, v in self._data.items())
