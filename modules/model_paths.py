"""
Resolver path model: pakai folder lokal bila berisi file, jika tidak fallback ke repo-id.

Tujuan: kamu bisa download model manual dari Hugging Face dan menaruhnya di folder lokal
(lihat models/README.md). Bila folder lokal ada isinya -> dipakai (offline). Bila kosong/
belum di-download -> otomatis pakai repo-id (transformers/faster-whisper auto-download).

Jadi sistem TIDAK rusak walau sebagian model belum kamu download manual.
"""
import os


def _has_files(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    for entry in os.scandir(path):
        # Anggap "berisi" bila ada minimal satu file (abaikan folder kosong & .gitkeep).
        if entry.is_file() and entry.name != ".gitkeep":
            return True
    return False


def resolve(local_dir: str, repo_id: str) -> str:
    """Kembalikan local_dir bila berisi file, selain itu repo_id."""
    if _has_files(local_dir):
        return local_dir
    return repo_id
