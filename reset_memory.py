"""
reset_memory.py — RESET PABRIK memori asisten.

Menghapus SEMUA memori sehingga asisten mulai dari nol:
  - memory.json        : memori kerja (ringkasan + percakapan terakhir)
  - data/profile.json  : profil terstruktur (nama, preferensi)
  - data/vectordb/     : memori jangka panjang RAG (ChromaDB)

Pakai:
  python reset_memory.py          # tanya konfirmasi dulu
  python reset_memory.py --yes    # langsung hapus tanpa tanya (mis. untuk skrip)

CATATAN: hentikan dulu main.py sebelum reset — kalau masih jalan, saat keluar
ia akan menulis ulang memory.json.
"""
import os
import sys
import shutil

# Agar output emoji tidak crash di terminal Windows lawas (cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _load_vectordb_dir(config_path="config.yaml", default="data/vectordb"):
    """Ambil path ChromaDB dari config.yaml bila ada, selain itu pakai default."""
    try:
        import yaml
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("memory", {}).get("persist_directory", default)
    except Exception:
        pass
    return default


def main():
    vectordb_dir = _load_vectordb_dir()
    targets = [
        ("Memori kerja (percakapan)", "memory.json", "file"),
        ("Profil terstruktur (nama/preferensi)", "data/profile.json", "file"),
        ("Memori jangka panjang RAG (ChromaDB)", vectordb_dir, "dir"),
    ]

    print("=" * 56)
    print("  🧹 RESET PABRIK MEMORI ASISTEN")
    print("=" * 56)

    existing = []
    for label, path, kind in targets:
        ada = os.path.exists(path)
        print(f"  [{'ADA   ' if ada else 'kosong'}] {label}")
        print(f"           -> {path}")
        if ada:
            existing.append((label, path, kind))

    if not existing:
        print("\n✅ Tidak ada memori untuk dihapus. Sudah bersih.")
        return

    auto = "--yes" in sys.argv or "-y" in sys.argv
    if not auto:
        print("\n⚠️  Ini akan MENGHAPUS PERMANEN semua yang bertanda [ADA] di atas.")
        print("    Pastikan main.py sudah dihentikan.")
        try:
            jawab = input("    Lanjut hapus semua? ketik 'ya': ").strip().lower()
        except EOFError:
            jawab = ""
        if jawab not in ("ya", "y", "yes"):
            print("\n❌ Dibatalkan. Tidak ada yang dihapus.")
            return

    print()
    gagal = 0
    for label, path, kind in existing:
        try:
            if kind == "dir":
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"  ✓ dihapus: {path}")
        except Exception as e:
            gagal += 1
            print(f"  ✗ gagal hapus {path}: {e}")

    if gagal:
        print(f"\n⚠️  Selesai dengan {gagal} kegagalan (lihat di atas).")
    else:
        print("\n✅ Reset selesai. Asisten akan mulai dari nol saat dijalankan lagi.")


if __name__ == "__main__":
    main()
