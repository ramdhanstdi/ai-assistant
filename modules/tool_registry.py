"""
Tool registry — lapisan aksi (Langkah 5).

Tiap tool = fungsi + schema (format OpenAI function-calling). LLM memilih tool;
Orchestrator mengeksekusinya lalu merangkai hasilnya ke jawaban.

Tool ditulis sebagai fungsi `fn(ctx, **args) -> str`. `ctx` memberi akses ke
komponen otak (mis. vectordb) tanpa meng-hardcode dependensi di dalam tool.
Tambah tool baru cukup daftarkan di build_default_registry().
"""
import datetime


class ToolRegistry:
    def __init__(self, context):
        self.context = context
        self._tools = {}  # name -> (schema, fn)

    def add(self, name, description, parameters, fn):
        schema = {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        }
        self._tools[name] = (schema, fn)

    def schemas(self):
        """Daftar schema untuk dikirim ke LLM (None bila tidak ada tool)."""
        return [s for s, _ in self._tools.values()] or None

    def has_tools(self):
        return bool(self._tools)

    def execute(self, name, arguments: dict) -> str:
        """Jalankan tool dan kembalikan hasil sebagai teks (untuk diberikan balik ke LLM)."""
        if name not in self._tools:
            return f"Tool '{name}' tidak dikenal."
        _, fn = self._tools[name]
        try:
            return str(fn(self.context, **(arguments or {})))
        except Exception as e:
            return f"Error menjalankan tool '{name}': {e}"


# ===================== Tool bawaan =====================

def _get_waktu(ctx):
    """Waktu & tanggal lokal saat ini, dalam Bahasa Indonesia sederhana."""
    now = datetime.datetime.now()
    hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][now.weekday()]
    return now.strftime(f"{hari}, %d-%m-%Y, jam %H:%M")


def _cari_memori(ctx, query: str):
    """Cari fakta/preferensi user dari memori jangka panjang (vector DB)."""
    if ctx is None or getattr(ctx, "vectordb", None) is None:
        return "Memori tidak tersedia."
    hasil = ctx.vectordb.search_context(query)
    return hasil if hasil else "Tidak ada yang relevan di memori."


def _ingat_profil(ctx, kunci: str, nilai: str):
    """Simpan fakta stabil user ke profil terstruktur (mis. nama, kota, preferensi)."""
    if ctx is None or getattr(ctx, "profile", None) is None:
        return "Profil tidak tersedia."
    ctx.profile.set(kunci, nilai)
    return f"Tersimpan ke profil: {kunci} = {nilai}"


def build_default_registry(context):
    """Bangun registry berisi tool bawaan. `context` butuh atribut .vectordb."""
    reg = ToolRegistry(context)
    reg.add(
        "get_waktu",
        "Ambil tanggal, hari, dan jam saat ini. Pakai bila user menanyakan waktu/tanggal/hari.",
        {"type": "object", "properties": {}, "required": []},
        _get_waktu,
    )
    reg.add(
        "cari_memori",
        "Cari fakta atau preferensi user yang tersimpan di memori jangka panjang.",
        {"type": "object",
         "properties": {"query": {"type": "string", "description": "kata kunci yang dicari"}},
         "required": ["query"]},
        _cari_memori,
    )
    reg.add(
        "ingat_profil",
        "Simpan fakta stabil & penting tentang user agar diingat lintas sesi "
        "(mis. nama, kota tinggal, preferensi). Pakai saat user memberi info pribadi yang layak diingat.",
        {"type": "object",
         "properties": {
             "kunci": {"type": "string", "description": "label fakta, mis. 'nama', 'kota', 'minuman_favorit'"},
             "nilai": {"type": "string", "description": "isi faktanya, mis. 'Ramdhan', 'Bandung', 'kopi'"}},
         "required": ["kunci", "nilai"]},
        _ingat_profil,
    )
    return reg
