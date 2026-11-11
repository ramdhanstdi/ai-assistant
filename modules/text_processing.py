"""
Modul Pemrosesan Teks & Pemenggal Kalimat Adaptif untuk TTS (Text-to-Speech).

Masalah pada sistem lama:
1. Pemenggalan per kata/tanda baca terlalu agresif (setiap koma, titik dua, dsb langsung dipotong).
   Akibatnya TTS mengucapkan 1-2 kata secara terisolasi (mis. "Halo," atau "Tentu saja,"),
   menghasilkan intonasi datar/jatuh yang canggung, jeda hening antar-kata, dan prosodi rusak.
2. Titik pada angka/desimal (mis. 14.30, 2.5), singkatan (mis. dr., bpk., dll.), dan elipsis (...)
   sering disalahartikan sebagai akhir kalimat sehingga memotong kata di tengah jalan.
3. Model MMS-TTS (VITS) memiliki vocab terbatas dan bahkan tidak memiliki angka '3', '7', '8', '9'
   di tokenizer-nya, serta karakter markdown (**, #, bullet points) merusak sintesis suara.

Solusi di modul ini:
- `normalize_for_tts`: Membersihkan format markdown, emoji, dan mengonversi angka/jam/rupiah/persen/
  singkatan menjadi ejaan kata Bahasa Indonesia yang baku dan mengalir secara alami.
- `StreamSentenceChunker`: Pemenggal kalimat cerdas berbasis stream dari LLM yang menjaga keutuhan
  klausa, mencegah pemotongan di tengah singkatan/angka/elipsis, serta menahan koma pendek agar
  disintesis sebagai satu kesatuan intonasi yang utuh dan melodis.
"""

import re
from typing import Generator, Optional, Set, Dict

# Singkatan umum Bahasa Indonesia yang diikuti titik
ABBREVIATIONS: Set[str] = {
    "dr", "prof", "bpk", "ibu", "sdr", "sdri", "ny", "tn", "ir", "h", "hj",
    "no", "hal", "jl", "jln", "rt", "rw", "kel", "kec", "kab", "prov",
    "dll", "dsb", "dst", "sbb", "ttd", "dkk", "mis", "cth", "yaitu",
    "pt", "cv", "tbk", "corp", "inc", "ltd", "co",
    "rp", "us$", "usd", "a.n", "u.p", "d.a",
}

# Pemetaan ekspansi singkatan ke bentuk lisan yang alami
ABBREVIATION_EXPANSIONS: Dict[str, str] = {
    "dr.": "dokter",
    "prof.": "profesor",
    "bpk.": "bapak",
    "bpk": "bapak",
    "sdr.": "saudara",
    "sdri.": "saudari",
    "ny.": "nyonya",
    "tn.": "tuan",
    "ir.": "insinyur",
    "no.": "nomor",
    "hal.": "halaman",
    "jl.": "jalan",
    "jln.": "jalan",
    "dll.": "dan lain-lain",
    "dsb.": "dan sebagainya",
    "dst.": "dan seterusnya",
    "sbb.": "sebagai berikut",
    "dkk.": "dan kawan-kawan",
    "mis.": "misalnya",
    "cth.": "contohnya",
    "ttd.": "tertanda",
    "a.n.": "atas nama",
    "u.p.": "untuk perhatian",
}


def terbilang(n: int) -> str:
    """
    Konversi bilangan bulat positif/nol ke kata bahasa Indonesia.
    Mendukung rentang 0 sampai triliun.
    """
    if n == 0:
        return "nol"
    if n < 0:
        return "minus " + terbilang(abs(n))

    satuan = ["", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh", "sebelas"]

    if n < 12:
        return satuan[n]
    elif n < 20:
        return terbilang(n - 10) + " belas"
    elif n < 100:
        return terbilang(n // 10) + " puluh" + ("" if n % 10 == 0 else " " + terbilang(n % 10))
    elif n < 200:
        return "seratus" + ("" if n - 100 == 0 else " " + terbilang(n - 100))
    elif n < 1000:
        return terbilang(n // 100) + " ratus" + ("" if n % 100 == 0 else " " + terbilang(n % 100))
    elif n < 2000:
        return "seribu" + ("" if n - 1000 == 0 else " " + terbilang(n - 1000))
    elif n < 1000000:
        return terbilang(n // 1000) + " ribu" + ("" if n % 1000 == 0 else " " + terbilang(n % 1000))
    elif n < 1000000000:
        return terbilang(n // 1000000) + " juta" + ("" if n % 1000000 == 0 else " " + terbilang(n % 1000000))
    elif n < 1000000000000:
        return terbilang(n // 1000000000) + " miliar" + ("" if n % 1000000000 == 0 else " " + terbilang(n % 1000000000))
    else:
        return terbilang(n // 1000000000000) + " triliun" + ("" if n % 1000000000000 == 0 else " " + terbilang(n % 1000000000000))


def normalize_for_tts(text: str) -> str:
    """
    Normalisasi teks agar terdengar natural saat diucapkan TTS:
    - Bersihkan format Markdown (bold, italic, list, link, code block)
    - Bersihkan emoji
    - Konversi angka (rupiah, persentase, jam, desimal, bilangan bulat) ke kata Bahasa Indonesia
    - Ekspansi singkatan umum ke bahasa lisan
    """
    if not text or not text.strip():
        return ""

    s = text

    # 1. Bersihkan blok kode ```...``` dan inline code `...`
    s = re.sub(r"```[\s\S]*?```", " ", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)

    # 2. Bersihkan link [teks](url) -> teks dan bare url
    s = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", s)
    s = re.sub(r"https?://\S+", "", s)

    # 3. Bersihkan formatting markdown: header #, blockquote >
    s = re.sub(r"^[#\s>]+", "", s, flags=re.MULTILINE)
    # Bold, italic, strikethrough: **teks**, *teks*, __teks__, ~~teks~~
    s = re.sub(r"[*_~]{1,3}", "", s)

    # 4. Bersihkan bullet markers di awal baris: "- ", "* ", "+ ", "1. "
    s = re.sub(r"^\s*[-*+]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*\d+\.\s+", "", s, flags=re.MULTILINE)

    # 5. Bersihkan Emoji Unicode
    s = re.sub(r"[\U00010000-\U0010ffff]", "", s)
    s = re.sub(r"[\u2600-\u27BF]", "", s)

    # 6. Normalisasi Simbol Khusus
    s = re.sub(r"[()]", ", ", s)
    s = re.sub(r"\s*&\s*", " dan ", s)
    s = re.sub(r"\s*@\s*", " di ", s)
    s = re.sub(r"\s*=\s*", " sama dengan ", s)
    s = re.sub(r"\s*\+\s*", " plus ", s)

    # 7. Normalisasi Rupiah: Rp 50.000 / Rp. 50.000 / IDR 50.000 -> lima puluh ribu rupiah
    def _sub_rp(m):
        raw = m.group(1).replace(".", "").replace(",", "")
        try:
            val = int(raw)
            return terbilang(val) + " rupiah"
        except Exception:
            return m.group(0)
    s = re.sub(r"\b(?:Rp\.?|IDR)\s*(\d[\d\.,]*)\b", _sub_rp, s, flags=re.IGNORECASE)

    # 8. Normalisasi Persen
    s = re.sub(r"(\d+)\s*%", r"\1 persen", s)

    # 9. Normalisasi Jam / Waktu (pukul 14.30 / jam 14:30)
    def _sub_time(m):
        h = int(m.group(1))
        mn = int(m.group(2))
        h_str = terbilang(h)
        if mn == 0:
            return f"jam {h_str} tepat"
        return f"jam {h_str} lewat {terbilang(mn)}"
    s = re.sub(r"\b(?:pukul|jam)\s*(\d{1,2})[:.](\d{2})\b", _sub_time, s, flags=re.IGNORECASE)

    # 10. Normalisasi Ribuan dengan tanda titik (mis. 10.000 atau 1.500.000)
    def _sub_thousands(m):
        raw = m.group(0).replace(".", "")
        try:
            return terbilang(int(raw))
        except Exception:
            return m.group(0)
    s = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b", _sub_thousands, s)

    # 11. Normalisasi Desimal (misal 3.14 atau 28,5)
    def _sub_dec(m):
        depan = int(m.group(1))
        belakang = m.group(2)
        belakang_words = " ".join(terbilang(int(d)) for d in belakang)
        return f"{terbilang(depan)} koma {belakang_words}"
    s = re.sub(r"\b(\d+)[.,](\d+)\b", _sub_dec, s)

    # 12. Normalisasi Bilangan Bulat Biasa (misal 3, 12, 100)
    def _sub_int(m):
        try:
            return terbilang(int(m.group(0)))
        except Exception:
            return m.group(0)
    s = re.sub(r"\b\d+\b", _sub_int, s)

    # 13. Normalisasi Singkatan Umum
    for abbr, full in ABBREVIATION_EXPANSIONS.items():
        pattern = r"\b" + re.escape(abbr)
        s = re.sub(pattern, full, s, flags=re.IGNORECASE)

    # 14. Rapikan spasi berlebih
    s = re.sub(r"\s+", " ", s).strip()
    return s


class StreamSentenceChunker:
    """
    Pemenggal kalimat adaptif untuk streaming dari LLM ke TTS.

    Prinsip:
    1. Memprioritaskan tanda akhir kalimat (. ? ! atau pemisah paragraf \\n\\n).
    2. Titik pada singkatan, desimal, dan elipsis (...) tidak memotong kalimat.
    3. Koma tidak memotong kalimat kecuali kalimat sudah sangat panjang (> max_clause_chars)
       dan posisi koma sudah cukup jauh (> min_clause_chars) agar intonasi tidak patah-patah.
    4. Potongan tidak terlalu pendek (< min_chunk_chars) untuk menjaga kestabilan prosodi TTS.
    """

    def __init__(self, min_chunk_chars: int = 25, max_clause_chars: int = 100, min_clause_chars: int = 35):
        self.min_chunk_chars = min_chunk_chars
        self.max_clause_chars = max_clause_chars
        self.min_clause_chars = min_clause_chars
        self.buffer = ""

    def feed(self, token: str) -> Generator[str, None, None]:
        """Terima token baru dan yield kalimat lengkap jika sudah terbentuk."""
        self.buffer += token
        while True:
            split_idx = self._find_split_point(self.buffer, is_final=False)
            if split_idx is None:
                break
            chunk = self.buffer[:split_idx].strip()
            self.buffer = self.buffer[split_idx:].lstrip()
            if chunk:
                yield chunk

    def flush(self) -> Generator[str, None, None]:
        """Keluarkan sisa buffer saat stream LLM sudah selesai."""
        while True:
            split_idx = self._find_split_point(self.buffer, is_final=True)
            if split_idx is None:
                break
            chunk = self.buffer[:split_idx].strip()
            self.buffer = self.buffer[split_idx:].lstrip()
            if chunk:
                yield chunk
        rem = self.buffer.strip()
        self.buffer = ""
        if rem:
            yield rem

    def _find_split_point(self, text: str, is_final: bool) -> Optional[int]:
        if not text:
            return None

        # Prioritas 1: Paragraf baru (\n\n)
        nl_pos = text.find("\n\n")
        if nl_pos != -1:
            return nl_pos + 2

        n = len(text)
        for i, ch in enumerate(text):
            # Prioritas 2A: Tanda tanya (?) atau seru (!)
            if ch in ("?", "!"):
                if i + 1 == n:
                    if is_final:
                        return i + 1
                elif text[i + 1] in (" ", "\n", "\t", '"', "'", "»"):
                    candidate = text[:i + 1].strip()
                    # Khusus ? dan !, potongan pendek (mis. salam 12+ chars) boleh di-yield
                    if len(candidate) >= self.min_chunk_chars or is_final or len(candidate) >= 12:
                        return i + 1

            # Prioritas 2B: Titik (.)
            elif ch == ".":
                # Jika titik berikutnya masih titik (bagian awal/tengah ellipsis '..'), jangan potong
                if i + 1 < n and text[i + 1] == ".":
                    continue

                # Jika diapit angka (desimal misal 14.30 atau 3.14), jangan potong
                if i > 0 and i + 1 < n and text[i - 1].isdigit() and text[i + 1].isdigit():
                    continue

                # Cek apakah didahului singkatan umum
                before_dot = text[:i]
                last_word_match = re.search(r"([a-zA-Z\.\$]+)$", before_dot)
                if last_word_match:
                    last_word = last_word_match.group(1).lower()
                    if last_word in ABBREVIATIONS:
                        continue

                # Titik harus diikuti spasi/newline/kutip, atau berada di akhir stream final
                if i + 1 == n:
                    if is_final:
                        return i + 1
                elif text[i + 1] in (" ", "\n", "\t", '"', "'", "»"):
                    candidate = text[:i + 1].strip()
                    if len(candidate) >= self.min_chunk_chars or is_final:
                        return i + 1

        # Prioritas 3: Jeda klausa panjang (koma, titik dua, titik koma)
        # HANYA aktif bila buffer sudah sangat panjang (> max_clause_chars) dan belum ada titik
        if len(text) >= self.max_clause_chars:
            for i, ch in enumerate(text):
                if ch in (",", ";", ":") and i >= self.min_clause_chars and (n - i) >= 20:
                    if i + 1 == n:
                        if is_final:
                            return i + 1
                    elif text[i + 1] in (" ", "\n", "\t"):
                        return i + 1

        return None
