import re

class Router:
    @staticmethod
    def identify_intent(text: str) -> str:
        """
        Melakukan NLP sederhana menggunakan regex mencari base intents.
        Intent yang ditangani:
        - "reset": Jika mengandung 'reset' atau 'hapus ingatan'
        - "vision": Jika mengandung 'lihat', 'kamera', atau 'foto'
        - "chat": Sisanya
        """
        text_lower = text.lower()
        
        # Cek intent reset memory
        if re.search(r'\b(reset|hapus ingatan)\b', text_lower):
            return "reset"
            
        # Cek intent penglihatan (vision)
        if re.search(r'\b(lihat|kamera|foto)\b', text_lower):
            return "vision"
            
        # Percakapan standar
        return "chat"
