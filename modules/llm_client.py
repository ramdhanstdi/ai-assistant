import requests
import json
import yaml
import os
from typing import List, Dict, Generator

class LLMClient:
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Konfigurasi file {config_path} tidak ditemukan!")
            
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        llm_conf = self.config.get('llm', {})
        
        # Ambil nilai parameter dari config.yaml
        base_url = llm_conf.get('api_base_url', 'http://localhost:1234/v1')
        self.api_key = llm_conf.get('api_key', 'lm-studio')
        self.temperature = llm_conf.get('temperature', 0.7)
        self.max_tokens = llm_conf.get('max_tokens', 512)
        
        self.endpoint = f"{base_url}/chat/completions"

    def stream_response(self, messages: List[Dict[str, str]], model="openai/gpt-oss-20b") -> Generator[str, None, None]:
        """
        Mengirim messages ke LLM server dan melakukan streaming responsnya.
        Menghasilkan generator (yield) potongan kalimat berdasarkan tanda baca.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(self.endpoint, headers=headers, json=payload, stream=True, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Error tidak dapat terhubung ke LLM: {e}")
            return

        current_sentence = []
        punctuation_marks = ['.', ',', '!', '?', '\n', ':', ';']

        for line in response.iter_lines():
            if not line:
                continue
            
            line_text = line.decode('utf-8')
            if line_text.startswith("data: "):
                data_str = line_text[len("data: "):]
                
                if data_str.strip() == "[DONE]":
                    break
                    
                try:
                    data_json = json.loads(data_str)
                    choices = data_json.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        
                        if content:
                            # Cetak raw token ke terminal untuk visualisasi real-time
                            print(content, end='', flush=True)
                            
                            current_sentence.append(content)
                            
                            # Jika menemukan tanda baca akhir kalimat
                            if any(p in content for p in punctuation_marks):
                                chunk_text = "".join(current_sentence).strip()
                                if chunk_text:
                                    yield chunk_text
                                # Reset buffer kalimat
                                current_sentence = []
                except json.JSONDecodeError:
                    continue

        # Proses sisa teks yang belum terbaca/ter-yield
        if current_sentence:
            chunk_text = "".join(current_sentence).strip()
            if chunk_text:
                yield chunk_text
