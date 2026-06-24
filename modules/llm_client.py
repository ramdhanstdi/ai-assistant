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
        # Model dibaca dari config (agnostik) -- sebelumnya hard-code di signature.
        self.model = llm_conf.get('model', 'openai/gpt-oss-20b')
        # Matikan "thinking/reasoning" untuk model reasoning (mis. Qwen3): tanpa ini,
        # reasoning bisa menghabiskan token budget sehingga 'content' jawaban KOSONG,
        # plus jauh lebih lambat. Aman untuk model non-reasoning (kwargs diabaikan).
        self.disable_thinking = llm_conf.get('disable_thinking', True)

        self.endpoint = f"{base_url}/chat/completions"

    def stream_response(self, messages: List[Dict[str, str]], model: str = None,
                        tools=None, tool_calls_out=None,
                        max_tokens: int = None, disable_thinking: bool = None) -> Generator[str, None, None]:
        """
        Streaming respons LLM; yield potongan kalimat (content) berdasarkan tanda baca.

        tools            : daftar schema function-calling (opsional). Bila model memutuskan
                           memanggil tool, ia mengirim tool_calls (bukan content).
        tool_calls_out   : list opsional; bila diberikan, tool_calls yang terdeteksi di stream
                           di-append ke sini sebagai dict {id, name, arguments}.
        max_tokens       : override batas token untuk panggilan ini (mis. ringkasan singkat).
        disable_thinking : override on/off reasoning untuk panggilan ini (mis. paksa cepat).
        """
        # Default ke model dari config bila pemanggil tidak menentukan.
        if model is None:
            model = self.model
        eff_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        eff_disable_thinking = self.disable_thinking if disable_thinking is None else disable_thinking

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": eff_max_tokens
        }
        if tools:
            payload["tools"] = tools
        # Nonaktifkan thinking pada model reasoning (Qwen3 dll) lewat template kwargs.
        if eff_disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            response = requests.post(self.endpoint, headers=headers, json=payload, stream=True, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Error tidak dapat terhubung ke LLM: {e}")
            return

        current_sentence = []
        punctuation_marks = ['.', ',', '!', '?', '\n', ':', ';']
        tool_acc = {}  # index -> {id, name, arguments}

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

                        # Akumulasi tool_calls (datang sebagai fragmen per index).
                        for tc in (delta.get("tool_calls") or []):
                            idx = tc.get("index", 0)
                            acc = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                            if tc.get("id"):
                                acc["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                acc["name"] = fn["name"]
                            if fn.get("arguments"):
                                acc["arguments"] += fn["arguments"]

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

        # Serahkan tool_calls yang terdeteksi ke pemanggil (bila ada).
        if tool_calls_out is not None and tool_acc:
            for idx in sorted(tool_acc):
                a = tool_acc[idx]
                if a["name"]:
                    tool_calls_out.append({"id": a["id"], "name": a["name"], "arguments": a["arguments"]})

        # Proses sisa teks yang belum terbaca/ter-yield
        if current_sentence:
            chunk_text = "".join(current_sentence).strip()
            if chunk_text:
                yield chunk_text
