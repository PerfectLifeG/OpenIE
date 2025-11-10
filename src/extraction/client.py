import requests
import time, os
from typing import Dict, Any, Optional, List

class LLMClient:
    def __init__(self, cfg: Dict[str, Any]):
        """
        初始化 LLM 客户端配置
        """
        self.cfg = cfg
        self.client = build_llm_client(cfg["llm"])

    def _call_llm(self, sys_prompt: str, user_prompt: str, assistant_prompt: Optional[str] = None, fewshots: Optional[List[Dict[str, str]]] = None) -> str:
        """
        返回 LLM 的回答
        """
        resp = self.client.chat(system=sys_prompt, user=user_prompt, assistant=assistant_prompt, fewshots=fewshots)
        # print(f"[DEBUG] LLM 返回内容：{resp}")

        # 兼容两种返回结构：1) Ollama /api/chat: {"message":{"content":"..."}}
        # 2) OpenAI/vLLM /v1/chat/completions: {"choices":[{"message":{"content":"..."}}]}
        content = None
        if isinstance(resp, dict):
            # Ollama
            if "message" in resp and isinstance(resp["message"], dict):
                content = resp["message"].get("content")
            # OpenAI / vLLM
            if content is None and "choices" in resp and isinstance(resp["choices"], list) and resp["choices"]:
                content = (resp["choices"][0].get("message") or {}).get("content")

        if not content or not isinstance(content, str):
            raise RuntimeError(f"LLM 返回结构不含文本内容: {str(resp)[:500]}")

        return content
    
# ---- 两个轻量客户端 ----
class OllamaClient:
    def __init__(self, base_url, model, temperature=0.0, max_tokens=1024,
                 response_format="json_object", retry=3, timeout=60, alias_map=None):
        self.base_url = base_url.rstrip("/")
        self.model = (alias_map or {}).get(model, model)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.response_format = response_format
        self.retry = int(retry); self.timeout = int(timeout)

    def chat(self, system, user, assistant=None, fewshots=None):
        messages = []
        if system: messages.append({"role":"system","content":system})
        if assistant: messages.append({"role":"assistant","content":assistant})
        if fewshots: messages.extend(fewshots)
        messages.append({"role":"user","content":user})
        # print(f"[PROMPT]\n")
        # print("\n".join(f"[{m['role']}] {m['content']}" for m in messages))
        payload = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens}
        }
        if self.response_format == "json_object":
            payload["format"] = "json"
        url = f"{self.base_url}/api/chat"
        last = None
        for a in range(1, self.retry+1):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout)
                if r.status_code == 200: 
                    return r.json()  # {"message":{"content": "..."}}
                if r.status_code in (429,500,502,503,504):
                    last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}"); time.sleep(min(2**a,10)); continue
                r.raise_for_status()
            except Exception as e:
                last = e; time.sleep(min(2**a,10))
        raise last or RuntimeError("ollama failed")

class OpenAICompatClient:
    def __init__(self, base_url, model, api_key=None, temperature=0.0, max_tokens=1024,
                 response_format="json_object", retry=3, timeout=60):
        self.base_url = base_url.rstrip("/")
        self.model = model; self.api_key = api_key or ""
        self.temperature = float(temperature); self.max_tokens = int(max_tokens)
        self.response_format = response_format; self.retry = int(retry); self.timeout = int(timeout)

    def chat(self, system, user, assistant=None, fewshots=None):
        messages = []
        if system: messages.append({"role":"system","content":system})
        if assistant: messages.append({"role":"assistant","content":assistant})
        if fewshots: messages.extend(fewshots)
        messages.append({"role":"user","content":user})
        # print(f"[PROMPT]\n")
        # print("\n".join(f"[{m['role']}] {m['content']}" for m in messages))
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        if self.response_format == "json_object":
            body["response_format"] = {"type":"json_object"}
        headers = {"Content-Type":"application/json"}
        if self.api_key: headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/v1/chat/completions"
        last=None
        for a in range(1, self.retry+1):
            try:
                r = requests.post(url, json=body, headers=headers, timeout=self.timeout)
                if r.status_code == 200: return r.json() 
                if r.status_code in (429,500,502,503,504):
                    last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}"); time.sleep(min(2**a,10)); continue
                r.raise_for_status()
            except Exception as e:
                last=e; time.sleep(min(2**a,10))
        raise last or RuntimeError("openai-compatible failed")
    
def build_llm_client(llm_cfg: Dict[str, Any]):
    """
    根据配置构建 LLM 客户端
    """
    provider = (llm_cfg.get("provider") or "").lower()
    if provider == "ollama":
        base_url = llm_cfg.get("base_url", os.environ.get("OLLAMA_BASE_URL","http://127.0.0.1:11434"))
        alias_map = llm_cfg.get("alias_map", {
            "Llama3-8B":"llama3:8b","Llama3-8B-Instruct":"llama3:8b-instruct",
            "LLama3.1-8B":"llama3.1:8b","LLama3.1-8B-Instruct":"llama3.1:8b-instruct",
            "qwen2.5-7B-Instruct":"qwen2.5:7b-instruct","qwen3-4B-instruct":"qwen3:4b-instruct",
        })
        return OllamaClient(
            base_url=base_url, model=llm_cfg["model"],
            temperature=llm_cfg.get("temperature",0.0),
            max_tokens=llm_cfg.get("max_tokens",1024),
            response_format=llm_cfg.get("response_format","json_object"),
            retry=llm_cfg.get("retry",3),
            timeout=llm_cfg.get("timeout",60),
            alias_map=alias_map
        )
    # default: openai-compatible
    base_url = llm_cfg.get("base_url", os.environ.get("OPENAI_BASE_URL","http://127.0.0.1:8000"))
    return OpenAICompatClient(
        base_url=base_url, model=llm_cfg["model"], api_key=llm_cfg.get("api_key"),
        temperature=llm_cfg.get("temperature",0.0),
        max_tokens=llm_cfg.get("max_tokens",1024),
        response_format=llm_cfg.get("response_format","json_object"),
        retry=llm_cfg.get("retry",3), timeout=llm_cfg.get("timeout",60)
    )