# -*- coding: utf-8 -*-
"""
最小可用的 LLM NER 抽取器（单次推理内完成“名词识别 + 类型归类”），并接入本地 Ollama。
要点：
1) 单次调用：提示里同时要求抽取 name / coarse_type / fine_type。
2) 强制 JSON：优先用 Ollama 的 format="json"，失败则做 JSON 兜底修复。
3) 幻觉抑制：名词必须来自原句（name == sentence[start:end]）。
4) 检索来源：按赛规仅用 train1.json 切块做“同域参考上下文”。
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.utils.lang import is_chinese

from client import build_llm_client

# -----------------------------------------------------------------------------
# NER 抽取器（单次推理：名词 + 类型一起抽）
# -----------------------------------------------------------------------------

class NerExtractor:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        """
        Args:
            cfg (Dict[str, Any]): 配置
        """
        self.cfg = cfg
        
        # 读取 few-shot 模板
        prompt_dir = Path(cfg["paths"]["prompt_dir"])
        fewshot_file_zh = prompt_dir / "ner_fewshot_zh.md"
        self.fewshot_prompt_zh = fewshot_file_zh.read_text(encoding="utf-8")
        self.fewshot_prompt_en = fewshot_file_zh.read_text(encoding="utf-8")
        
        self.client = build_llm_client(cfg["llm"])


        # 3) 预置系统提示（中英双语）；few-shot 由外部控制是否启用
        self.zh_system = (
            "你是严格的名词抽取器。每个名词实体只能有一个 `coarse_type` 和 `fine_type`。\n"
            "要求：只输出合法 JSON；名词必须从原文复制。\n"
            "coarse_type 必须从候选列表选择；fine_type 必须为合理的细分类名称。"
            "本任务中：fine_type 必须使用“中文短标签”，不得出现英文或拼音。"
        )
        
        # f"""
        #     你是严格的名词实体抽取器。

        #     要求：只输出合法 JSON；名词必须从原文复制。
        #     任务：
        #     - 输入给你一段话，包含一条 "sentence"，和一个 "coarse_types" 列表。
        #     - 你的任务是抽取 "sentence" 中的所有名词实体：
        #     - 对每个名词实体提供：
        #     - "name":原文中的名词实体
        #     - "coarse_type":从 "coarse_types" 中选择一个
        #     - "fine_type":基于 "sentence" 上下文决定一个更细分的类型
            
        #     输出格式：
        #     - 返回一个 JSON 对象，包含一个名词实体列表：
        #     - "entities": 一个名词实体列表，每个名词实体包含一个 "name" 字段、一个 "coarse_type" 字段 和 一个 "fine_type" 字段
            
        #     要求：
        #     - 输出必须是合法的 JSON
        #     - "coarse_type" 必须从候选列表中选择
        #     - "fine_type" 必须为合理的细分类名称
        # """
        
        
        
        self.en_system = f"""
You are an intelligent named entity recognition assistant for both Chinese and English text. Each entity must have a valid coarse type and a fine-grained type.
Task:
- You will be given a paragraph in JSON format, containing an "sentence", and a list of possible coarse entity types in "coarse_types".
- Your goal is to extract all named entities mentioned in the sentence.
- For each entity, provide:
- "name": the exact text of the entity
- "coarse_type": choose one value from the provided "coarse_types"
- "fine_type": determine a more specific type based on context
Output format:
- Return a JSON object with:
- "entities": a list of objects, each with "name", "coarse_type", "fine_type"
Requirements:
- The output must be valid JSON.
- Every "coarse_type" must be one of the types provided in the input's "coarse_types".
- Include all entities mentioned in the sentence, in either Chinese or English.
        """
        # (
        #     "You are a strict entity extractor.\n"
        #     "Output valid JSON only; each entity must be copied from the source sentence with char-level offset [start,end).\n"
        #     "coarse_type must be chosen from candidates; fine_type should be a plausible granular name."
        #     "In this task, fine_type must use short labels in Enlish, not Chinese or pinyin."
        # )
        
        
    def _process_sample(self, ex: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
        """
        处理单个样本的并发请求
        """
        try:
            parsed = self.extract(ex)  # 调用之前的 extract 方法
            results.append(parsed)
        except Exception as e:
            print(f"[ERROR] 出错处理样本 {ex.get('id')}: {e}")

    # ---------------------------- prompt 构造 ----------------------------
    def _make_user_prompt(self, ex: Dict[str, Any], is_zh: bool) -> str:
        """
        对传入的样本构造用户级 prompt
        """
        text = ex["sentence"]
        coarse = ex.get("coarse_types", [])
        domain = ex.get("domain", "")
        if domain:
            # ===== 从文件加载的 few-shot 放在最前面（md 文本） =====
            if is_zh:
                fewshot_block = self.fewshot_prompt_zh.strip()
                task_block = (
                    f"任务：这个句子属于 {domain} 领域的内容，从句子中找出属于分类集合（coarse_types）中任意分类的所有名词，并分别指明每个名词属于哪个分类。只输出 JSON。\n\n"
                    f"sentence：\n{text}\n\n"
                    "coarse_types：\n"
                    f"{coarse}\n\n"
                )
            else:
                fewshot_block = self.fewshot_prompt_en.strip()
                task_block = (
                    f"Task: This sentence belongs to the {domain} domain. Extract any words from the sentence that belong to any of the coarse types (coarse_types). Output valid JSON.\n\n"
                    f"sentence:\n{text}\n\n"
                    "coarse_types:\n"
                    f"{coarse}\n\n"
                )
        else:
            # ===== 从文件加载的 few-shot 放在最前面（md 文本） =====
            if is_zh:
                fewshot_block = self.fewshot_prompt_zh.strip()
                task_block = (
                    "任务：从句子中找出属于分类集合（coarse_types）中任意分类的所有名词，并分别指明每个名词属于哪个分类。只输出 JSON。\n\n"
                    f"sentence：\n{text}\n\n"
                    "coarse_types：\n"
                    f"{coarse}\n\n"
                )
            else:
                fewshot_block = self.fewshot_prompt_en.strip()
                task_block = (
                    "Task: Extract any words from the sentence that belong to any of the coarse types (coarse_types). Output valid JSON.\n\n"
                    f"sentence:\n{text}\n\n"
                    "coarse_types:\n"
                    f"{coarse}\n\n"
                )

        # ===== 本次样本的任务块 =====
        

        # ===== 拼接为最终 user prompt =====
        if fewshot_block:
            user = (
                f"{fewshot_block}\n\n"
                f"{task_block}"
            )
            
        else:
            user = task_block
            print("[NER] no few-shot prompt")

            # === 预览到控制台 ===
            print("\n===== PROMPT PREVIEW START =====")
            print(user)
            print("===== PROMPT PREVIEW END =====\n")

        return user
    
    def _make_assistant_prompt(self, ex: Dict[str, Any], is_zh: bool) -> Optional[str]:
        """
        对传入的样本构造助手级 prompt
        """
        # if is_zh:
        #     return self.assistant_prompt_zh
        # else:
        #     return self.assistant_prompt_en
        return None


    # ---------------------------- LLM 调用 ----------------------------
    def _call_llm(self, sys_prompt: str, user_prompt: str, assistant_prompt: Optional[str] = None, fewshots: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        resp = self.client.chat(system=sys_prompt, user=user_prompt, assistant=assistant_prompt, fewshots=fewshots)

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

        # 打印返回的原始内容，检查错误行列
        # print(f"[DEBUG] LLM 返回内容：{content}")

        try:
            return json.loads(content)
        except Exception as e:
            # 打印出错信息，检查内容是否有问题
            print(f"[ERROR] JSON 解析失败，返回内容如下：\n{content}")
            raise e

    # ---------------------------- 对外主接口 ----------------------------
    def extract(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        """
        对传入的样本调取 LLM，返回抽取结果。
        """
        is_zh = is_chinese(ex.get("sentence", ""))
        sys_prompt = self.zh_system if is_zh else self.en_system
        user_prompt = self._make_user_prompt(ex, is_zh)
        assistant_prompt = self._make_assistant_prompt(ex, is_zh)
        
        parsed = self._call_llm(sys_prompt, user_prompt, assistant_prompt)
        return {
            "id": ex.get("id", ""),
            "domain": ex.get("domain", ""), # 若不存在，则不抽取
            "sentence": ex.get("sentence", ""),
            "coarse_types": ex.get("coarse_types", []),
            "entities": parsed.get("entities", []),
        }
        
    def extract_batch(self, dataset: List[Dict[str, Any]], max_threads: int = 10) -> List[Dict[str, Any]]:
        """
        使用线程池并发处理多个请求
        """
        results = []
        
        # 创建线程处理每个样本
        with ThreadPoolExecutor(max_threads) as executor:
            for ex in dataset:
                # 使用线程池提交任务，处理每个样本
                executor.submit(self._process_sample, ex, results)
        
        # 返回处理结果
        return results
