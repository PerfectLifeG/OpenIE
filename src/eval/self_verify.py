# -*- coding: utf-8 -*-
"""
自我验证器（SelfVerifier）
--------------------------------
功能：
1) 读取自我验证专用 Prompt 模板（system/user）；
2) 将输入 json/jsonl 中的每个样例渲染为 prompts；
3) （TODO）调用 LLM 获取自我验证结论；
4) 汇总所有样例的 prompts 与 LLM 返回，保存为一个 JSON 文件（覆盖旧文件）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 复用已有的 LLM 调用能力
from src.extraction.client import LLMClient

class SelfVerifier(LLMClient):
    """
    自我验证类：
    - 每条样本的结构如下：
    {
        "id": <str>,
        "source": <str or None>,
        "sentence": <str>,
        "coarse_types": [...],
        "entities": [ {"name": 实体, "coarse_type": 粗粒度}, ... ],
    }
    - 使用专用 Prompt 模板构造 prompts；
    - 调用 LLM 进行回答；
    - 保存所有样例验证结果。
    """
    def _make_coarse_type_verify_prompt(self, example: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """
        根据传入的 json 样本构造 prompts
        """
        # print("[INFO] 构造 verify prompts")
        entities = example.get("entities", [])
        sentence = example.get("sentence", "")
        
        system_prompts = []
        user_prompts = []
        
        for entity in entities:
            name = entity.get("name", "")
            coarse_type = entity.get("coarse_type", "")
            system_prompt = self._generate_system_prompt(coarse_type)
            user_prompt = self._generate_user_prompt(name, coarse_type, sentence)
            
            system_prompts.append(system_prompt)
            user_prompts.append(user_prompt)
        return system_prompts, user_prompts
    
    def _read_prompt(self, filename: str) -> str:
        """从 cfg.paths.prompt_dir 读取模板文本"""
        prompt_dir = Path(self.cfg.get("paths", {}).get("prompt_dir", "./prompts"))
        return (prompt_dir / filename).read_text(encoding="utf-8")
    
    def _generate_system_prompt(self, coarse_type: str) -> str:
        """
        根据粗粒度类型生成对应的 system_prompt
        """
        template = self._read_prompt("system_coarse_type_verify_prompt.txt")
        # TODO 替换 [examples] 为检索到的 few-shots
        return (template
            .replace("[Entity Type]", coarse_type))


    def _generate_user_prompt(self, name: str, coarse_type: str, sentence: str) -> str:
        """
        根据给定的句子生成对应的 user_prompt
        """
        template = self._read_prompt("user_coarse_type_verify_prompt.txt")
        
        return (template
                .replace("[Sentence]", sentence)
                .replace("[Entity]", name)
                .replace("[Entity Type]", coarse_type))
        
        # ===================== 工具：解析 yes/no =====================
    def _parse_yes_no(self, text: Optional[str]) -> Optional[bool]:
        """
        解析 LLM 的回答为 True/False：
        - True: yes / true / 是 / 对 / 正确 / 属于 / Y（大小写不敏感）
        - False: no / false / 否 / 不 / 不是 / 不属于 / N
        - 其他：返回 None
        """
        if not text:
            return None
        s = str(text).strip().lower()

        # 只保留首段，避免长文本里首句就是答案
        first = s.splitlines()[0].strip()

        true_set = {"yes", "y", "true", "是", "对", "正确", "属于", "是的"}
        false_set = {"no", "n", "false", "否", "不是", "不", "不属于"}

        # 精准匹配
        if first in true_set:
            return True
        if first in false_set:
            return False

        # 宽松判断：句子里包含明显肯/否定词
        for t in true_set:
            if t in first:
                return True
        for t in false_set:
            if t in first:
                return False

        return None

    # ===================== 对单条样例的自我验证 =====================
    def verify_for_one_example(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        """
        对单条样例进行自我验证：
        1) 使用 _make_coarse_type_verify_prompt 生成 (system_prompts, user_prompts)；
        2) 逐条调用 LLM 获取 yes/no；
        3) 解析为布尔值并汇总返回。

        返回结构：
        {
          "id": <str>,
          "source": <str or None>,
          "sentence": <str>,
          "coarse_types": [...],
          "entities": [...],  # 原始实体（name, coarse_type）
          "verification": [
              {"name": ..., "coarse_type": ..., "is_valid": true/false/None, "llm_answer": "..."},
              ...
          ],
          "verified_entities": [ {"name": ..., "coarse_type": ...}, ... ],  # 仅保留判定为 True 的
          "prompts_and_answers": [
              {"name": ..., "coarse_type": ..., "system_prompt": sp, "user_prompt": up, "llm_answer": ans},
              ...
          ]
        }
        """
        sample_id = ex.get("id")
        sentence = ex.get("sentence", "")
        coarse_list = list(ex.get("coarse_types", []))
        entities = list(ex.get("entities", []))  # [{"name":..., "coarse_type":...}, ...]

        # 1) 生成 prompts（与 entities 数量对齐）
        system_prompts, user_prompts = self._make_coarse_type_verify_prompt(ex)
        if len(system_prompts) != len(entities) or len(user_prompts) != len(entities):
            raise ValueError("prompts 与 entities 数量不一致，请检查 _make_coarse_type_verify_prompt 的构造逻辑。")

        verification_items: List[Dict[str, Any]] = []
        trace_items: List[Dict[str, Any]] = []

        # 2) 逐实体进行验证
        for ent, sp, up in zip(entities, system_prompts, user_prompts):
            name = ent.get("name", "")
            ct = ent.get("coarse_type", "")
            llm_answer = self._call_llm(sys_prompt=sp, user_prompt=up)  # 返回字符串
            yn = self._parse_yes_no(llm_answer)

            verification_items.append({
                "name": name,
                "coarse_type": ct,
                "is_valid": yn,
                "llm_answer": llm_answer
            })

            trace_items.append({
                "name": name,
                "coarse_type": ct,
                "system_prompt": sp,
                "user_prompt": up,
                "llm_answer": llm_answer
            })

        # 3) 提取判定为 True 的实体
        verified_entities = [
            {"name": v["name"], "coarse_type": v["coarse_type"]}
            for v in verification_items
            if v["is_valid"] is True
        ]

        return {
            "id": sample_id,
            "source": ex.get("source"),
            "sentence": sentence,
            "coarse_types": coarse_list,
            "entities": entities,                 # 原始抽取的实体
            "verification": verification_items,   # 每个实体的验证结论
            "verified_entities": verified_entities,
            "prompts_and_answers": trace_items    # 便于追踪
        }

    # ===================== 批量处理并保存 =====================
    def verify_and_save_all(self, ex_list: List[Dict[str, Any]], output_json_path: Union[str, os.PathLike]) -> Path:
        """
        对传入的样例列表逐一自我验证，汇总为 JSON 并保存（若已存在则先删除后写入）。
        返回输出文件的 Path。
        """
        all_results: List[Dict[str, Any]] = []
        for idx, ex in enumerate(ex_list):
            # 若 id 为空则按顺序生成
            if not ex.get("id", ""):
                ex["id"] = f"{idx}"
            one = self.verify_for_one_example(ex)
            all_results.append(one)

        out_path = Path(output_json_path)
        if out_path.exists():
            out_path.unlink()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        return out_path
