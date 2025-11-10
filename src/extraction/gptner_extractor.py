from typing import List, Dict, Any, Optional
from pathlib import Path
import re, json, os

from .client import LLMClient

class EntityExtractor(LLMClient): 
    def _make_prompt_by_coarse_type(self, ex: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """
        根据传入的样本构造根据 coarse_types 抽取实体的 user_prompt
        """
        # 从 ex 中获取 coarse_types 和 sentence
        coarse_types = ex.get('coarse_types', [])
        sentence = ex.get('sentence', '')

        # 初始化 prompts 列表
        system_prompts = []
        user_prompts = []

        # 迭代所有 coarse_types 来构建 system_prompt 和 user_prompt
        for coarse_type in coarse_types:
            # 构建 system_prompt
            system_prompt = self._generate_system_prompt(coarse_type)
            system_prompts.append(system_prompt)

            # 构建 user_prompt
            user_prompt = self._generate_user_prompt(sentence, coarse_type)
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
        # 读取 system_coarse_types_prompt.txt 文件模板
        system_prompt_template = self._read_prompt("system_coarse_types_prompt.txt")

        # 替换模板中的占位符 [Entity Type] 为粗粒度类型
        system_prompt = system_prompt_template.replace("[Entity Type]", coarse_type)
        
        # TODO 替换 [examples] 为检索到的 few-shots
        return system_prompt

    def _generate_user_prompt(self, sentence: str, coarse_type: str) -> str:
        """
        根据给定的句子生成对应的 user_prompt
        """
        # 读取 user_coarse_types_prompt.txt 文件模板
        user_prompt_template = self._read_prompt("user_coarse_types_prompt.txt")

        # 替换模板中的占位符 [Sentence] 为实际的句子，替换 [Entity Type] 为粗粒度类型
        user_prompt = user_prompt_template.replace("[Sentence]", sentence).replace("[Entity Type]", coarse_type)
        
        return user_prompt
    def extract_for_one_example(self, ex: Dict[str, Any]) -> Dict[str, Any]:
        """
        对单个样例：
        1) 调用 _make_prompt_by_coarse_type 生成该 id 下所有 coarse_type 的 prompts；
        2) 逐个调用 LLM 获取“已标注的句子”；
        3) 从已标注句子中解析实体，并与对应 coarse_type 绑定；
        4) 去重汇总后返回结构化结果。

        返回结构：
        {
          "id": <str>,
          "source": <str or None>,
          "sentence": <str>,
          "coarse_types": [...],
          "entities": [ {"name": 实体, "coarse_type": 粗粒度}, ... ],
          "prompts_and_answers": [
              {"coarse_type": ct, "system_prompt": sp, "user_prompt": up, "llm_answer": ans}, ...
          ]
        }
        """
        sample_id   = ex.get("id")
        sentence    = ex.get("sentence", "")
        coarse_list = list(ex.get("coarse_types", []))

        # 1) 生成 prompts（与 coarse_types 对齐）
        system_prompts, user_prompts = self._make_prompt_by_coarse_type(ex)
        if len(system_prompts) != len(coarse_list) or len(user_prompts) != len(coarse_list):
            raise ValueError("prompts 与 coarse_types 数量不一致，请检查构造逻辑。")

        results_entities: List[Dict[str, str]] = []
        trace_items: List[Dict[str, str]] = []

        # 2) 逐 coarse_type 推理并解析
        for ct, sp, up in zip(coarse_list, system_prompts, user_prompts):
            llm_answer = self._call_llm(sys_prompt=sp, user_prompt=up)

            # 3) 解析已标注句子中的实体
            names = self._parse_tagged_entities(
                llm_answer if llm_answer else "",  # 若还未接入 LLM，则解析到空列表
                markers=self.cfg["extraction"]["entity_markers"]  # 支持外部自定义标记
            )

            # 4) 绑定 coarse_type，并做去重
            for n in names:
                if n:
                    results_entities.append({"name": n, "coarse_type": ct})

            trace_items.append({
                "coarse_type": ct,
                "system_prompt": sp,
                "user_prompt": up,
                "llm_answer": llm_answer
            })

        # 5) 去重（按 name+coarse_type）
        deduped = self._dedup_name_ct(results_entities)

        return {
            "id": sample_id,
            "source": ex.get("source"),
            "sentence": sentence,
            "coarse_types": coarse_list,
            "entities": deduped,                 # 目标产物：实体-粗粒度列表
            "prompts_and_answers": trace_items   # 便于追踪每个 coarse_type 的答案
        }

    # ===================== 核心新增：批量处理并保存 =====================
    def extract_and_save_all(self, ex_list: List[Dict[str, Any]], output_json_path: str | os.PathLike) -> Path:
        """
        对传入的样例列表逐一抽取，汇总为 JSON 并保存（若已存在则先删除后写入）。
        返回输出文件的 Path。
        """
        all_results: List[Dict[str, Any]] = []
        for idx, ex in enumerate(ex_list):
            # 对 ex 的 id 进行处理，如果 id 为空，则根据顺序生成 id
            if not ex.get("id", ""):
                ex["id"] = f"{idx}"
            one = self.extract_for_one_example(ex)
            all_results.append(one)

        out_path = Path(output_json_path)
        # 覆盖写入（若存在先删）
        if out_path.exists():
            out_path.unlink()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        return out_path

    # ===================== 工具：解析标注实体 =====================
    def _parse_tagged_entities(
        self,
        text: str,
        markers: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:
        """
        从“已标注的句子”中解析实体字符串。
        - 自定义：cfg["entity_markers"] = [{"begin":"<<","end":">>"}, ...]

        返回去重后的实体名称列表，按出现顺序稳定。
        """
        if not text:
            return []

        found: List[str] = []

        # 2) 解析 cfg 中的自定义标记对
        if markers:
            for pair in markers:
                beg = re.escape(pair.get("begin", ""))
                end = re.escape(pair.get("end", ""))
                if beg and end:
                    pattern = rf"{beg}(.*?){end}"
                    for m in re.finditer(pattern, text, flags=re.DOTALL):
                        val = m.group(1).strip()
                        if val:
                            found.append(val)

        # 3) 去重并保持次序
        seen = set()
        uniq = []
        for s in found:
            key = s
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        return uniq

    # ===================== 工具：按 (name, coarse_type) 去重 =====================
    def _dedup_name_ct(self, pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        out = []
        for obj in pairs:
            key = (obj.get("name", ""), obj.get("coarse_type", ""))
            if key not in seen and obj.get("name"):
                seen.add(key)
                out.append({"name": key[0], "coarse_type": key[1]})
        return out
