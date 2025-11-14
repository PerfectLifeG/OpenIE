import json
from typing import Dict, Tuple, List, Any

from ..information_extraction import OpenIE
from .openie_openai import ChunkInfo
from ..utils.logging_utils import get_logger
from ..prompts import PromptTemplateManager
from ..llm.vllm_offline import VLLMOffline
from src.extraction.prompts.templates.few_shot import *


from src.retrieval.inverted_retrieval import InvertedRetrieval
logger = get_logger(__name__)


class VLLMOfflineOpenIE(OpenIE):
    def __init__(self, global_config):

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = VLLMOffline(global_config)
        self.global_config = global_config

    def build_dynamic_prompt(self, system_msg, few_shot_examples, passage):
        # 开始模板，放 system 消息
        prompt_template = [{"role": "system", "content": system_msg}]

        # 动态插入 few-shot 示例
        for example_in, example_out in few_shot_examples:
            prompt_template.append({"role": "user", "content": example_in})
            prompt_template.append({"role": "assistant", "content": example_out})

        # 最后加入新的 passage
        prompt_template.append({"role": "user", "content": passage})

        return prompt_template


    def batch_openie(self, chunks: Dict[str, ChunkInfo],temp ,tp) -> Tuple[List[str], List[str]]:
        """
        Conduct batch OpenIE synchronously using vLLM offline batch mode, including NER and triple extraction

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk
            and the corresponding value is the chunk info to insert.  #哈希值:文本

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = chunks#key:content

        # for i, (key, content) in enumerate(chunk_passages.items()):
        #     if i >= 50:
        #         break
        #     print(f"--- Chunk {i}, key: {key} ---")
        #     print(content)  # content 中的 \n 会被正确换行
        #     print("=" * 40)  # 可选，用于分隔

        #检索
        train_path='/home/penglin.ge/code/OpenIE/data/train2.json'
        out_dir='/home/penglin.ge/code/OpenIE/outputs'
        retriever = InvertedRetrieval(data_path=train_path, indexdir=out_dir)

        fallback_few_shot = [
            (one_shot_ner_paragraph, one_shot_ner_output),
            (one_shot_ner_paragraph2, one_shot_ner_output2),
            (one_shot_ner_paragraph3, one_shot_ner_output3),
        ]

        ner_input_messages = []

        for passage in chunk_passages.values():
            data = json.loads(passage)
            passage_schemas = data.get("schema", [])
            passage_coarse_types = data.get("coarse_types", [])

            # --- 1. 按 schema 检索 topk ---
            schema_shots = []
            for schema in passage_schemas:
                schema_shots.extend(retriever.retrieve_by_schema(schema, k=3, seed=42))

            # --- 2. 按 coarse_type 检索 topk ---
            coarse_shots = []
            for ctype in passage_coarse_types:
                coarse_shots.extend(retriever.retrieve_by_coarse_type(ctype, k=3, seed=42))

            # --- 3. 交集 few-shot ---
            schema_ids = {id(ex) for ex in schema_shots}
            coarse_ids = {id(ex) for ex in coarse_shots}
            intersect_shots = [ex for ex in schema_shots if id(ex) in coarse_ids]

            few_shot_selected: List[Dict[str, Any]] = []

            # 优先取交集 few-shot
            num_needed = 3
            if intersect_shots:
                take = min(len(intersect_shots), num_needed)
                few_shot_selected.extend(intersect_shots[:take])
                num_needed -= take

            # 交集不足，再取 schema-only
            if num_needed > 0:
                schema_only_shots = [ex for ex in schema_shots if id(ex) not in {id(ex2) for ex2 in intersect_shots}]
                take = min(len(schema_only_shots), num_needed)
                few_shot_selected.extend(schema_only_shots[:take])
                num_needed -= take

            # schema + intersect 仍不足，再取 coarse-only
            if num_needed > 0:
                coarse_only_shots = [ex for ex in coarse_shots if id(ex) not in {id(ex2) for ex2 in intersect_shots}]
                take = min(len(coarse_only_shots), num_needed)
                few_shot_selected.extend(coarse_only_shots[:take])
                num_needed -= take

            # 最后不足 3 条，用 fallback 补充
            if num_needed > 0:
                few_shot_selected.extend(fallback_few_shot[:num_needed])

            # 转成 [input, output]
            few_shot_pairs = retriever.to_io_pairs(few_shot_selected)

            # --- 4. 构建 prompt ---
            prompt = self.prompt_template_manager.build_chat_prompt(
                template_name="openIE",
                new_passage=passage,
                few_shot=few_shot_pairs
            )
            ner_input_messages.append(prompt)

        for j, prompt in enumerate(ner_input_messages[:5]):
            print(f"\n=== Prompt {j} ===")
            for msg in prompt:
                print(f"[{msg['role']}]")
                print(msg['content'])
                print("-" * 30)

        #vllm_offline的批推理
        # ner_input_messages = ner_input_messages[:2]
        #
        ner_output, ner_output_metadata = self.llm_model.batch_infer(
            ner_input_messages,
            json_template=self.global_config.prompt,
            max_tokens=512,
            temp=temp,
            tp=tp,
        )
        # for i, raw_text in enumerate(ner_output):
        #     print(f"第{i + 1}条对话原始输出:\n{raw_text}\n")
        #
        # print("Metadata:", ner_output_metadata)


        return ner_output, []
