import json
from typing import Dict, Tuple, List, Any

from ..information_extraction import OpenIE
from .openie_openai import ChunkInfo
from ..utils.logging_utils import get_logger
from ..prompts import PromptTemplateManager
from ..llm.vllm_offline import VLLMOffline

from src.retrieval.inverted_retrieval import InvertedRetrieval

logger = get_logger(__name__)


class VLLMOfflineOpenIE(OpenIE):
    def __init__(self, global_config):

        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = VLLMOffline(global_config)
        self.global_config = global_config

    def batch_openie(self, chunks: Dict[str, ChunkInfo], temp, tp) -> Tuple[List[str], List[str]]:
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
        chunk_passages = chunks  # key:content

        # for i, (key, content) in enumerate(chunk_passages.items()):
        #     if i >= 50:
        #         break
        #     print(f"--- Chunk {i}, key: {key} ---")
        #     print(content)  # content 中的 \n 会被正确换行
        #     print("=" * 40)  # 可选，用于分隔

        # 检索
        train_path = '/home/penglin.ge/code/OpenIE/data/train2.json'
        out_dir = '/home/penglin.ge/code/OpenIE/outputs'

        retriever = InvertedRetrieval(data_path=train_path, indexdir=out_dir)

        from src.extraction.prompts.templates.few_shot import few_shot
        fallback_few_shot = few_shot

        ner_input_messages = []

        for key, passage_json in chunk_passages.items():
            data = json.loads(passage_json)
            passage_sentence = data.get("sentence", "")
            passage_schemas = data.get("schema", [])
            passage_coarse_types = data.get("coarse_types", [])
            current_passage_input = {
                "sentence": passage_sentence,
                "schema": passage_schemas,
                "coarse_types": passage_coarse_types
            }

            # --- 1. 按 schema/coarse_type 检索 topk ---
            schema_shots = [shot for schema in passage_schemas for shot in
                            retriever.retrieve_by_schema(schema, k=3, seed=42)]
            coarse_shots = [shot for ctype in passage_coarse_types for shot in
                            retriever.retrieve_by_coarse_type(ctype, k=3, seed=42)]

            # --- 2. 交集 few-shot ---
            intersect_shots = [shot for shot in schema_shots if shot in coarse_shots]

            few_shot_selected = []
            num_needed = 5

            # 按优先级选择 few-shot
            for source in [intersect_shots,
                           [s for s in schema_shots if s not in intersect_shots],
                           [s for s in coarse_shots if s not in intersect_shots],
                           fallback_few_shot]:
                take = min(len(source), num_needed)
                few_shot_selected.extend(source[:take])
                num_needed -= take
                if num_needed <= 0:
                    break

            # --- 3. 转成结构化 [input, output] ---
            few_shot_pairs = [
                (
                    {"sentence": shot.get("sentence"),
                     "schema": shot.get("schema"),
                     "coarse_types": shot.get("coarse_types")},
                    {"output": shot.get("output")}
                )
                for shot in few_shot_selected
            ]

            # --- 4. 构建 prompt ---
            prompt = self.prompt_template_manager.build_chat_prompt(
                template_name="openIE2",
                new_passage=current_passage_input,
                few_shot=few_shot_pairs
            )
            ner_input_messages.append(prompt)

        # for j, prompt in enumerate(ner_input_messages[:5]):
        #     print(f"\n=== Prompt {j} ===")
        #     for msg in prompt:
        #         print(f"[{msg['role']}]")
        #         print(msg['content'])
        #         print("-" * 30)

        # vllm_offline的批推理
        ner_output, ner_output_metadata = self.llm_model.batch_infer(
            ner_input_messages,
            json_template=self.global_config.prompt,
            max_tokens=2048,
            temp=temp,
            tp=tp,
        )

        # for i, raw_text in enumerate(ner_output):
        #     print(f"第{i + 1}条对话原始输出:\n{raw_text}\n")
        #
        # print("Metadata:", ner_output_metadata)

        return ner_output, []
