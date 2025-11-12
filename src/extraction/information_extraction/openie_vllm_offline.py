from typing import Dict, Tuple, List

from ..information_extraction import OpenIE
from .openie_openai import ChunkInfo
from ..utils.logging_utils import get_logger
from ..prompts import PromptTemplateManager
from ..llm.vllm_offline import VLLMOffline

logger = get_logger(__name__)


class VLLMOfflineOpenIE(OpenIE):
    def __init__(self, global_config):

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = VLLMOffline(global_config)
        self.global_config = global_config

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

        #构造 NER 输入 Prompt
        ner_input_messages = [self.prompt_template_manager.render(name=self.global_config.prompt, passage=p) for p in chunk_passages.values()] #为每个 passage 生成 NER prompt
        for j, prompt in enumerate(ner_input_messages[:]):
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
