from typing import List

from .base import LLMConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

from transformers import PreTrainedTokenizer

def convert_text_chat_messages_to_strings(messages: List[TextChatMessage], tokenizer: PreTrainedTokenizer, add_assistant_header=True) -> List[str]:
    return tokenizer.apply_chat_template(conversation=messages, tokenize=False)

def convert_text_chat_messages_to_input_ids(messages: List[TextChatMessage], tokenizer: PreTrainedTokenizer, add_assistant_header=True) -> List[List[int]]:
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        chat_template=None,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
        tools=None,
        documents=None,
    )
    encoded = tokenizer(prompt, add_special_tokens=False)
    return encoded['input_ids']
from vllm import SamplingParams, LLM
from vllm.sampling_params import StructuredOutputsParams
class VLLMOffline:

    def _init_llm_config(self) -> None:
        self.llm_config = LLMConfig()

    def __init__(self, global_config, cache_dir=None, cache_filename=None, max_model_len=4096, **kwargs):
        model_name = kwargs.get('model_name', global_config.llm_name)
        if model_name is None:
            model_name = 'meta-llama/Llama-3.3-70B-Instruct'

        pipeline_parallel_size = 1
        #tensor_parallel_size = kwargs.get('num_gpus', torch.cuda.device_count()) #张量并行数
        tensor_parallel_size = 1

        # if '8B' in model_name:
        #     tensor_parallel_size = 1
        # if 'bnb' in model_name:
        #     kwargs['quantization'] = 'bitsandbytes'
        #     kwargs['load_format'] = 'bitsandbytes'
        #     tensor_parallel_size = 1
        #     pipeline_parallel_size = kwargs.get('num_gpus', torch.cuda.device_count())

        import os
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn' #启动多个 GPU 工作进程
        self.model_name = model_name

        # engine_args = EngineArgs(
        #     max_seq_len=max_model_len,
        #     # 如果需要，可以在这里加其他 EngineArgs 参数
        # )
        max_model_len = kwargs.get("max_model_len", 4096)
        # self.client = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, pipeline_parallel_size=pipeline_parallel_size, #使用vllm库
        #                   seed=kwargs.get('seed', 0), dtype='auto',
        #                   # max_seq_len_to_capture=max_model_len,
        #                   enable_prefix_caching=True,
        #                   enforce_eager=False, gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.6),
        #                   max_model_len=max_model_len, quantization=kwargs.get('quantization', None), load_format=kwargs.get('load_format', 'auto'), trust_remote_code=True)

        self.client = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            seed=kwargs.get("seed", 0),
            dtype="auto",
            enable_prefix_caching=True,
            enforce_eager=False,
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.5),  # 保守显存
            max_model_len=max_model_len,
            quantization=kwargs.get("quantization", None),
            load_format=kwargs.get("load_format", "auto"),
            trust_remote_code=True
        )

        self.tokenizer = self.client.get_tokenizer()
        if cache_filename is None:
            cache_filename = f'{model_name.replace("/", "_")}_cache.sqlite'
        if cache_dir is None:
            cache_dir = os.path.join(global_config.save_dir, "llm_cache")#outputs/musique
        self.cache_file_name = os.path.join(cache_dir, cache_filename)
    
    def infer(self, messages: List[TextChatMessage], max_tokens=2048):
        logger.info(f"Calling VLLM offline, # of messages {len(messages)}")
        messages_list = [messages]
        prompt_ids = convert_text_chat_messages_to_input_ids(messages_list, self.tokenizer)

        vllm_output = self.client.generate(prompt_token_ids=prompt_ids,  sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0))
        response = vllm_output[0].outputs[0].text
        prompt_tokens = len(vllm_output[0].prompt_token_ids)
        completion_tokens = len(vllm_output[0].outputs[0].token_ids )
        metadata = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
        return response, metadata

    def batch_infer(
        self,
        messages_list: List[List[TextChatMessage]],
        max_tokens: int = 2048,
        json_template: str = None,
        temp: float = 0.0,
        tp: float = 0.0
    ):
        """
        messages_list: List[List[TextChatMessage]]  每条列表表示一条对话历史
        返回: parsed_responses, metadata
        """

        # -----------------------------
        # 配置结构化输出 (JSON)
        # -----------------------------
        structured_outputs_params = None
        if json_template == 'ner' or json_template == 'ner_3':
            # NER 输出 schema
            ner_schema = {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "coarse_type": {"type": "string"},
                                "fine_type": {"type": "string"}
                            },
                            "required": ["name", "coarse_type", "fine_type"]
                        }
                    }
                },
                "required": ["entities"]
            }
        elif json_template == 'ner_1':
            ner_schema = {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["entities"]
            }
        elif json_template == 'ner_2':
            ner_schema = {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "coarse_type": {"type": "string"}
                            },
                            "required": ["name", "coarse_type"]
                        }
                    }
                },
                "required": ["entities"]
            }
        structured_outputs_params = StructuredOutputsParams(json=ner_schema)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            # temperature=float(temp),
            # top_p=float(tp),
            structured_outputs=structured_outputs_params
        )

        # -----------------------------
        # 将每条消息列表转换为 token
        # -----------------------------
        # 先把 TextChatMessage 列表转换为文本 prompt
        all_prompts = [
            self.tokenizer.apply_chat_template(
                conversation=messages,
                chat_template=None,
                tokenize=False,
                add_generation_prompt=True,
                continue_final_message=False,
                tools=None,
                documents=None,
            )
            for messages in messages_list
        ]

        # print(all_prompts)

        # 调用 vLLM generate
        vllm_output = self.client.generate(
            prompts=all_prompts,
            sampling_params=sampling_params
        )

        # -----------------------------
        # 解析输出
        # -----------------------------
        raw_responses = []
        all_prompt_tokens = []
        all_completion_tokens = []

        for completion in vllm_output:
            prompt_tokens_len = len(completion.prompt_token_ids)
            all_prompt_tokens.append(prompt_tokens_len)

            if completion.outputs:
                output_tokens_len = len(completion.outputs[0].token_ids)
                all_completion_tokens.append(output_tokens_len)
                text = completion.outputs[0].text  # **原封不动文本**
            else:
                output_tokens_len = 0
                all_completion_tokens.append(0)
                text = "没有输出"  # 没有输出就空字符串

            raw_responses.append(text)

        metadata = {
            "prompt_tokens": sum(all_prompt_tokens),
            "completion_tokens": sum(all_completion_tokens),
            "num_request": len(messages_list)
        }

        return raw_responses, metadata
