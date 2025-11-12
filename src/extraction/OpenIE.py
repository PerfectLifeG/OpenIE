import json
import os
from dataclasses import asdict

from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE

from .utils.misc_utils import *

logger = logging.getLogger(__name__)


class OpenIE:

    def __init__(self, global_config=None, ):

        self.global_config = global_config

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"HippoRAG init with config:\n  {_print_config}\n")

        self.openie = VLLMOfflineOpenIE(self.global_config)

        self.openie_results_path = os.path.join(self.global_config.save_dir,
                                                f'{self.global_config.prompt}_{self.global_config.llm_name.replace("/", "_")}.json')

    def save_ner_outputs(self, new_ner_results, openie_results_path):
        """
        new_ner_results: dict or list
            - dict: {chunk_id: raw_text_dict}
            - list: [raw_text_dict1, raw_text_dict2, ...]
        openie_results_path: str, 保存路径
        """
        if openie_results_path == "/home/penglin.ge/code/OpenIE/outputs/test2/ner_3__home_penglin.ge_code_DoRA_commonsense_reasoning_model3.json":
            pass



        import os, json

        dir_path = os.path.dirname(openie_results_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # 如果是 dict，就取 values 转成 list
        if isinstance(new_ner_results, dict):
            results_list = list(new_ner_results.values())
        elif isinstance(new_ner_results, list):
            results_list = new_ner_results
        else:
            raise TypeError(f"Unsupported type: {type(new_ner_results)}")

        # 写成 JSONL，每行一个 JSON 对象
        with open(openie_results_path, "w", encoding="utf-8") as f:
            f.write("[\n")  # 开头
            for i, item in enumerate(results_list):
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except:
                        item = {}  # 解析失败就用空 dict
                json.dump(item, f, ensure_ascii=False, indent=2)
                if i != len(results_list) - 1:
                    f.write(",\n")  # 每个对象后面加逗号（除了最后一个）
                else:
                    f.write("\n")
            f.write("]")  # 结尾

        print(f"NER 输出已保存到: {openie_results_path}")

    # def save_ner_outputs(self, new_ner_results_dict, openie_results_path):
    #     """
    #     new_ner_results_dict: dict, {chunk_id: raw_text}
    #     openie_results_path: str, 保存路径，可以是文件夹或文件
    #     """
    #     # 如果路径是文件夹，创建文件夹
    #     dir_path = os.path.dirname(openie_results_path)
    #     if dir_path and not os.path.exists(dir_path):
    #         os.makedirs(dir_path, exist_ok=True)
    #
    #     with open(openie_results_path, "w", encoding="utf-8") as f:
    #         json.dump(new_ner_results_dict, f, ensure_ascii=False, indent=2)
    #
    #     print(f"NER 输出已保存到: {openie_results_path}")

    def pre_openie(self, docs: Dict, temp=0.0, tp=0.0):
        logger.info(f"Performing OpenIE Offline")

        # chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        # all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        # new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        # if len(chunk_keys_to_process) > 0:
        new_ner_results_list, new_triple_results_list = self.openie.batch_openie(docs, temp, tp)  # 批处理
        # self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        # 融入文档 ID
        doc_ids = list(docs.keys())
        new_ner_results_with_id = []

        for doc_id, ner_str in zip(doc_ids, new_ner_results_list):
            try:
                ner_obj = json.loads(ner_str)  # 解析字符串
            except json.JSONDecodeError:
                ner_obj = {"entities": []}
            # 将 ID 加入对象
            ner_with_id = {"id": doc_id}
            ner_with_id.update(ner_obj)

            new_ner_results_with_id.append(ner_with_id)

        # if self.global_config.save_openie:
        # self.save_openie_results(new_ner_results_dict)
        self.save_ner_outputs(new_ner_results_with_id, self.openie_results_path)

        # assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.') #终止程序运行
