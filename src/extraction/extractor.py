import os

import numpy as np
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json

import torch
torch._dynamo.disable()

from .OpenIE import OpenIE
from .utils.misc_utils import string_to_bool
from .utils.config_utils import BaseConfig

import argparse

from ..eval.eval import evaluate_ner

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from tqdm import tqdm


def extractor(dataset:str,mode,temp,tp):
    if mode == 1:
        llm_name = '/home/penglin.ge/code/DoRA/commonsense_reasoning/model1'
    elif mode == 2:
        llm_name = '/home/penglin.ge/code/DoRA/commonsense_reasoning/Qwen_model2'
    else:
        llm_name = '/home/penglin.ge/code/DoRA/commonsense_reasoning/Qwen_model3'

    parser = argparse.ArgumentParser(description="LLM OpenIE")
    parser.add_argument('--dataset', type=str, default=dataset, help='Dataset name')
    parser.add_argument('--llm_name', type=str, default=llm_name, help='LLM name')
    parser.add_argument('--save_dir', type=str, default='outputs', help='Save directory')
    parser.add_argument('--prompt', type=str, default=f'ner_{mode}')
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    llm_name = args.llm_name
    if save_dir == 'outputs':
        save_dir = save_dir + '/' + dataset_name
    else:
        save_dir = save_dir + '_' + dataset_name

    corpus_path = f"data/{dataset_name}.json"
    # corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json" #语料库，title+text
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    # 读取实体
    if mode == 2:
        entities_path = f"/home/penglin.ge/code/OpenIE/outputs/{dataset}/ner_1__home_penglin.ge_code_DoRA_commonsense_reasoning_model1.json"
    elif mode == 3:
        entities_path = f"/home/penglin.ge/code/OpenIE/outputs/{dataset}/ner_2__home_penglin.ge_code_DoRA_commonsense_reasoning_Qwen_model2.json"

    if mode == 2:
        with open(entities_path, "r") as f:
            entities_list = json.load(f)
        id2entities = {item["id"]: item.get("entities", []) for item in entities_list}
    if mode == 3:
        with open(entities_path, "r") as f:
            entities_list = json.load(f)
        id2triples = {item["id"]: item.get("triples", []) for item in entities_list}

    docs = {}
    for i,item in enumerate(corpus):
        if args.prompt == 'ner_2':
            item_no_id = {k: v for k, v in item.items() if k == "sentence" or k == "coarse_types" or k == "schema"}  # 去掉 id  or k == "coarse_types"
        elif args.prompt == 'ner_1':
            item_no_id = {k: v for k, v in item.items() if k == "sentence" or k == "coarse_types" or k == "schema"}  # 去掉 id
        else:
            item_no_id = {k: v for k, v in item.items() if k == "sentence"}  # 去掉 id

        item_id = item.get("id", i)

        if args.prompt == 'ner_2':
            item_no_id["entities"] = id2entities.get(item_id, [])
        if args.prompt == 'ner_3':
            item_no_id["triples"] = id2triples.get(item_id, [])

        json_str = json.dumps(item_no_id, ensure_ascii=False, indent=2)
        docs[item_id] = json_str  # 保留 id 对应的内容（不含 id）



    config = BaseConfig(
        save_dir=save_dir,
        llm_name=llm_name,
        dataset=dataset_name,
        prompt=args.prompt,
    )

    logging.basicConfig(level=logging.INFO)
    llm = OpenIE(global_config=config)

    # hipporag.index(docs) #openIE
    llm.pre_openie(docs, temp, tp)

    # search_best_params(hipporag, docs)



def search_best_params(mode, docs, result_log="coarse_param_search_results.csv"):
    mxf1 = 0
    ans = ()

    data_path = '/home/penglin.ge/code/HippoRAG-main/reproduce/dataset/dev1.json'
    save_path = '/home/penglin.ge/code/HippoRAG-main/outputs/dev1/openie_results_ner__home_penglin.ge_data_huggingface_model_Qwen2.5-7B-Instruct.json'

    if not os.path.exists(result_log):
        with open(result_log, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["temperature", "top_p", "f1", "precision", "recall"])

    for top_p in tqdm(np.arange(0.9, 1.01, 0.01), desc="Top-p", leave=False, unit="step"):
        for temp in tqdm(np.arange(0, 0.31, 0.01), desc="Temperature", unit="step"):
            print(f"Testing temperature={temp:.2f}, top_p={top_p:.2f}")

            # 运行主程序
            mode.pre_openie(docs,temp,top_p)
            # main('dev1', 1, temp, top_p)

            # 评估
            ret = evaluate_ner(data_path, save_path, 'medium')
            f1 = ret['overall']['f1']
            recall = ret['overall']['recall']
            pre = ret['overall']['precision']


            print(f"→ F1 = {f1:.4f}")

            # 追加写入结果
            with open(result_log, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([f"{temp:.2f}", f"{top_p:.2f}",  f"{f1:.4f}", f"{pre:.4f}", f"{recall:.4f}"])

            # 更新最优结果
            if f1 > mxf1:
                ans = (temp, top_p)
                mxf1 = f1
                print(f"⭐ New best: F1={f1:.4f} precision={pre:.4f} recall={recall:.4f} at (temperature={temp:.2f}, top_p={top_p:.2f})")

    print("\n✅ 最优参数：", ans, "F1 =", mxf1)


if __name__ == "__main__":
    extractor('error', 1, 0.17, 0.95)
    extractor('error', 2, 0.2, 0.9)
    extractor('error', 3, 0.5, 0.95)
