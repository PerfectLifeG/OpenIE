# -*- coding: utf-8 -*-
"""
文件功能：项目主入口。加载配置与数据，调用 LLM 抽取，做本体映射、JSON 校验与结果落盘，并可在 dev 上评测。
"""

import argparse
from pathlib import Path
from .utils.io_tools import load_yaml, ensure_dir, iter_find_file, write_json_overwrite
from .utils.seed import set_global_seed
from .data.dataset import load_json_dataset
# from .extraction.llm_extractor import NerExtractor
from .extraction.gptner_extractor import EntityExtractor
from .eval.evaluate import evaluate_ner
from .eval.self_verify import SelfVerifier

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    cfg = load_yaml(args.config)

    set_global_seed(cfg["project"]["seed"])

    out_dir = Path(cfg["paths"]["output_dir"])
    ensure_dir(out_dir)

    # 递归查找数据文件
    data_dir = Path(cfg["paths"]["data_dir"])
    dev_path = iter_find_file(data_dir, "dev2.json")
    test_path = iter_find_file(data_dir, "test2.json")

    if args.split == "dev":
        data_path = dev_path
        save_path = Path(cfg["paths"]["dev_pred_path"])
        verify_path = Path(cfg["paths"]["dev_self_verify_path"])
    else:
        data_path = test_path
        save_path = Path(cfg["paths"]["test_pred_path"])
        verify_path = Path(cfg["paths"]["test_self_verify_path"])
        print(f"[INFO] 检测到测试集文件：{data_path}")

    dataset = load_json_dataset(data_path, max_examples=cfg["runtime"]["max_examples"])

    # LLM 抽取
    extractor = EntityExtractor(cfg)

    # outputs = []
    result = extractor.extract_and_save_all(dataset, save_path)
    print("[OK] 抽取完成。路径：", result)
    
    # LLM 验证
    verifier = SelfVerifier(cfg)
    dataset = load_json_dataset(result, max_examples=cfg["runtime"]["max_examples"])
    verifier.verify_and_save_all(dataset, verify_path)
    print("[OK] 验证完成。路径：", verify_path)
    # for ex in dataset:
    #     result = extractor.extract(ex)
    #     outputs.append(result)
    
    # # 多线程
    # max_threads = 30
    # outputs = extractor.extract_batch(dataset, max_threads=max_threads)

    # write_json_overwrite(save_path, outputs, overwrite=cfg["project"]["overwrite_outputs"])

    # if args.split == "dev":
    #     report = evaluate_ner(dev_gold_path=str(data_path), pred_path=str(save_path), strict=cfg["evaluation"]["strict_span_match"])
    #     print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
