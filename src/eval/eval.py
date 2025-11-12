# -*- coding: utf-8 -*-
"""
在 dev 上计算 NER 指标：
模式1：严格模式 - (name, coarse_type, fine_type) 三个都对
模式2：中等模式 - (name, coarse_type) 两个对
模式3：宽松模式 - 只对 name
"""

import json

def _load_jsonl_or_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except json.decoder.JSONDecodeError:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def _to_keys(sample, mode="strict"):
    """
    根据模式生成不同的key
    mode: "strict" - (name, coarse_type, fine_type)
          "medium" - (name, coarse_type)
          "loose"  - name
    """
    keys = set()
    for e in sample.get("entities", []):
        if isinstance(e, dict):  # dict格式的NER输出
            subj = e.get("subject", ["", "", ""])
            obj = e.get("object", ["", "", ""])
            rel = e.get("relationship", "")

            subj_name = subj[0] if subj else ""
            obj_name = obj[0] if obj else ""

            keys.add(subj_name)
            keys.add(obj_name)
            # print(subj_name)
            # print(obj_name)
            # name = e.get("name", "")
            # ct = e.get("coarse_type", "")
            # ft = e.get("fine_type", "")
        else:  # 简单字符串格式的NER输出
            name, ct, ft = e, "", ""
            keys.add(name)


        # if mode == "strict":
        #     keys.add((name, ct, ft))  # 三个都对
        # elif mode == "medium":
        #     keys.add((name, ct))  # 两个对
        # elif mode == "loose":
        #     keys.add(name)  # 只对name
    return keys


def evaluate_ner(dev_gold_path, pred_path, mode="strict", error_output_path=None):
    """
    评测NER性能
    mode: "strict" - (name, coarse_type, fine_type) 三个都对
          "medium" - (name, coarse_type) 两个对
          "loose"  - 只对 name
    """
    gold = _load_jsonl_or_json(dev_gold_path)  # gold: json list
    pred = _load_jsonl_or_json(pred_path)  # pred: jsonl list

    tp = fp = fn = 0

    # 新增：错误分析数据
    error_analysis = []

    for idx, (g, p) in enumerate(zip(gold, pred)):
        g_entities = g.get("output", [])
        p_entities = p.get("entities", [])

        gset = _to_keys({"entities": g_entities}, mode)
        pset = _to_keys({"entities": p_entities}, mode)

        tp_i = len(gset & pset)
        fp_i = len(pset - gset)
        fn_i = len(gset - pset)

        tp += tp_i
        fp += fp_i
        fn += fn_i

        # 新增：记录错误样本
        if error_output_path and (fp_i > 0 or fn_i > 0):
            error_sample = {
                "index": idx,
                "sentence": g.get("sentence", ""),
                "coarse_types": g.get("coarse_types", []),
                "gold_entities": g_entities,
                "pred_entities": p_entities
            }
            error_analysis.append(error_sample)


    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f}

    report = {"overall": prf(tp, fp, fn), "mode": mode}


    # 新增：保存错误分析结果
    if error_output_path and error_analysis:
        with open(error_output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "error_statistics": {
                    "total_samples": len(gold),
                    "error_samples": len(error_analysis),
                    "error_rate": len(error_analysis) / len(gold),
                    "mode": mode
                },
                "error_details": error_analysis
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ 错误分析已保存到: {error_output_path}")
        print(f"📊 错误样本统计: {len(error_analysis)}/{len(gold)} 个样本存在错误")

    return report



def run():
    data_path = '/home/penglin.ge/code/HippoRAG-main/reproduce/dataset/test2.json'
    save_path = '/home/penglin.ge/code/OpenIE/outputs/test2/ner_1__home_penglin.ge_code_DoRA_commonsense_reasoning_model1.json'
    error_output_path = '/home/penglin.ge/code/OpenIE/outputs/dev2/error.json'

    # 三种模式分别评测
    modes = ["strict", "medium", "loose"]
    mode_names = {
        "strict": "严格模式 (name, coarse_type, fine_type)",
        "medium": "中等模式 (name, coarse_type)",
        "loose": "宽松模式 (name)"
    }

    all_reports = {}

    for mode in modes:
        print(f"\n=== {mode_names[mode]} ===")
        report = evaluate_ner(
            dev_gold_path=data_path,
            pred_path=save_path,
            mode=mode,
            error_output_path=error_output_path.replace('.json', f'_{mode}.json')
        )
        all_reports[mode] = report
        print(json.dumps(report, ensure_ascii=False, indent=2))

    # 输出对比结果
    print("\n" + "=" * 50)
    print("三种模式对比结果:")
    print("=" * 50)
    for mode in modes:
        overall = all_reports[mode]["overall"]
        print(
            f"{mode_names[mode]:<40} | F1: {overall['f1']:.4f} | Precision: {overall['precision']:.4f} | Recall: {overall['recall']:.4f}")


if __name__ == "__main__":
    run()