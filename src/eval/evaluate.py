import json
from collections import Counter
from typing import Dict, Tuple, Any, List, Optional
from sentence_transformers import SentenceTransformer, util
import os

_SBER_MODEL = None

# 加载句向量模型
def _get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
               local_dir: str = r"D:\hf_models\all-MiniLM-L6-v2",
               device: str = "cpu"):
    """加载句向量模型"""
    global _SBER_MODEL
    if _SBER_MODEL is not None:
        return _SBER_MODEL

    load_target = None
    if local_dir and os.path.isdir(local_dir):
        load_target = local_dir
    else:
        load_target = model_name  # 本地不可用则回退远程名

    _SBER_MODEL = SentenceTransformer(load_target, device=device)
    _SBER_MODEL.eval()  # 设置为推理模式
    return _SBER_MODEL


def _cos_sim(a_emb, b_emb) -> float:
    """计算余弦相似度"""
    return float(util.cos_sim(a_emb, b_emb))


def _load_jsonl_or_json(path: str):
    """加载 JSON 或 JSONL 文件"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except json.decoder.JSONDecodeError:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def _to_keys(sample: Dict[str, Any], strict: bool = True):
    """生成实体键，严格模式为 (subject[0], subject[1], relationship, object[0], object[1])，宽松模式为 (subject[0], relationship, object[0])"""
    keys = set()
    for e in sample.get("output", []):
        subject = e.get("subject", ["", "", ""])
        object_ = e.get("object", ["", "", ""])
        relationship = e.get("relationship", "")

        if strict:
            # 严格模式需要完整的匹配：头实体名、粗粒度、关系、尾实体名、尾实体粗粒度
            keys.add((subject[0], subject[1], relationship, object_[0], object_[1]))
        else:
            # 宽松模式：比对头实体名、关系、尾实体名
            keys.add((subject[0], relationship, object_[0]))
    return keys


def _to_map_name_ct_2_ft(sample: Dict[str, Any]) -> Dict[Tuple[str, str], str]:
    """将实体转为 (name, coarse_type) -> fine_type 的映射"""
    m: Dict[Tuple[str, str], str] = {}
    for e in sample.get("output", []):
        subject = e.get("subject", ["", "", ""])
        object_ = e.get("object", ["", "", ""])
        m[(subject[0], subject[1])] = subject[2]  # subject -> fine_type
        m[(object_[0], object_[1])] = object_[2]  # object -> fine_type
    return m


def evaluate_ner(dev_gold_path: str, pred_path: str,
                 strict: bool = True, by_type: bool = False, *,
                 strict_semantic: bool = False, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 threshold: float = 0.80):
    """
    参数
    ----
    strict: True=严格评测（name+coarse_type+relationship）；False=宽松（name+relationship+object 比对）
    by_type: 是否输出按 coarse_type 的统计（仅 strict=True 时可用）
    strict_semantic: 严格评测时是否用 fine_type 语义相似度（默认 False）
    model_name: 句向量模型名（仅 strict_semantic=True 时会被使用）
    threshold: 语义相似阈值（余弦），默认 0.80
    """
    gold = _load_jsonl_or_json(dev_gold_path)  # 标准答案
    pred = _load_jsonl_or_json(pred_path)      # 预测结果

    tp = fp = fn = 0
    type_counter = Counter()

    if not strict:
        # 宽松模式：比对头实体名、关系、尾实体名
        for g, p in zip(gold, pred):
            gset = _to_keys({"output": g.get("output", [])}, strict=False)
            pset = _to_keys({"output": p.get("output", [])}, strict=False)

            tp_i = len(gset & pset)
            fp_i = len(pset - gset)
            fn_i = len(gset - pset)

            tp += tp_i; fp += fp_i; fn += fn_i

        # 宽松模式不支持 by_type（保持与原行为一致）
        by_type = False

    else:
        # 严格模式：检查 subject[0]、subject[1]、relationship、object[0]、object[1]
        if not strict_semantic:
            for g, p in zip(gold, pred):
                gset = _to_keys({"output": g.get("output", [])}, strict=True)
                pset = _to_keys({"output": p.get("output", [])}, strict=True)

                tp_i = len(gset & pset)
                fp_i = len(pset - gset)
                fn_i = len(gset - pset)

                tp += tp_i; fp += fp_i; fn += fn_i

                if by_type:
                    for _, ct in (gset & pset):
                        type_counter[(ct, "tp")] += 1
                    for _, ct in (pset - gset):
                        type_counter[(ct, "fp")] += 1
                    for _, ct in (gset - pset):
                        type_counter[(ct, "fn")] += 1

        else:
            # 超严格模式：比较 fine_type 语义相似度
            model = _get_model(model_name)

            for g, p in zip(gold, pred):
                gmap = _to_map_name_ct_2_ft({"output": g.get("output", [])})
                pmap = _to_map_name_ct_2_ft({"output": p.get("output", [])})

                gkeys = set(gmap.keys())
                pkeys = set(pmap.keys())

                # 1. 完全不匹配的三元组：计入 FP 和 FN
                only_pred = pkeys - gkeys
                only_gold = gkeys - pkeys
                fp += len(only_pred)
                fn += len(only_gold)

                if by_type:
                    for _, ct in only_pred:
                        type_counter[(ct, "fp")] += 1
                    for _, ct in only_gold:
                        type_counter[(ct, "fn")] += 1

                # 2. 对于匹配的三元组，比较 fine_type 的语义相似度
                intersect = gkeys & pkeys
                if intersect:
                    # 准备文本列表：pred_fines, gold_fines
                    pred_fines: List[str] = [pmap[k] for k in intersect]
                    gold_fines: List[str] = [gmap[k] for k in intersect]

                    # 空字符串处理：若某侧为空，直接认为相似度为 0
                    to_encode_pred = [t if t is not None and t != "" else "<EMPTY>" for t in pred_fines]
                    to_encode_gold = [t if t is not None and t != "" else "<EMPTY>" for t in gold_fines]

                    pred_emb = model.encode(to_encode_pred, normalize_embeddings=True)
                    gold_emb = model.encode(to_encode_gold, normalize_embeddings=True)

                    # 逐一比较语义相似度
                    for (k, pe, ge, pf, gf) in zip(intersect, pred_emb, gold_emb, pred_fines, gold_fines):
                        # 完全字符串相同：计为 TP
                        if pf == gf:
                            tp += 1
                            if by_type:
                                type_counter[(k[1], "tp")] += 1
                            continue

                        sim = _cos_sim(pe, ge)
                        if sim >= threshold:
                            tp += 1
                            if by_type:
                                type_counter[(k[1], "tp")] += 1
                        else:
                            # 键匹配但 fine_type 语义未达阈值：记为 FP + FN
                            fp += 1
                            fn += 1
                            if by_type:
                                type_counter[(k[1], "fp")] += 1
                                type_counter[(k[1], "fn")] += 1

    # 计算 Precision, Recall, F1
    def prf(tp_: int, fp_: int, fn_: int):
        p = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        r = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f}

    # 汇总报告
    report = {}
    report["overall"] = prf(tp, fp, fn)
    if by_type and strict:
        per_type = {}
        coarse_types = {k[0] for k in type_counter.keys()}  # k=(ct, "tp"/"fp"/"fn")
        for ct in sorted(coarse_types):
            tp_t = type_counter[(ct, "tp")]
            fp_t = type_counter[(ct, "fp")]
            fn_t = type_counter[(ct, "fn")]
            per_type[ct] = prf(tp_t, fp_t, fn_t)
        report["by_coarse_type"] = per_type

    # 附加：记录语义评测配置，便于复现
    report["config"] = {
        "strict": strict,
        "strict_semantic": strict_semantic,
        "model_name": model_name if (strict and strict_semantic) else None,
        "threshold": threshold if (strict and strict_semantic) else None
    }
    return report
