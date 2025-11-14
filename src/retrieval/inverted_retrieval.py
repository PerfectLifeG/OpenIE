# -*- coding: utf-8 -*-
"""
文件名：src/retrieval/inverted_retrieval.py

功能拆分为三个阶段：
1) build_indexes()    -> 从数据构建索引并写入 JSON（离线跑一次即可）
2) load_indexes()     -> 从磁盘加载数据集和索引到内存
3) retrieve_*()       -> 只用内存中的数据和索引进行检索
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set

from src.utils.io_tools import load_json_or_jsonl, file_sha256, ensure_dir, write_json_overwrite

COARSE_INDEX_FILENAME = "coarse_index.json"
REL_INDEX_FILENAME = "relationship_index.json"


class InvertedRetrieval:
    """最小可用的倒排索引构建与 few-shots 检索类封装。"""

    def __init__(self, data_path: Path, indexdir: Path) -> None:
        self.data_path = Path(data_path)
        self.indexdir = Path(indexdir)

        # 缓存（用于检索阶段）
        self._dataset: Optional[List[Dict[str, Any]]] = None
        self._coarse_index: Optional[Dict[str, List[int]]] = None
        self._rel_index: Optional[Dict[str, List[int]]] = None

    # ========== 阶段1：构建并写出索引（只在需要时执行一次） ==========

    def build_indexes(
            self,
            dataset: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Path, Path]:
        """
        从数据构建并写出两类索引文件（互相独立）：
            indexdir/coarse_index.json
            indexdir/relationship_index.json
        """
        ensure_dir(self.indexdir)
        if dataset is None:
            dataset = load_json_or_jsonl(self.data_path)

        meta = {
            "data_path": str(self.data_path),
            "sha256": file_sha256(self.data_path),
            "num_examples": len(dataset),
            "created_at": datetime.now().isoformat(timespec="seconds") + "Z",
        }

        coarse = self._build_coarse_index(dataset)
        rel = self._build_relationship_index(dataset)

        coarse_obj = {"meta": meta, "index": coarse}
        rel_obj = {"meta": meta, "index": rel}

        coarse_path = self.indexdir / COARSE_INDEX_FILENAME
        rel_path = self.indexdir / REL_INDEX_FILENAME

        write_json_overwrite(records=coarse_obj, path=coarse_path)
        write_json_overwrite(records=rel_obj, path=rel_path)

        return coarse_path, rel_path

    # ========== 阶段2：加载数据集和索引到内存 ==========

    def load_indexes(self) -> None:
        """
        从磁盘加载数据集和索引到内存。
        仅负责“加载”，不负责“构建”；如果索引不存在会报错。
        """
        coarse_idx_path = self.indexdir / COARSE_INDEX_FILENAME
        rel_idx_path = self.indexdir / REL_INDEX_FILENAME

        if not coarse_idx_path.exists() or not rel_idx_path.exists():
            raise FileNotFoundError(
                f"索引文件不存在，请先在离线脚本中调用 build_indexes() 构建：\n"
                f"  {coarse_idx_path}\n  {rel_idx_path}"
            )

        # 一次性把数据和索引都读到内存
        self._dataset = load_json_or_jsonl(self.data_path)

        coarse_obj = json.loads(coarse_idx_path.read_text(encoding="utf-8"))
        rel_obj = json.loads(rel_idx_path.read_text(encoding="utf-8"))

        self._coarse_index = coarse_obj.get("index", {})
        self._rel_index = rel_obj.get("index", {})

    def _ensure_loaded(self) -> None:
        """内部使用：确保检索前已经加载了数据和索引。"""
        if self._dataset is None or self._coarse_index is None or self._rel_index is None:
            self.load_indexes()

    # ========== 阶段3：检索（只依赖内存，不再重读文件） ==========

    def retrieve_by_coarse_type(
            self,
            coarse_type: str,
            k: int,
            seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        按 coarse_type 检索，返回随机 k 条样例（不足 k 则尽量返回全部；不存在则空列表）。
        """
        self._ensure_loaded()
        assert self._dataset is not None and self._coarse_index is not None

        cand = self._coarse_index.get(coarse_type, [])
        if not cand:
            return []
        picked = self._random_pick(cand, k, seed=seed)
        return [self._dataset[i] for i in picked]

    def retrieve_by_schema(
            self,
            schema: str,
            k: int,
            seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        按 schema（relationship）检索，返回随机 k 条样例（不足 k 则尽量全部；不存在则空列表）。
        """
        self._ensure_loaded()
        assert self._dataset is not None and self._rel_index is not None

        cand = self._rel_index.get(schema, [])
        if not cand:
            return []
        picked = self._random_pick(cand, k, seed=seed)
        return [self._dataset[i] for i in picked]

    # ========== 工具：将检索结果转成二维 [input, output] ==========

    @staticmethod
    def to_io_pairs(examples: List[Dict[str, Any]]) -> List[List[Any]]:
        """
        将检索得到的样例结构（list[dict]）转换为二维列表：
        第1维：第几个样例；
        第2维：0 是输入 input，1 是输出 output。
        input 默认包含：sentence / schema / coarse_types；
        output 为原样的 output 列表。

        参数
        ----
        examples : List[Dict[str, Any]]
            形如 [{'source':..., 'sentence':..., 'schema':..., 'coarse_types':..., 'output':[...]}, ...]
            的样例列表（即检索返回的结果）。

        返回
        ----
        List[List[Any]]
            [[input, output], [input, output], ...]
        """
        io_matrix: List[List[Any]] = []
        for ex in examples:
            input_part = {
                "sentence": ex.get("sentence", ""),
                "schema": ex.get("schema", []),
                "coarse_types": ex.get("coarse_types", []),
            }
            output_part = ex.get("output", [])
            io_matrix.append([input_part, output_part])
        return io_matrix

    # ========== 内部：索引构建细节（与原脚本一致） ==========

    @staticmethod
    def _build_coarse_index(dataset: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        构建 coarse_type 倒排索引：
          key: 实体粗粒度字符串（来自 subject[1] 或 object[1]）
          val: 命中的样例行号列表（不重复）
        """
        index: Dict[str, Set[int]] = {}
        for i, ex in enumerate(dataset):
            outputs = ex.get("output", [])
            # print(outputs)
            if not isinstance(outputs, list):
                continue
            hit_keys: Set[str] = set()
            for triple in outputs:
                subj = triple.get("subject", [])
                obj = triple.get("object", [])
                if isinstance(subj, list) and len(subj) >= 2 and isinstance(subj[1], str):
                    hit_keys.add(subj[1])
                if isinstance(obj, list) and len(obj) >= 2 and isinstance(obj[1], str):
                    hit_keys.add(obj[1])
            for key in hit_keys:
                index.setdefault(key, set()).add(i)
        return {k: sorted(list(v)) for k, v in index.items()}

    @staticmethod
    def _build_relationship_index(dataset: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        构建 relationship（= schema）倒排索引：
          key: relationship 字符串（比如“所属专辑”）
          val: 命中的样例行号列表（不重复）
        """
        index: Dict[str, Set[int]] = {}
        for i, ex in enumerate(dataset):
            outputs = ex.get("output", [])
            if not isinstance(outputs, list):
                continue
            hit_keys: Set[str] = set()
            for triple in outputs:
                rel = triple.get("relationship")
                if isinstance(rel, str) and rel:
                    hit_keys.add(rel)
            for key in hit_keys:
                index.setdefault(key, set()).add(i)
        return {k: sorted(list(v)) for k, v in index.items()}

    # ========== 内部：随机抽取（与原脚本随机行为一致） ==========

    @staticmethod
    def _random_pick(indices: List[int], k: int, seed: Optional[int] = None) -> List[int]:
        """从 indices 中随机选 k 个（不放回）。不足 k 时返回打乱后的全部。"""
        if seed is not None:
            random.seed(seed)
        if k <= 0:
            return []
        if len(indices) <= k:
            tmp = indices[:]
            random.shuffle(tmp)
            return tmp
        return random.sample(indices, k)


if __name__ == '__main__':
    train_path = '/home/penglin.ge/code/OpenIE/data/train2.json'
    out_dir = '/home/penglin.ge/code/OpenIE/outputs'
    retriever = InvertedRetrieval(data_path=train_path, indexdir=out_dir)
    retriever.build_indexes()
    shots1 = retriever.retrieve_by_coarse_type(coarse_type="人", k=4, seed=42)
    paired_shots = retriever.to_io_pairs(shots1)
    # shots2 = retriever.retrieve_by_schema(schema="所属专辑", k=6, seed=123)

    print(shots1)
    print(paired_shots)
    # print(shots2)