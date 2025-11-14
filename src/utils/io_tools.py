# -*- coding: utf-8 -*-
"""
文件功能：I/O 工具。支持 YAML 加载、递归找文件、覆盖写出 JSONL 等。
"""
from __future__ import annotations
import json, os
from pathlib import Path
import yaml
import hashlib
from typing import Any, List, Dict, Optional
import collections.abc as cabc

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def iter_find_file(root: Path, filename: str):
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            return Path(dirpath) / filename
    raise FileNotFoundError(f"未在 {root} 下递归找到 {filename}")

def write_jsonl_overwrite(path: Path, records, overwrite=True):
    print(f"[INFO] 写入文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and overwrite:
        path.unlink()  # 覆盖旧文件
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -*- coding: utf-8 -*-
"""
将 records 写入“标准 JSON 文件”（JSON 数组），并按需覆盖已有文件。
- 默认覆盖：若文件已存在会先删除再写入（与您的偏好一致）。
- 支持可迭代对象：会将生成器/迭代器收集为列表后再序列化。
- 中文友好：ensure_ascii=False；默认美化缩进。
"""


from typing import Any, Iterable

def write_json_overwrite(
    path: Path,
    records: Any,                     # 允许传任意对象（dict/list/tuple/可迭代/标量）
    overwrite: bool = True,
    indent: int = 2,
    sort_keys: bool = False,
    default: Optional[Any] = str,     # 保持你原来的默认
    add_trailing_newline: bool = True,
    fsync: bool = False,
) -> None:
    """
    将 `records` 写为 JSON 到 path。

    行为：
      - dict/list/tuple            -> 原样写入（对象 JSON）
      - 其它可迭代(非str/bytes)     -> 收集为 list 再写（与旧版一致：写 JSON 数组）
      - 其余标量/对象               -> 原样写入（让 json.dump 去处理）
    """
    print(f"[INFO] 写入文件（原子方式）：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} 已存在且 overwrite=False，已停止写入。")

    # === 关键兼容逻辑 ===
    if isinstance(records, (dict, list, tuple)):
        data = records
    elif isinstance(records, cabc.Iterable) and not isinstance(records, (str, bytes, bytearray)):
        # 兼容旧逻辑：把任意可迭代（非字符串/字节）当成“记录流”，收集为数组
        data = list(records)
    else:
        # 标量或其它对象：直接让 json.dump 处理
        data = records

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, sort_keys=sort_keys, default=default)
        if add_trailing_newline:
            f.write("\n")
        if fsync:
            f.flush()
            os.fsync(f.fileno())
    # 同一目录内原子替换
    tmp.replace(path)

def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """计算文件的 SHA256 摘要，用于校验数据一致性。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    兼容加载：
      - JSON 列表：  [ {...}, {...}, ... ]
      - JSONL 行：   {...}\n{...}\n
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    first = text[0]
    if first == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON 顶层应为 list。")
        return data
    # 否则按 JSONL 逐行解析
    records: List[Dict[str, Any]] = []
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"解析 JSONL 第 {i+1} 行失败: {e}")
        if not isinstance(obj, dict):
            raise ValueError(f"JSONL 第 {i+1} 行不是对象。")
        records.append(obj)
    return records