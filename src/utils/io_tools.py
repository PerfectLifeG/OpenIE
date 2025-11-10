# -*- coding: utf-8 -*-
"""
文件功能：I/O 工具。支持 YAML 加载、递归找文件、覆盖写出 JSONL 等。
"""
from __future__ import annotations
import json, os
from pathlib import Path
import yaml

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
    records: Iterable[Any],
    overwrite: bool = True,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    """
    将 records 写为一个 JSON 数组到 path。
    """
    print(f"[INFO] 写入文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if overwrite:
            path.unlink()  # 覆盖旧文件
        else:
            raise FileExistsError(f"{path} 已存在且 overwrite=False，已停止写入。")

    # 收集为列表（兼容生成器/迭代器）
    data = list(records)

    # 写入为标准 JSON（数组）
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,  # 友好显示中文
            indent=indent,       # 美化缩进
            sort_keys=sort_keys, # 可选：键排序
            default=str          # 兜底序列化非常规对象
        )
        f.write("\n")  # 结尾换行，友好对齐工具习惯
