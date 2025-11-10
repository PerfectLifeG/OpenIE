# -*- coding: utf-8 -*-
"""
文件功能：加载 train/dev/test JSON 数据，统一为列表结构。
"""

import json
from pathlib import Path

def load_json_dataset(path: Path, max_examples=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_examples is not None:
        data = data[:max_examples]
    return data
