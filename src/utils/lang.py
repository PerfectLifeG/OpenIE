# -*- coding: utf-8 -*-
"""
文件功能：简单语言判定与中英提示选择。
"""

import re

def is_chinese(text: str) -> bool:
    """
    判断字符串是否为中文
    """
    return bool(re.search(r"[\u4e00-\u9fff]", text))
