# -*- coding: utf-8 -*-
"""
文件功能：统一设定随机种子，便于复现。
"""

import random, numpy as np

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
