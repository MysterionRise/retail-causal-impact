# path: src/coupon_causal/__init__.py
"""
Coupon Causal Impact Analysis

A production-quality package for estimating causal effects of retail coupon campaigns.
"""

__version__ = "0.1.0"

from . import ate, balance, cate, data, features, policy, propensity, refute, utils, viz

__all__ = [
    "ate",
    "balance",
    "cate",
    "data",
    "features",
    "policy",
    "propensity",
    "refute",
    "utils",
    "viz",
]
