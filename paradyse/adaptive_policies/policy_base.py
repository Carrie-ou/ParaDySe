"""
Author: Peng Liang
Date: 2025-2-15 22:06:04
LastEditTime: 2025-2-15 22:05:59
LastEditors: Peng Liang

Description: Policy基类
"""
from abc import ABC, abstractmethod


class PolicyBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select_method(self, *args, **kwargs) -> str:
        pass
