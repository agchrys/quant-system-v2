"""
策略插件系统 — 基类与注册中心

所有量化策略以插件形式实现，继承 StrategyBase 并注册到 StrategyRegistry。
使用方式：
    strategy = StrategyRegistry.get("v8_15")
    result = strategy.run_pipeline(config)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
import importlib
import pkgutil
import os
from loguru import logger


class StrategyBase(ABC):
    """策略基类 — 所有策略插件必须继承此类"""

    # 策略元信息（子类必须覆盖）
    name: str = ""
    version: str = ""
    description: str = ""
    author: str = ""

    @abstractmethod
    def run_pipeline(self, config: dict, **kwargs) -> dict:
        """
        执行完整策略流水线。

        Args:
            config: 全局配置字典
            **kwargs: 可选参数（skip_data, strategy_config等）

        Returns:
            回测结果字典
        """
        pass

    @abstractmethod
    def config_section_name(self) -> str:
        """返回该策略在 config.yaml 中的配置段名称"""
        pass

    def get_config(self, config: dict) -> dict:
        """从全局配置中提取本策略的配置段"""
        section = self.config_section_name()
        strategy_config = config.get("strategies", {}).get(section, {})
        return strategy_config

    def __str__(self) -> str:
        return f"[{self.name} v{self.version}] {self.description}"


class StrategyRegistry:
    """策略注册中心 — 自动发现并管理所有策略插件"""

    _strategies: Dict[str, Type[StrategyBase]] = {}

    @classmethod
    def register(cls, strategy_cls: Type[StrategyBase]) -> Type[StrategyBase]:
        """注册一个策略类（可作为装饰器使用）"""
        instance = strategy_cls()
        name = instance.name
        if not name:
            name = strategy_cls.__name__.lower()
        cls._strategies[name] = strategy_cls
        logger.debug(f"策略已注册: {instance}")
        return strategy_cls

    @classmethod
    def get(cls, name: str) -> Optional[StrategyBase]:
        """按名称获取策略实例"""
        strategy_cls = cls._strategies.get(name)
        if strategy_cls is None:
            return None
        return strategy_cls()

    @classmethod
    def list_strategies(cls) -> List[Dict[str, str]]:
        """列出所有已注册的策略"""
        result = []
        for name, strategy_cls in cls._strategies.items():
            inst = strategy_cls()
            result.append({
                "name": name,
                "version": inst.version,
                "description": inst.description,
            })
        return result

    @classmethod
    def discover(cls, package_path: str = None) -> int:
        """
        自动发现并注册 strategies/ 目录下的所有策略插件。
        扫描所有 .py 文件，导入它们（注册装饰器会自动执行）。
        """
        if package_path is None:
            package_path = os.path.dirname(os.path.abspath(__file__))

        count = 0
        for importer, modname, ispkg in pkgutil.iter_modules([package_path]):
            if modname.startswith('_') or modname == 'base':
                continue
            try:
                importlib.import_module(f"strategies.{modname}")
                count += 1
            except Exception as e:
                logger.warning(f"策略加载失败: {modname} — {e}")

        logger.info(f"策略自动发现: {count} 个模块, {len(cls._strategies)} 个策略")
        return len(cls._strategies)


# 自动发现（模块导入时执行）
StrategyRegistry.discover()
