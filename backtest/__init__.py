"""
回测模块（Layer 5）。

提供事件驱动的回测引擎、性能指标计算和结果数据模型。
"""

from backtest.engine import BacktestEngine, round_to_lot
from backtest.metrics import PerformanceMetrics
from backtest.result import BacktestResult, Trade

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "PerformanceMetrics",
    "Trade",
    "round_to_lot",
]
