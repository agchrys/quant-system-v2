"""
回测结果数据模型。

定义回测引擎输出的结构化数据类，包括：
- Trade: 单笔交易记录
- BacktestResult: 完整的回测运行结果
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Trade:
    """单笔交易记录。

    Attributes:
        date: 交易日期 (YYYY-MM-DD)
        stock_code: 股票代码
        action: 交易方向，buy 或 sell
        price: 成交价格
        shares: 成交股数（整百股）
        commission: 佣金
        tax: 印花税
        value: 交易金额（含税费）
    """

    date: str
    stock_code: str
    action: str  # "buy" or "sell"
    price: float
    shares: int
    commission: float
    tax: float
    value: float

    def __post_init__(self) -> None:
        """验证交易数据的合法性。"""
        if self.action not in ("buy", "sell"):
            raise ValueError(f"action 必须是 'buy' 或 'sell'，实际: {self.action}")
        if self.shares % 100 != 0:
            raise ValueError(f"股数必须为整百的倍数，实际: {self.shares}")
        if self.shares <= 0:
            raise ValueError(f"股数必须为正数，实际: {self.shares}")
        if self.price <= 0:
            raise ValueError(f"价格必须为正数，实际: {self.price}")
        if self.value < 0:
            raise ValueError(f"交易金额不能为负，实际: {self.value}")

    @property
    def turnover(self) -> float:
        """成交额（不含税费）。"""
        return self.shares * self.price

    @property
    def total_cost(self) -> float:
        """交易总成本（含税费）。"""
        return self.commission + self.tax


@dataclass
class BacktestResult:
    """回测运行结果。

    Attributes:
        initial_capital: 初始资金
        final_value: 最终总资产
        nav_series: 每日净值序列 (pd.Series, index=date, values=nav)
        trades: 所有调仓交易记录列表
        risk_events: 风控事件列表
        positions_history: 每日持仓快照 {date: {stock_code: (shares, market_value)}}
        peak_date: 最大回撤区间峰值日期
        valley_date: 最大回撤区间谷值日期
        recovery_date: 最大回撤区间恢复日期（None 表示未恢复）
    """

    initial_capital: float
    final_value: float
    nav_series: pd.Series
    trades: List[Trade] = field(default_factory=list)
    risk_events: List[Dict] = field(default_factory=list)
    positions_history: Dict[str, Dict[str, tuple]] = field(default_factory=dict)
    peak_date: Optional[str] = None
    valley_date: Optional[str] = None
    recovery_date: Optional[str] = None

    def __post_init__(self) -> None:
        """验证回测结果的完整性。"""
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital 必须为正，实际: {self.initial_capital}")
        if self.final_value < 0:
            raise ValueError(f"final_value 不能为负，实际: {self.final_value}")
        if not isinstance(self.nav_series, pd.Series) or self.nav_series.empty:
            raise ValueError("nav_series 必须为非空的 pd.Series")

    @property
    def total_return(self) -> float:
        """总收益率。"""
        return self.final_value / self.initial_capital - 1.0

    @property
    def num_trades(self) -> int:
        """总交易笔数。"""
        return len(self.trades)

    @property
    def num_risk_events(self) -> int:
        """风控事件总数。"""
        return len(self.risk_events)

    @property
    def start_date(self) -> str:
        """回测起始日期。"""
        return str(self.nav_series.index[0])

    @property
    def end_date(self) -> str:
        """回测结束日期。"""
        return str(self.nav_series.index[-1])

    @property
    def duration_days(self) -> int:
        """回测持续天数。"""
        return len(self.nav_series)
