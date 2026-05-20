"""
事件驱动回测引擎。

实现 BacktestEngine 类，按交易日推进，支持：
- 定期调仓（开盘价成交、滑点、佣金、印花税）
- 风险控制器集成
- 每日组合净值记录
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from loguru import logger

from backtest.result import BacktestResult, Trade


class BacktestEngine:
    """事件驱动回测引擎。

    按交易日为单位推进模拟，在每个调仓日根据选股信号执行
    虚拟交易，扣除交易成本，并每日记录组合净值。

    Args:
        config_path: YAML 配置文件路径，包含回测参数。
    """

    def __init__(self, config_path: str) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            config: Dict[str, Any] = yaml.safe_load(f)

        self.initial_capital: float = float(config["initial_capital"])
        self.commission_rate: float = float(config.get("commission_rate", 0.0003))
        self.stamp_duty_rate: float = float(config.get("stamp_duty_rate", 0.001))
        self.slippage: float = float(config.get("slippage", 0.001))
        self.rebalance_frequency: int = int(
            config.get("rebalance_frequency", 21)
        )

        logger.info(
            "回测引擎初始化 | 初始资金={:.2f} 佣金率={:.4%} 印花税率={:.4%} "
            "滑点={:.4%} 调仓频率={}天",
            self.initial_capital,
            self.commission_rate,
            self.stamp_duty_rate,
            self.slippage,
            self.rebalance_frequency,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        signals: Dict[str, Dict[str, float]],
        prices: pd.DataFrame,
        market_states: Optional[Dict[str, Any]] = None,
    ) -> BacktestResult:
        """运行回测。

        Args:
            signals: 每日选股信号字典。
                     格式 {date_str: {stock_code: target_weight}}
            prices: 日线价格数据。MultiIndex DataFrame，
                    第一层 index 为日期 (str)，第二层为股票代码 (str)。
                    必须包含 'open' / 'close' 列。
            market_states: 可选，市场状态 {date_str: state_value}，
                           传递给风险控制器。

        Returns:
            BacktestResult 对象，包含净值序列、交易记录等。
        """
        market_states = market_states or {}
        prices_sorted = prices.sort_index()
        trading_dates: List[str] = sorted(prices_sorted.index.get_level_values(0).unique())

        if not trading_dates:
            logger.error("价格数据中没有交易日")
            raise ValueError("prices 数据为空")

        logger.info("回测开始 | 日期范围: {} ~ {} | 共 {} 个交易日",
                     trading_dates[0], trading_dates[-1], len(trading_dates))

        # 初始化投资组合
        portfolio: Dict[str, Any] = {
            "cash": self.initial_capital,
            "positions": {},  # {stock_code: shares}
            "history": [],    # [(date, nav)]
        }

        trades_record: List[Trade] = []
        risk_events: List[Dict] = []
        positions_history: Dict[str, Dict[str, tuple]] = {}

        # 计算调仓日索引（基于 rebalance_frequency）
        rebalance_indices: set = set(
            range(0, len(trading_dates), self.rebalance_frequency)
        )

        for day_idx, date in enumerate(trading_dates):
            day_prices = prices_sorted.xs(date, level=0)

            # ---- 检查是否调仓 ----
            is_rebalance = day_idx in rebalance_indices

            if is_rebalance and date in signals:
                target_weights = signals[date]
                portfolio = self._rebalance(
                    portfolio=portfolio,
                    target_weights=target_weights,
                    prices=day_prices,
                    date=date,
                    trades_record=trades_record,
                )

            # ---- 风控检查 ----
            if market_states and date in market_states:
                risk_signal = market_states[date]
                r_event = self._apply_risk_control(portfolio, risk_signal)
                if r_event is not None:
                    risk_events.append(r_event)

            # ---- 每日估值 ----
            positions_value = 0.0
            for stock_code, shares in list(portfolio["positions"].items()):
                if stock_code in day_prices.index:
                    close_px = float(day_prices.loc[stock_code, "close"])
                    market_value = shares * close_px
                    positions_value += market_value
                else:
                    # 股票不在当天价格中，按 0 估值
                    logger.warning("{} {} 无当日价格，按 0 估值", date, stock_code)

            total_value = portfolio["cash"] + positions_value
            nav = total_value / self.initial_capital

            portfolio["history"].append((date, nav))

            # ---- 记录持仓快照 ----
            snapshot: Dict[str, tuple] = {}
            for sc, sh in portfolio["positions"].items():
                if sc in day_prices.index:
                    cp = float(day_prices.loc[sc, "close"])
                    snapshot[sc] = (sh, sh * cp)
                else:
                    snapshot[sc] = (sh, 0.0)
            positions_history[date] = snapshot

        # ---- 生成结果 ----
        nav_series = pd.Series(
            data=[h[1] for h in portfolio["history"]],
            index=[h[0] for h in portfolio["history"]],
            name="nav",
        )

        peak, valley, recovery = self._locate_max_drawdown(nav_series)

        result = BacktestResult(
            initial_capital=self.initial_capital,
            final_value=nav * self.initial_capital,
            nav_series=nav_series,
            trades=trades_record,
            risk_events=risk_events,
            positions_history=positions_history,
            peak_date=peak,
            valley_date=valley,
            recovery_date=recovery,
        )

        logger.info("回测结束 | 最终净值={:.4f} | 总交易={} 笔 | 风控事件={} 次",
                     nav, len(trades_record), len(risk_events))
        return result

    def get_positions(self, date: str) -> Dict[str, int]:
        """获取指定日期的持仓详情。

        Args:
            date: 日期字符串 (YYYY-MM-DD)

        Returns:
            持仓字典 {stock_code: shares}
        """
        # 该方法需要在 run() 之后调用，或由外部提供 portfolio 状态。
        # 当前作为查询接口保留，实际使用时需配合上下文。
        raise NotImplementedError(
            "get_positions 在回测运行后无法单独查询；"
            "请从 BacktestResult.positions_history 中获取。"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebalance(
        self,
        portfolio: Dict[str, Any],
        target_weights: Dict[str, float],
        prices: pd.DataFrame,
        date: str,
        trades_record: List[Trade],
    ) -> Dict[str, Any]:
        """在调仓日执行组合再平衡。

        根据目标权重计算目标市值，然后逐一执行买卖交易。

        Args:
            portfolio: 当前投资组合状态
            target_weights: {stock_code: target_weight}
            prices: 当日所有股票价格数据
            date: 当前日期
            trades_record: 交易记录列表（就地追加）

        Returns:
            更新后的投资组合
        """
        total_capital = portfolio["cash"] + sum(
            shares * self._get_price(stock_code, prices, "open")
            for stock_code, shares in portfolio["positions"].items()
        )

        # 获取当前持仓中所有股票 + 信号中所有股票
        all_stocks = set(portfolio["positions"].keys()) | set(target_weights.keys())

        for stock_code in all_stocks:
            # 获取开盘价
            if stock_code not in prices.index:
                # 没有价格数据，跳过
                continue
            open_price = float(prices.loc[stock_code, "open"])
            if open_price <= 0:
                continue

            target_weight = target_weights.get(stock_code, 0.0)
            target_value = total_capital * target_weight

            portfolio = self._execute_trade(
                portfolio=portfolio,
                stock_code=stock_code,
                target_value=target_value,
                price=open_price,
                date=date,
                trades_record=trades_record,
            )

        # 清理零持仓
        portfolio["positions"] = {
            sc: sh for sc, sh in portfolio["positions"].items() if sh > 0
        }

        return portfolio

    def _execute_trade(
        self,
        portfolio: Dict[str, Any],
        stock_code: str,
        target_value: float,
        price: float,
        date: str,
        trades_record: List[Trade],
    ) -> Dict[str, Any]:
        """执行单笔交易。

        根据目标市值计算目标股数（四舍五入到整百股），
        扣除佣金、印花税和滑点后更新持仓与现金。

        Args:
            portfolio: 当前投资组合
            stock_code: 股票代码
            target_value: 目标持仓市值
            price: 当前价格
            date: 交易日期
            trades_record: 交易记录列表（就地追加）

        Returns:
            更新后的投资组合
        """
        current_shares = portfolio["positions"].get(stock_code, 0)
        current_value = current_shares * price

        # 滑点调整价格
        if target_value > current_value:
            # 买入：价格向上滑点
            exec_price = price * (1 + self.slippage)
        elif target_value < current_value:
            # 卖出：价格向下滑点
            exec_price = price * (1 - self.slippage)
        else:
            return portfolio  # 无需交易

        target_shares = round_to_lot(target_value / exec_price, lot_size=100)
        delta = target_shares - current_shares

        if delta > 0:
            # ---- 买入 ----
            cost = delta * exec_price
            commission = cost * self.commission_rate
            # A 股买入不收印花税
            tax = 0.0
            total_cost = cost + commission + tax

            if total_cost > portfolio["cash"]:
                # 现金不足，按可用资金重新计算
                max_cost = portfolio["cash"]
                # 反向解出可买股数
                fee_factor = 1 + self.commission_rate
                max_cost_excl_fee = max_cost / fee_factor
                delta = round_to_lot(max_cost_excl_fee / exec_price, lot_size=100)
                if delta <= 0:
                    return portfolio
                cost = delta * exec_price
                commission = cost * self.commission_rate
                total_cost = cost + commission + tax

            portfolio["cash"] -= total_cost
            portfolio["positions"][stock_code] = portfolio["positions"].get(
                stock_code, 0
            ) + delta

            trades_record.append(
                Trade(
                    date=date,
                    stock_code=stock_code,
                    action="buy",
                    price=round(exec_price, 4),
                    shares=delta,
                    commission=round(commission, 2),
                    tax=round(tax, 2),
                    value=round(total_cost, 2),
                )
            )

        elif delta < 0:
            # ---- 卖出 ----
            sell_shares = -delta
            proceeds = sell_shares * exec_price
            commission = proceeds * self.commission_rate
            # 仅卖出时收印花税
            tax = proceeds * self.stamp_duty_rate
            net_proceeds = proceeds - commission - tax

            portfolio["cash"] += net_proceeds
            portfolio["positions"][stock_code] = current_shares - sell_shares

            trades_record.append(
                Trade(
                    date=date,
                    stock_code=stock_code,
                    action="sell",
                    price=round(exec_price, 4),
                    shares=sell_shares,
                    commission=round(commission, 2),
                    tax=round(tax, 2),
                    value=round(net_proceeds, 2),
                )
            )

        return portfolio

    def _apply_risk_control(
        self, portfolio: Dict[str, Any], risk_signals: Any
    ) -> Optional[Dict[str, Any]]:
        """应用风险控制。

        根据风险控制器的输出对投资组合进行调整。
        当前实现为框架——子类可覆写此方法以实现具体风控逻辑。

        Args:
            portfolio: 当前投资组合
            risk_signals: 风险信号，具体格式由风控模块决定

        Returns:
            风控事件字典，若未触发风控则返回 None。
        """
        # 默认实现：记录风控信号，不做实际仓位调整
        logger.debug("风控信号: {}", risk_signals)
        return {
            "type": "info",
            "signal": risk_signals,
            "action": "none",
        }

    @staticmethod
    def _get_price(
        stock_code: str, prices: pd.DataFrame, field: str
    ) -> float:
        """从价格 DataFrame 中获取指定字段值。

        Args:
            stock_code: 股票代码
            prices: 当日价格 DataFrame
            field: 价格字段名（open / close / high / low）

        Returns:
            价格值，无法获取时返回 0.0
        """
        try:
            return float(prices.loc[stock_code, field])
        except (KeyError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def _locate_max_drawdown(
        nav_series: pd.Series,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """定位最大回撤区间。

        Args:
            nav_series: 净值序列 (index=date)

        Returns:
            (peak_date, valley_date, recovery_date)
        """
        if len(nav_series) < 2:
            return None, None, None

        cummax = nav_series.cummax()
        drawdown = nav_series / cummax - 1.0

        valley_idx = drawdown.idxmin()
        valley_date = str(valley_idx)

        # 峰值是谷值之前 cummax 首次到达峰值的日期
        peak_mask = cummax.loc[:valley_idx]
        peak_date = str(peak_mask.idxmax())

        # 恢复日期：谷值之后第一次回到峰值的位置
        peak_value = cummax.loc[peak_date]
        recovery_candidates = nav_series.loc[valley_idx:]
        recovery_candidates = recovery_candidates[recovery_candidates >= peak_value]
        recovery_date = str(recovery_candidates.index[0]) if not recovery_candidates.empty else None

        return peak_date, valley_date, recovery_date


# ------------------------------------------------------------------
# 模块级别工具函数
# ------------------------------------------------------------------

def round_to_lot(value: float, lot_size: int = 100) -> int:
    """四舍五入到整手股数。

    使用标准算术四舍五入（而非 Python 的 banker's rounding），
    确保 .5 边界值正确向上取整。

    Args:
        value: 计算出的股数（浮点数）
        lot_size: 每手股数，A 股通常为 100

    Returns:
        整手倍数
    """
    lots = int(value / lot_size + 0.5)
    return max(0, lots * lot_size)
