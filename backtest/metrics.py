"""
回测性能指标计算模块。

提供 PerformanceMetrics 类，基于 BacktestResult 计算全面的
收益与风险评估指标，包括年化收益、最大回撤、夏普比率等。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None  # type: ignore[assignment]

from backtest.result import BacktestResult


class PerformanceMetrics:
    """回测性能指标计算器。

    基于 BacktestResult 中的净值序列、交易记录和风控事件，
    计算完整的收益与风险评估指标。

    所有方法均为静态方法，可直接调用或实例化后调用。
    """

    TRADING_DAYS_PER_YEAR: int = 252
    """A 股年化交易日数。"""

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """初始化指标计算器。

        Args:
            risk_free_rate: 年化无风险利率，默认 2%
        """
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # 收益指标
    # ------------------------------------------------------------------

    def calc_annual_return(self, nav_series: pd.Series) -> float:
        """计算年化收益率（CAGR）。

        根据净值序列的起止值，按时间长度计算复合年化增长率。

        Args:
            nav_series: 净值序列 (index=date)

        Returns:
            年化收益率（小数形式，如 0.15 表示 15%）
        """
        if len(nav_series) < 2:
            return 0.0

        start_nav = nav_series.iloc[0]
        end_nav = nav_series.iloc[-1]

        # 根据实际日历天数计算年数
        try:
            days = (pd.to_datetime(nav_series.index[-1])
                    - pd.to_datetime(nav_series.index[0])).days
        except Exception:
            days = len(nav_series)

        num_years = days / 365.0

        if num_years <= 0 or start_nav <= 0:
            return 0.0

        return float((end_nav / start_nav) ** (1.0 / num_years) - 1.0)

    def calc_win_rate(self, nav_series: pd.Series) -> float:
        """计算胜率（正收益交易日占比）。

        Args:
            nav_series: 净值序列

        Returns:
            胜率，范围 [0, 1]
        """
        if len(nav_series) < 2:
            return 0.0

        daily_returns = nav_series.pct_change().dropna()
        if len(daily_returns) == 0:
            return 0.0

        return float((daily_returns > 0).sum() / len(daily_returns))

    def calc_profit_factor(self, nav_series: pd.Series) -> float:
        """计算盈亏比（Profit Factor）。

        总盈利 / 总亏损绝对值。当日收益为正则累加，为负则累加亏损。
        若亏损为 0，返回无穷大。

        Args:
            nav_series: 净值序列

        Returns:
            盈亏比
        """
        if len(nav_series) < 3:
            return 0.0

        daily_returns = nav_series.pct_change().dropna()
        total_gain = daily_returns[daily_returns > 0].sum()
        total_loss = -daily_returns[daily_returns < 0].sum()

        if total_loss == 0:
            return float("inf") if total_gain > 0 else 1.0

        return float(total_gain / total_loss)

    def calc_monthly_returns(
        self, nav_series: pd.Series
    ) -> Dict[str, Any]:
        """计算月度收益率分析。

        将净值序列按自然月分组，计算每个月的收益率。
        返回包含月度矩阵、正/负月份统计等信息。

        Args:
            nav_series: 净值序列 (index 可解析为 datetime)

        Returns:
            dict，包含：
            - "returns": 月度收益率 Series (index=YYYY-MM)
            - "positive_months": 正收益月数
            - "negative_months": 负收益月数
            - "win_rate": 月度胜率
            - "best_month": 最佳月收益
            - "worst_month": 最差月收益
        """
        if len(nav_series) < 2:
            return {}

        idx = pd.to_datetime(nav_series.index)
        values = pd.Series(nav_series.values, index=idx)
        monthly_groups = values.resample("ME")

        monthly_returns: Dict[str, float] = {}
        for label, group in monthly_groups:
            if len(group) >= 2:
                ret = group.iloc[-1] / group.iloc[0] - 1.0
                monthly_returns[str(label.strftime("%Y-%m"))] = round(ret, 6)

        ret_series = pd.Series(monthly_returns)
        pos = (ret_series > 0).sum()
        neg = (ret_series < 0).sum()

        return {
            "returns": ret_series,
            "positive_months": int(pos),
            "negative_months": int(neg),
            "win_rate": float(pos / (pos + neg)) if (pos + neg) > 0 else 0.0,
            "best_month": float(ret_series.max()) if not ret_series.empty else 0.0,
            "worst_month": float(ret_series.min()) if not ret_series.empty else 0.0,
        }

    # ------------------------------------------------------------------
    # 风险指标
    # ------------------------------------------------------------------

    def calc_max_drawdown(
        self, nav_series: pd.Series
    ) -> Tuple[float, Optional[str], Optional[str], Optional[str]]:
        """计算最大回撤。

        从净值峰值到后续谷值的最大跌幅。

        Args:
            nav_series: 净值序列

        Returns:
            (max_drawdown, peak_date, valley_date, recovery_date)
            - max_drawdown: 最大回撤幅度（正数，如 0.2 表示回撤 20%）
            - peak_date: 峰值日期
            - valley_date: 谷值日期
            - recovery_date: 恢复日期（None 表示尚未恢复）
        """
        if len(nav_series) < 2:
            return 0.0, None, None, None

        rolling_max = nav_series.expanding().max()
        drawdown = (nav_series - rolling_max) / rolling_max

        valley_idx = drawdown.idxmin()
        max_dd = float(abs(drawdown.min()))

        if max_dd <= 0:
            return 0.0, None, None, None

        # 峰值在 valley 之前
        peak_idx = rolling_max.loc[:valley_idx].idxmax()

        # 恢复日期（净值回到峰值后）
        recovery_candidates = nav_series.loc[valley_idx:]
        recovery_candidates = recovery_candidates[
            recovery_candidates >= nav_series[peak_idx]
        ]
        if len(recovery_candidates) > 1:
            recovery_date = str(recovery_candidates.index[1])
        else:
            recovery_date = None

        return (
            max_dd,
            str(peak_idx.date()) if hasattr(peak_idx, "date") else str(peak_idx),
            str(valley_idx.date()) if hasattr(valley_idx, "date") else str(valley_idx),
            recovery_date,
        )

    def calc_sharpe_ratio(
        self, nav_series: pd.Series, risk_free_rate: Optional[float] = None
    ) -> float:
        """计算夏普比率。

        Sharpe = (年化收益 - 无风险利率) / 年化波动率

        Args:
            nav_series: 净值序列
            risk_free_rate: 年化无风险利率，覆盖默认值

        Returns:
            夏普比率
        """
        rf = self.risk_free_rate if risk_free_rate is None else risk_free_rate
        if len(nav_series) < 5:
            return 0.0

        daily_returns = nav_series.pct_change().dropna()
        if len(daily_returns) == 0:
            return 0.0

        excess_daily = daily_returns - rf / self.TRADING_DAYS_PER_YEAR
        std_excess = float(excess_daily.std())

        if std_excess == 0:
            return 0.0

        sharpe = (np.sqrt(self.TRADING_DAYS_PER_YEAR)
                  * float(excess_daily.mean()) / std_excess)
        return float(sharpe)

    def calc_calmar_ratio(self, nav_series: pd.Series) -> float:
        """计算卡玛比率。

        Calmar = 年化收益 / 最大回撤

        Args:
            nav_series: 净值序列

        Returns:
            卡玛比率
        """
        ann_ret = self.calc_annual_return(nav_series)
        max_dd, _, _, _ = self.calc_max_drawdown(nav_series)

        if max_dd <= 0:
            return 0.0

        return ann_ret / max_dd

    def calc_sortino_ratio(
        self, nav_series: pd.Series, risk_free_rate: Optional[float] = None
    ) -> float:
        """计算索提诺比率。

        使用下行波动率替代总波动率，仅考虑负收益的波动。

        Args:
            nav_series: 净值序列
            risk_free_rate: 年化无风险利率，覆盖默认值

        Returns:
            索提诺比率
        """
        rf = self.risk_free_rate if risk_free_rate is None else risk_free_rate
        if len(nav_series) < 5:
            return 0.0

        daily_returns = nav_series.pct_change().dropna()
        if len(daily_returns) == 0:
            return 0.0

        daily_rf = rf / self.TRADING_DAYS_PER_YEAR
        excess = daily_returns - daily_rf

        downside = excess[excess < 0]
        if len(downside) < 2:
            return 0.0

        downside_vol = float(np.sqrt((downside**2).sum() / len(downside)))
        if downside_vol == 0:
            return 0.0

        sortino = (np.sqrt(self.TRADING_DAYS_PER_YEAR)
                   * float(excess.mean()) / downside_vol)
        return float(sortino)

    # ------------------------------------------------------------------
    # 滚动指标
    # ------------------------------------------------------------------

    def calc_rolling_metrics(
        self, nav_series: pd.Series, window: int = 252
    ) -> Dict[str, pd.Series]:
        """计算滚动指标。

        以指定窗口长度在净值序列上滑动，计算滚动年化收益、
        滚动年化波动率和滚动夏普比率。

        Args:
            nav_series: 净值序列
            window: 滚动窗口长度（交易日数），默认 252（一年）

        Returns:
            {
                "rolling_annual_return": pd.Series,
                "rolling_volatility": pd.Series,
                "rolling_sharpe": pd.Series,
            }
        """
        if len(nav_series) < window + 1:
            logger.warning(
                "净值序列长度 ({}) 不足以计算 {} 天滚动指标",
                len(nav_series), window,
            )
            return {}

        daily_returns = nav_series.pct_change().dropna()
        ann_factor = self.TRADING_DAYS_PER_YEAR

        rolling_ret = daily_returns.rolling(window).apply(
            lambda x: (1 + x).prod() ** (ann_factor / window) - 1
        )
        rolling_vol = daily_returns.rolling(window).std() * np.sqrt(ann_factor)
        rolling_sharpe = (rolling_ret - self.risk_free_rate) / rolling_vol

        return {
            "rolling_annual_return": rolling_ret,
            "rolling_volatility": rolling_vol,
            "rolling_sharpe": rolling_sharpe,
        }

    # ------------------------------------------------------------------
    # 综合报告
    # ------------------------------------------------------------------

    def summary(self, result: BacktestResult) -> Dict[str, Any]:
        """生成完整性能指标摘要字典。

        计算所有收益和风险指标，将结果汇总为便于序列化的字典。

        Args:
            result: BacktestResult 对象

        Returns:
            包含所有计算指标的字典
        """
        nav = result.nav_series

        max_dd, peak_d, valley_d, recovery_d = self.calc_max_drawdown(nav)
        monthly = self.calc_monthly_returns(nav)
        daily_returns = nav.pct_change().dropna()

        summary_dict: Dict[str, Any] = {
            # 基本信息
            "start_date": result.start_date,
            "end_date": result.end_date,
            "duration_days": result.duration_days,
            "initial_capital": result.initial_capital,
            "final_value": round(result.final_value, 2),
            "total_return": round(float(result.total_return), 4),
            "total_return_pct": f"{result.total_return * 100:.2f}%",

            # 收益指标
            "annual_return": round(self.calc_annual_return(nav), 4),
            "annual_return_pct": f"{self.calc_annual_return(nav) * 100:.2f}%",
            "win_rate": round(self.calc_win_rate(nav), 4),
            "profit_factor": round(self.calc_profit_factor(nav), 4),

            # 风险指标
            "max_drawdown": round(max_dd, 4),
            "max_drawdown_pct": f"{max_dd * 100:.2f}%",
            "peak_date": peak_d,
            "valley_date": valley_d,
            "recovery_date": recovery_d,
            "sharpe_ratio": round(self.calc_sharpe_ratio(nav), 4),
            "calmar_ratio": round(self.calc_calmar_ratio(nav), 4),
            "sortino_ratio": round(self.calc_sortino_ratio(nav), 4),
            "volatility": round(float(daily_returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)), 4),

            # 月度分析
            "monthly_positive": monthly.get("positive_months", 0),
            "monthly_negative": monthly.get("negative_months", 0),
            "monthly_win_rate": monthly.get("win_rate", 0.0),
            "best_month": monthly.get("best_month", 0.0),
            "worst_month": monthly.get("worst_month", 0.0),

            # 交易统计
            "num_trades": result.num_trades,
            "num_risk_events": result.num_risk_events,
        }

        return summary_dict

    def report(
        self,
        result: BacktestResult,
        save_path: Optional[str] = None,
    ) -> str:
        """生成格式化文本报告。

        使用 tabulate 库生成易于阅读的性能报告，
        包含收益指标、风险指标、月度收益矩阵和调仓统计。

        Args:
            result: BacktestResult 对象
            save_path: 可选，报告保存路径

        Returns:
            格式化后的报告文本
        """
        if tabulate is None:
            logger.warning(
                "tabulate 未安装，报告格式可能不美观。pip install tabulate"
            )
            table_fmt = "plain"
        else:
            table_fmt = "simple"

        nav = result.nav_series
        summary_dict = self.summary(result)
        monthly_info = self.calc_monthly_returns(nav)

        lines: List[str] = []
        lines.append("=" * 68)
        lines.append("                    回测性能评估报告")
        lines.append("=" * 68)

        # ---- 基本信息 ----
        lines.append("")
        lines.append("【基本信息】")
        info_table: List[List[str]] = [
            ["回测区间", f"{summary_dict['start_date']} ~ {summary_dict['end_date']}"],
            ["交易天数", str(summary_dict["duration_days"])],
            ["初始资金", f"{summary_dict['initial_capital']:,.2f}"],
            ["最终资产", f"{summary_dict['final_value']:,.2f}"],
            ["总交易笔数", str(summary_dict["num_trades"])],
            ["风控事件", str(summary_dict["num_risk_events"])],
        ]
        lines.append(tabulate(info_table, tablefmt=table_fmt))

        # ---- 收益指标 ----
        lines.append("")
        lines.append("【收益指标】")
        ret_table: List[List[str]] = [
            ["总收益率", summary_dict["total_return_pct"]],
            ["年化收益率 (CAGR)", summary_dict["annual_return_pct"]],
            ["胜率 (日)", f"{summary_dict['win_rate'] * 100:.2f}%"],
            ["盈亏比", f"{summary_dict['profit_factor']:.2f}"],
            ["月度胜率", f"{summary_dict['monthly_win_rate'] * 100:.2f}%"],
            ["最佳月份", f"{summary_dict['best_month'] * 100:.2f}%"],
            ["最差月份", f"{summary_dict['worst_month'] * 100:.2f}%"],
        ]
        lines.append(tabulate(ret_table, tablefmt=table_fmt))

        # ---- 风险指标 ----
        lines.append("")
        lines.append("【风险指标】")
        risk_table: List[List[str]] = [
            ["最大回撤", summary_dict["max_drawdown_pct"]],
            ["峰值日期", str(summary_dict["peak_date"])],
            ["谷值日期", str(summary_dict["valley_date"])],
            ["恢复日期", str(summary_dict["recovery_date"] or "未恢复")],
            ["夏普比率", f"{summary_dict['sharpe_ratio']:.4f}"],
            ["卡玛比率", f"{summary_dict['calmar_ratio']:.4f}"],
            ["索提诺比率", f"{summary_dict['sortino_ratio']:.4f}"],
            ["年化波动率", f"{summary_dict['volatility'] * 100:.2f}%"],
        ]
        lines.append(tabulate(risk_table, tablefmt=table_fmt))

        # ---- 月度收益矩阵 ----
        lines.append("")
        lines.append("【月度收益矩阵】")
        monthly_returns = monthly_info.get("returns")
        if (monthly_returns is not None
                and isinstance(monthly_returns, pd.Series)
                and not monthly_returns.empty):
            year_month_groups: Dict[str, List[Tuple[str, float]]] = {}
            for idx_val, ret_val in monthly_returns.items():
                year = str(idx_val)[:4]
                month_label = str(idx_val)[5:7]
                if year not in year_month_groups:
                    year_month_groups[year] = []
                year_month_groups[year].append((month_label, float(ret_val)))

            years = sorted(year_month_groups.keys())
            months_all = [f"{m:02d}" for m in range(1, 13)]

            matrix_header = ["月份"] + years + ["平均"]
            matrix_rows: List[List[str]] = []

            for m in months_all:
                row: List[str] = [m]
                for yr in years:
                    found = [r for label, r in year_month_groups[yr] if label == m]
                    if found:
                        row.append(f"{found[0] * 100:+.2f}%")
                    else:
                        row.append("--")
                vals = []
                for yr in years:
                    found = [r for label, r in year_month_groups[yr] if label == m]
                    if found:
                        vals.append(found[0])
                avg_val = float(np.mean(vals)) if vals else 0.0
                row.append(f"{avg_val * 100:+.2f}%")
                matrix_rows.append(row)

            lines.append(tabulate(
                matrix_rows, headers=matrix_header, tablefmt=table_fmt
            ))

            # 年度汇总
            lines.append("")
            yearly_rows: List[List[str]] = []
            for yr in years:
                yr_returns = [r for _, r in year_month_groups[yr]]
                yr_avg = float(np.mean(yr_returns)) if yr_returns else 0.0
                yr_total = float(
                    (1 + np.array(yr_returns)).prod() - 1
                ) if yr_returns else 0.0
                yr_pos = sum(1 for r in yr_returns if r > 0)
                yr_neg = sum(1 for r in yr_returns if r < 0)
                yearly_rows.append([
                    yr,
                    f"{yr_avg * 100:+.2f}%",
                    f"{yr_total * 100:+.2f}%",
                    f"{yr_pos}/{yr_pos + yr_neg}",
                ])
            lines.append(tabulate(
                yearly_rows,
                headers=["年份", "月均收益", "年度收益", "胜率(月)"],
                tablefmt=table_fmt,
            ))
        else:
            lines.append("  （无月度数据）")

        # ---- 交易统计 ----
        lines.append("")
        lines.append("【交易统计】")
        if result.trades:
            buys = sum(1 for t in result.trades if t.action == "buy")
            sells = sum(1 for t in result.trades if t.action == "sell")
            total_commission = sum(t.commission for t in result.trades)
            total_tax = sum(t.tax for t in result.trades)
            trade_table: List[List[str]] = [
                ["买入笔数", str(buys)],
                ["卖出笔数", str(sells)],
                ["总佣金", f"{total_commission:,.2f}"],
                ["总印花税", f"{total_tax:,.2f}"],
                ["总交易成本", f"{total_commission + total_tax:,.2f}"],
            ]
            lines.append(tabulate(trade_table, tablefmt=table_fmt))
        else:
            lines.append("  （无交易记录）")

        lines.append("")
        lines.append("=" * 68)

        report_text = "\n".join(lines)

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info("性能报告已保存至: {}", save_path)

        return report_text
