"""
传统因子计算模块

提供 FactorCalculator 类，基于日线行情数据的 MultiIndex DataFrame
计算约 20 个核心量价因子与基本面因子。

输入格式:
    DataFrame, index=MultiIndex([date, stock_code])
    columns = [open, high, low, close, volume, amount, turnover_rate, pe, pb, roe]

因子方法返回格式:
    DataFrame, index=date, columns=stock_code
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class FactorCalculator:
    """基于日线行情数据的传统因子计算器。"""

    # 输入必需列
    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume", "amount",
                        "turnover_rate", "pe", "pb", "roe"]

    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        初始化计算器。

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据，index=MultiIndex([date, stock_code])
        """
        self._df = df

    def _resolve_df(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """解析传入的 df 或使用实例存储的 df。"""
        if df is not None:
            return df
        if self._df is None:
            raise ValueError("请提供行情数据 DataFrame")
        return self._df

    def _validate(self, df: pd.DataFrame) -> None:
        """验证输入 DataFrame 的列和索引结构。"""
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必需列: {missing}")
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("索引必须是 MultiIndex([date, stock_code])")
        if df.index.nlevels != 2:
            raise ValueError("MultiIndex 必须有两层")

    def _unstack_df(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """将指定列按 (date, stock_code) 解栈为 date x stock_code 矩阵。"""
        return df[col].unstack()

    def _compute_returns(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算指定周期的收益率。

        Parameters
        ----------
        df : pd.DataFrame
            日线行情数据
        period : int
            周期天数

        Returns
        -------
        pd.DataFrame
            date x stock_code 的收益率矩阵
        """
        close = self._unstack_df(df, "close")
        return close / close.shift(period) - 1.0

    # ------------------------------------------------------------------
    # 量价因子组
    # ------------------------------------------------------------------

    def momentum_1m(self, df: Optional[pd.DataFrame] = None, N: int = 20) -> pd.DataFrame:
        """过去 20 日收益率 (动量因子, 1 个月)。

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据
        N : int, default=20
            计算窗口

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        result = self._compute_returns(self._resolve_df(df), N)
        return result

    def momentum_3m(self, df: Optional[pd.DataFrame] = None, N: int = 60) -> pd.DataFrame:
        """过去 60 日收益率 (动量因子, 3 个月)。"""
        result = self._compute_returns(self._resolve_df(df), N)
        return result

    def momentum_6m(self, df: Optional[pd.DataFrame] = None, N: int = 120) -> pd.DataFrame:
        """过去 120 日收益率 (动量因子, 6 个月)。"""
        result = self._compute_returns(self._resolve_df(df), N)
        return result

    def reversal_5d(self, df: Optional[pd.DataFrame] = None, N: int = 5) -> pd.DataFrame:
        """过去 5 日反转因子 (负收益)。

        反转因子 = -1 * 过去 N 日收益率。高值表示前期跌幅大，预期反转。

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据
        N : int, default=5
            计算窗口

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        ret = self._compute_returns(self._resolve_df(df), N)
        return -ret

    def ma_deviation(self, df: Optional[pd.DataFrame] = None, N: int = 20) -> pd.DataFrame:
        """收盘价偏离 N 日均线的比例。

        deviation = (close / SMA(close, N)) - 1

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据
        N : int, default=20
            均线窗口

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        _df = self._resolve_df(df)
        close = self._unstack_df(_df, "close")
        ma = close.rolling(window=N, min_periods=N).mean()
        return close / ma - 1.0

    def volatility_20d(self, df: Optional[pd.DataFrame] = None, N: int = 20) -> pd.DataFrame:
        """过去 20 日波动率 (对数收益率标准差)。

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据
        N : int, default=20
            计算窗口

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        _df = self._resolve_df(df)
        close = self._unstack_df(_df, "close")
        log_ret = np.log(close / close.shift(1))
        vol = log_ret.rolling(window=N, min_periods=N).std()
        return vol

    def volatility_60d(self, df: Optional[pd.DataFrame] = None, N: int = 60) -> pd.DataFrame:
        """过去 60 日波动率。"""
        _df = self._resolve_df(df)
        close = self._unstack_df(_df, "close")
        log_ret = np.log(close / close.shift(1))
        vol = log_ret.rolling(window=N, min_periods=N).std()
        return vol

    def max_drawdown_20d(self, df: Optional[pd.DataFrame] = None, N: int = 20) -> pd.DataFrame:
        """过去 N 日最大回撤。

        max_drawdown = max(1 - price / cummax(price)) 在窗口期内。

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据
        N : int, default=20
            计算窗口

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        _df = self._resolve_df(df)
        close = self._unstack_df(_df, "close")

        def _rolling_mdd(series: pd.Series) -> float:
            cummax = series.cummax()
            dd = series / cummax - 1.0
            return dd.min()

        mdd = close.rolling(window=N, min_periods=N).apply(_rolling_mdd, raw=False)
        return mdd

    def turnover_rate(self, df: Optional[pd.DataFrame] = None, N: int = 20) -> pd.DataFrame:
        """过去 N 日平均换手率。

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据
        N : int, default=20
            计算窗口

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        _df = self._resolve_df(df)
        turnover = self._unstack_df(_df, "turnover_rate")
        return turnover.rolling(window=N, min_periods=N).mean()

    def turnover_change(self, df: Optional[pd.DataFrame] = None, N: int = 5) -> pd.DataFrame:
        """换手率近期变化: 近 N 日均值 / 近 60 日均值 - 1。

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据
        N : int, default=5
            短期窗口

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        _df = self._resolve_df(df)
        turnover = self._unstack_df(_df, "turnover_rate")
        short_ma = turnover.rolling(window=N, min_periods=N).mean()
        long_ma = turnover.rolling(window=60, min_periods=60).mean()
        return short_ma / long_ma - 1.0

    def volume_ratio(self, df: Optional[pd.DataFrame] = None, N: int = 5) -> pd.DataFrame:
        """量比: 近 N 日平均成交量 / 近 20 日平均成交量。

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据
        N : int, default=5
            短期窗口

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        _df = self._resolve_df(df)
        volume = self._unstack_df(_df, "volume")
        short_ma = volume.rolling(window=N, min_periods=N).mean()
        long_ma = volume.rolling(window=20, min_periods=20).mean()
        return short_ma / long_ma

    def amihud_illiquidity(self, df: Optional[pd.DataFrame] = None, N: int = 20) -> pd.DataFrame:
        """Amihud 非流动性指标。

        计算方式: 过去 N 日 |日收益率| / 成交金额(元) 的均值, 再乘以 10^6。
        值越大表示流动性越差。

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据
        N : int, default=20
            计算窗口

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        _df = self._resolve_df(df)
        close = self._unstack_df(_df, "close")
        amount = self._unstack_df(_df, "amount")

        daily_ret = close.pct_change().abs()
        illiq_ratio = daily_ret / amount  # |r| / dollar volume
        illiq_ratio = illiq_ratio.rolling(window=N, min_periods=N).mean() * 1e6
        return illiq_ratio

    def close_to_high(self, df: Optional[pd.DataFrame] = None, N: int = 20) -> pd.DataFrame:
        """收盘价距 N 日最高价的比例。

        ratio = close / rolling_max(high, N)

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据
        N : int, default=20
            计算窗口

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        _df = self._resolve_df(df)
        close = self._unstack_df(_df, "close")
        high = self._unstack_df(_df, "high")
        rolling_high = high.rolling(window=N, min_periods=N).max()
        return close / rolling_high

    def close_to_low(self, df: Optional[pd.DataFrame] = None, N: int = 20) -> pd.DataFrame:
        """收盘价距 N 日最低价的比例。

        ratio = close / rolling_min(low, N)

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据
        N : int, default=20
            计算窗口

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        _df = self._resolve_df(df)
        close = self._unstack_df(_df, "close")
        low = self._unstack_df(_df, "low")
        rolling_low = low.rolling(window=N, min_periods=N).min()
        return close / rolling_low

    # ------------------------------------------------------------------
    # 基本面因子组
    # ------------------------------------------------------------------

    def pe_rank(self, df: Optional[pd.DataFrame] = None,
                window: int = 252) -> pd.DataFrame:
        """PE 的历史分位数 (滚动 window 天)。

        值域 [0, 1]，表示当前 PE 在历史窗口中的位置。

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据
        window : int, default=252
            滚动窗口 (约 1 年交易日)

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        _df = self._resolve_df(df)
        pe = self._unstack_df(_df, "pe")

        def _rank_pct(series: pd.Series) -> float:
            return series.rank(pct=True).iloc[-1]

        rank = pe.rolling(window=window, min_periods=int(window * 0.6)).apply(
            _rank_pct, raw=False
        )
        return rank

    def pb_rank(self, df: Optional[pd.DataFrame] = None,
                window: int = 252) -> pd.DataFrame:
        """PB 的历史分位数 (滚动 window 天)。"""
        _df = self._resolve_df(df)
        pb = self._unstack_df(_df, "pb")

        def _rank_pct(series: pd.Series) -> float:
            return series.rank(pct=True).iloc[-1]

        rank = pb.rolling(window=window, min_periods=int(window * 0.6)).apply(
            _rank_pct, raw=False
        )
        return rank

    def roe_factor(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """ROE 值 (净资产收益率)。

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        _df = self._resolve_df(df)
        return self._unstack_df(_df, "roe")

    def earning_yield(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """盈利收益率 (E/P = 1/PE)。

        高盈利收益率通常表示被低估。

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        _df = self._resolve_df(df)
        pe = self._unstack_df(_df, "pe")
        # 避免除零
        ey = np.where(pe > 0, 1.0 / pe, np.nan)
        return pd.DataFrame(ey, index=pe.index, columns=pe.columns)

    def revenue_growth(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """营收同比增长率。

        注意: 需要 DataFrame 中包含 'revenue' 列 (营业收入)。
        若缺少该列则会警告并返回全 NaN。

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据 (需含 'revenue' 列)

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        _df = self._resolve_df(df)
        if "revenue" not in _df.columns:
            import warnings
            warnings.warn("'revenue' 列不存在, 返回全 NaN")
            close = self._unstack_df(_df, "close")
            return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        revenue = self._unstack_df(_df, "revenue")
        yoy = revenue.pct_change(periods=252)  # 年对年变化
        return yoy

    def profit_growth(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """净利润同比增长率。

        注意: 需要 DataFrame 中包含 'profit' 列 (净利润)。
        若缺少该列则会警告并返回全 NaN。

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据 (需含 'profit' 列)

        Returns
        -------
        pd.DataFrame
            date x stock_code
        """
        _df = self._resolve_df(df)
        if "profit" not in _df.columns:
            import warnings
            warnings.warn("'profit' 列不存在, 返回全 NaN")
            close = self._unstack_df(_df, "close")
            return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        profit = self._unstack_df(_df, "profit")
        yoy = profit.pct_change(periods=252)
        return yoy

    # ------------------------------------------------------------------
    # 批量计算
    # ------------------------------------------------------------------

    def compute_all(self, df: Optional[pd.DataFrame] = None,
                    factor_list: Optional[List[str]] = None) -> pd.DataFrame:
        """计算所有或指定因子, 返回因子 DataFrame。

        Parameters
        ----------
        df : pd.DataFrame, optional
            日线行情数据
        factor_list : list of str, optional
            要计算的因子名列表。为 None 时计算所有因子。

        Returns
        -------
        pd.DataFrame
            因子数据, index=date, columns=MultiIndex([factor_name, stock_code])
        """
        _df = df if df is not None else self._df
        if _df is None:
            raise ValueError("请提供行情数据 DataFrame")

        self._validate(_df)

        # 所有可用因子方法及其默认参数
        all_factor_methods: Dict[str, tuple] = {
            "momentum_1m":    (self.momentum_1m, {}),
            "momentum_3m":    (self.momentum_3m, {}),
            "momentum_6m":    (self.momentum_6m, {}),
            "reversal_5d":    (self.reversal_5d, {}),
            "ma_deviation":   (self.ma_deviation, {}),
            "volatility_20d": (self.volatility_20d, {}),
            "volatility_60d": (self.volatility_60d, {}),
            "max_drawdown_20d": (self.max_drawdown_20d, {}),
            "turnover_rate":  (self.turnover_rate, {}),
            "turnover_change": (self.turnover_change, {}),
            "volume_ratio":   (self.volume_ratio, {}),
            "amihud_illiquidity": (self.amihud_illiquidity, {}),
            "close_to_high":  (self.close_to_high, {}),
            "close_to_low":   (self.close_to_low, {}),
            "pe_rank":        (self.pe_rank, {}),
            "pb_rank":        (self.pb_rank, {}),
            "roe_factor":     (self.roe_factor, {}),
            "earning_yield":  (self.earning_yield, {}),
            "revenue_growth": (self.revenue_growth, {}),
            "profit_growth":  (self.profit_growth, {}),
        }

        to_compute = factor_list or list(all_factor_methods.keys())
        unknown = [f for f in to_compute if f not in all_factor_methods]
        if unknown:
            raise ValueError(f"未知因子名: {unknown}")

        frames: Dict[str, pd.DataFrame] = {}
        for name in to_compute:
            method, kwargs = all_factor_methods[name]
            frames[name] = method(df=_df, **kwargs)

        # 合并为 MultiIndex columns
        concat = pd.concat(frames, axis=1)  # 外层 = factor_name
        concat.columns = pd.MultiIndex.from_tuples(concat.columns)
        return concat


def compute_all_factors(df: pd.DataFrame,
                        factor_list: Optional[List[str]] = None) -> pd.DataFrame:
    """便捷函数: 一步计算所有因子。

    Parameters
    ----------
    df : pd.DataFrame
        日线行情数据
    factor_list : list of str, optional
        因子名列表

    Returns
    -------
    pd.DataFrame
        MultiIndex columns 因子数据
    """
    calc = FactorCalculator()
    return calc.compute_all(df, factor_list)
