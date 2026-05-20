"""
市场状态判断 (MarketTimer)
=========================
基于技术指标判断当前市场状态，动态调整仓位上限。

状态分类:
  - trend_up:    趋势上涨 → 满仓(100%)
  - sideways:    震荡行情 → 半仓(60%)
  - trend_down:  趋势下跌 → 轻仓(30%)
  - high_vol:    高波动 → 极低仓位(20%)
"""

import numpy as np
import pandas as pd
from loguru import logger


class MarketTimer:
    """市场状态判断器"""

    # 市场状态常量
    TREND_UP = "trend_up"
    SIDEWAYS = "sideways"
    TREND_DOWN = "trend_down"
    HIGH_VOL = "high_vol"

    # 各状态对应的仓位上限
    POSITION_LIMITS = {
        TREND_UP: 1.00,
        SIDEWAYS: 0.60,
        TREND_DOWN: 0.30,
        HIGH_VOL: 0.20,
    }

    def __init__(self, lookback_short: int = 20, lookback_long: int = 60,
                 vol_window: int = 20):
        """
        参数:
            lookback_short: 短期均线窗口（默认20日≈1个月）
            lookback_long: 长期均线窗口（默认60日≈3个月）
            vol_window: 波动率计算窗口（默认20日）
        """
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.vol_window = vol_window
        logger.info(
            f"MarketTimer 初始化: "
            f"短期={lookback_short}d, 长期={lookback_long}d, "
            f"波动率窗口={vol_window}d"
        )

    def get_state(self, index_df: pd.DataFrame, date: str = None) -> str:
        """
        判断指定日期的市场状态

        参数:
            index_df: 指数日线 DataFrame，需包含 'close' 列
            date: 目标日期（str 或 Timestamp），默认使用最后一天

        返回:
            市场状态字符串
        """
        if date is not None:
            idx = index_df.index.get_loc(date) if date in index_df.index else -1
            if idx == -1:
                logger.warning(f"日期 {date} 不在数据中，使用最新数据")
                df = index_df
            else:
                df = index_df.iloc[:idx + 1]
        else:
            df = index_df

        close = df["close"].values
        if len(close) < self.lookback_long + 5:
            logger.warning(f"数据不足 ({len(close)} 天)，默认返回震荡")
            return self.SIDEWAYS

        # ========== 技术指标计算 ==========

        # 1. 均线偏离度
        ma_short = np.mean(close[-self.lookback_short:])
        ma_long = np.mean(close[-self.lookback_long:])
        current_price = close[-1]
        ma_deviation = (current_price - ma_short) / ma_short  # 相对于短期均线
        ma_trend = (ma_short - ma_long) / ma_long  # 均线斜率

        # 2. 波动率
        returns = np.diff(close[-self.vol_window - 1:]) / close[-self.vol_window - 1:-1]
        volatility = np.std(returns) * np.sqrt(252)
        vol_percentile = self._calc_percentile(volatility, df, lookback=252)

        # 3. 动量强度
        momentum_20d = (close[-1] - close[-self.lookback_short]) / close[-self.lookback_short]
        momentum_60d = (close[-1] - close[-self.lookback_long]) / close[-self.lookback_long]

        # 4. 价格位置（当前价在 60 日高低点中的位置）
        high_60d = np.max(close[-self.lookback_long:])
        low_60d = np.min(close[-self.lookback_long:])
        price_position = (current_price - low_60d) / (high_60d - low_60d) if high_60d != low_60d else 0.5

        logger.debug(
            f"市场指标: MA偏离={ma_deviation:.4f}, "
            f"均线趋势={ma_trend:.4f}, "
            f"波动率={volatility:.4f}, "
            f"动量20d={momentum_20d:.4f}, "
            f"价格位置={price_position:.4f}"
        )

        # ========== 状态判断逻辑 ==========

        # 高波动检测（优先）
        if volatility > 0.35 or vol_percentile > 0.90:
            logger.info(f"市场状态: 高波动 (波动率={volatility:.2%})")
            return self.HIGH_VOL

        # 趋势检测（均线 + 动量）
        if ma_trend > 0.02 and momentum_20d > 0 and momentum_60d > 0 and price_position > 0.6:
            logger.info(f"市场状态: 趋势上涨 (均线趋势={ma_trend:.2%})")
            return self.TREND_UP

        # 下跌趋势
        if ma_trend < -0.02 and momentum_20d < 0 and price_position < 0.4:
            logger.info(f"市场状态: 趋势下跌 (均线趋势={ma_trend:.2%})")
            return self.TREND_DOWN

        # 默认：震荡
        logger.info(f"市场状态: 震荡 (波动率={volatility:.2%}, 价格位置={price_position:.2%})")
        return self.SIDEWAYS

    def get_position_limit(self, state: str = None, index_df: pd.DataFrame = None,
                           date: str = None) -> float:
        """
        获取最大仓位限制

        参数:
            state: 市场状态（如果提供，直接返回对应限制）
            index_df: 指数数据（与 date 配合使用以自动判断状态）
            date: 目标日期

        返回:
            仓位限制比例 (0-1)
        """
        if state is None and index_df is not None:
            state = self.get_state(index_df, date)

        limit = self.POSITION_LIMITS.get(state, 0.5)
        logger.info(f"市场状态: {state}, 仓位限制: {limit:.0%}")
        return limit

    def get_state_history(self, index_df: pd.DataFrame) -> pd.Series:
        """
        获取历史每一天的市场状态序列

        参数:
            index_df: 指数日线 DataFrame

        返回:
            Series: {date: state}
        """
        states = {}
        dates = index_df.index.tolist()

        for i in range(self.lookback_long, len(dates)):
            date = dates[i]
            sub_df = index_df.iloc[:i + 1]
            state = self.get_state(sub_df, date)
            states[date] = state

        return pd.Series(states, name="market_state")

    # ---------- 辅助方法 ----------

    def _calc_percentile(self, current_vol: float, df: pd.DataFrame,
                         lookback: int = 252) -> float:
        """计算当前波动率在历史中的分位数"""
        close = df["close"].values
        if len(close) < lookback + 20:
            lookback = len(close) - 20
        if lookback < 30:
            return 0.5

        all_returns = np.diff(close[-lookback - self.vol_window:]) / \
                      close[-lookback - self.vol_window - 1:-1]
        # 滚动波动率
        all_vols = []
        for i in range(len(all_returns) - self.vol_window + 1):
            v = np.std(all_returns[i:i + self.vol_window]) * np.sqrt(252)
            all_vols.append(v)
        if not all_vols:
            return 0.5
        percentile = np.sum(np.array(all_vols) < current_vol) / len(all_vols)
        return percentile
