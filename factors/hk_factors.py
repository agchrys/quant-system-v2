"""
Layer 2: 增强因子工程 — 港股专用因子集 (30+ 因子)

在 traditional_factors.py 基础上扩展，增加港股特色因子：
- 流动性溢价 (liquidity premium)
- 截面动量 (cross-sectional momentum)
- 波动率偏度 (volatility skew)
- 量价背离 (price-volume divergence)
- 北向资金代理 (turnover anomaly)
"""

import pandas as pd
import numpy as np


def compute_hk_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    为单只港股 DataFrame 计算增强因子集。

    Parameters
    ----------
    df : pd.DataFrame, index=date
        必须包含: open, high, low, close, volume, amount, turnover

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    close = df['close']
    volume = df['volume'].replace(0, np.nan)
    turnover = df.get('turnover', pd.Series(np.nan, index=df.index))

    # === 1. 动量族 (6因子) ===
    for n in [5, 10, 20, 60, 120]:
        df[f'mom_{n}d'] = close.pct_change(n)
    df['mom_accel'] = df['mom_20d'] - df['mom_60d']  # 动量加速

    # === 2. 波动率族 (5因子) ===
    returns = close.pct_change()
    for n in [10, 20, 60]:
        df[f'vol_{n}d'] = returns.rolling(n).std()
    df['vol_skew'] = returns.rolling(20).skew()  # 波动率偏度（上涨偏 vs 下跌偏）
    df['vol_ratio'] = df['vol_10d'] / (df['vol_60d'] + 1e-10)  # 短/长波动率比

    # === 3. 均线偏离族 (4因子) ===
    for n in [20, 60, 120]:
        ma = close.rolling(n).mean()
        df[f'ma_dev_{n}d'] = close / ma - 1
    df['ma_bull_align'] = (
        (close.rolling(5).mean() > close.rolling(10).mean()) &
        (close.rolling(10).mean() > close.rolling(20).mean()) &
        (close > close.rolling(20).mean())
    ).astype(float)

    # === 4. 成交量族 (5因子) ===
    for n in [5, 20]:
        vol_ma = volume.rolling(n).mean()
        df[f'vol_chg_{n}d'] = volume.pct_change(n)
        df[f'vol_ratio_{n}d'] = volume / (vol_ma + 1)
    df['vol_trend'] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1) - 1
    df['vol_divergence'] = (
        (close.pct_change(10) > 0) & (df['vol_chg_20d'] < 0)
    ).astype(float) * (-1)  # 价涨量缩 = 负面

    # === 5. 技术指标族 (6因子) ===
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi_14'] = 100 - 100 / (1 + gain / (loss + 1e-10))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = (ema12 - ema26) / close
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean() / close

    # KDJ
    low9, high9 = df['low'].rolling(9).min(), df['high'].rolling(9).max()
    rsv = (close - low9) / (high9 - low9 + 1e-10) * 100
    df['kdj_k'] = rsv.ewm(com=2, adjust=False).mean() / 100
    df['kdj_d'] = df['kdj_k'].ewm(com=2, adjust=False).mean()

    # === 6. 价格形态族 (4因子) ===
    for n in [20, 60]:
        high_n = df['high'].rolling(n).max()
        low_n = df['low'].rolling(n).min()
        df[f'price_pos_{n}d'] = (close - low_n) / (high_n - low_n + 1e-10)

    df['ret_consistency'] = (
        (returns > 0).rolling(10).sum() / 10
    )  # 近10日上涨天数比例

    df['hl_ratio'] = (df['high'] - df['low']) / close  # 日内振幅

    # === 7. 风险调整族 (3因子) ===
    for n in [20, 60]:
        max_dd = (close / close.rolling(n).max() - 1).rolling(n).min()
        df[f'max_dd_{n}d'] = max_dd

    df['sharpe_60d'] = returns.rolling(60).mean() / (returns.rolling(60).std() + 1e-10)

    # === 8. 流动性族 (2因子) ===
    df['turnover_ma20'] = turnover.rolling(20).mean() if not turnover.isna().all() else np.nan
    df['turnover_chg'] = turnover / (turnover.rolling(20).mean() + 1e-10) - 1 if not turnover.isna().all() else np.nan

    # Amihud 非流动性
    df['amihud'] = (abs(returns) / (df['amount'].replace(0, np.nan) + 1e-10)).rolling(20).mean()

    # === 9. 截面排名代理 (2因子) ===
    # 这些需要跨股票计算，此处做占位标记，在训练时做截面处理
    # 近20日收益相对排名（标记，实际在截面处理中计算）
    df['return_streak'] = ((returns > 0).astype(int) * 2 - 1).rolling(5).mean()

    # 清理 NaN（前向填充 + 0 填充）
    df = df.ffill().fillna(0)

    return df
