"""
数据工具函数 — Layer 1: 数据基础设施层

提供交易日历、日期对齐、行业分类等辅助功能。
"""

from datetime import datetime
from typing import List, Optional

import akshare as ak
import pandas as pd
from loguru import logger


def get_trade_calendar(
    start_date: str = "20200101",
    end_date: str = "20251231",
) -> pd.DatetimeIndex:
    """获取 A 股交易日历。

    使用 akshare ``tool_trade_date_hist_sina`` 或 ``stock_info_trade_date_em``
    获取指定日期范围内的所有交易日。

    Parameters
    ----------
    start_date : str, optional
        起始日期（格式 ``YYYYMMDD``），默认 ``20200101``。
    end_date : str, optional
        结束日期（格式 ``YYYYMMDD``），默认 ``20251231``。

    Returns
    -------
    pd.DatetimeIndex
        交易日期的有序索引。

    Examples
    --------
    >>> cal = get_trade_calendar("20240101", "20240131")
    >>> len(cal)
    22
    """
    try:
        # 优先使用新浪交易日历（数据完整）
        df = ak.tool_trade_date_hist_sina()
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        else:
            raise ValueError("未找到 trade_date 列")
    except Exception as e:
        logger.warning("交易日历获取失败 ({}), 尝试备用接口 ...", e)
        try:
            df = ak.stock_info_trade_date_em()
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"])
            else:
                raise ValueError("备用接口也未找到 trade_date 列")
        except Exception as e2:
            logger.error("所有交易日历接口均失败: {}", e2)
            return pd.DatetimeIndex([])

    # 过滤日期范围
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    mask = (df["trade_date"] >= start) & (df["trade_date"] <= end)
    trade_dates = df.loc[mask, "trade_date"].sort_values().drop_duplicates()

    logger.info(
        "交易日历: {} ~ {} (共 {} 个交易日)",
        start_date,
        end_date,
        len(trade_dates),
    )
    return pd.DatetimeIndex(trade_dates)


def align_to_trade_dates(
    dates: pd.Series,
    trade_calendar: Optional[pd.DatetimeIndex] = None,
    direction: str = "nearest",
) -> pd.Series:
    """将任意日期对齐到最近的交易日。

    Parameters
    ----------
    dates : pd.Series
        待对齐的日期序列（``datetime`` 或 ``Timestamp``）。
    trade_calendar : pd.DatetimeIndex, optional
        交易日历，若为 ``None`` 则自动获取 2000-01-01 至 2030-12-31 的日历。
    direction : str, optional
        对齐方向。

        - ``"nearest"`` — 对齐到最近的交易日（默认）
        - ``"backward"`` — 对齐到向前最近的交易日（前一个交易日）
        - ``"forward"`` — 对齐到向后最近的交易日（后一个交易日）

    Returns
    -------
    pd.Series
        对齐后的日期序列。若无法对齐（如日历为空）则返回原序列。

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-07")])
    >>> align_to_trade_dates(s)
    0   2024-01-05
    1   2024-01-08
    dtype: datetime64[ns]
    """
    if trade_calendar is None or trade_calendar.empty:
        logger.info("未提供交易日历，自动获取 ...")
        trade_calendar = get_trade_calendar("20000101", "20301231")

    if trade_calendar.empty:
        logger.warning("align_to_trade_dates: 交易日历为空，返回原序列")
        return dates

    # 转为 Timestamp
    dates = pd.to_datetime(dates)
    cal_series = pd.Series(trade_calendar, index=trade_calendar)

    if direction == "nearest":
        # 使用 merge_asof 做最近匹配
        aligned = pd.merge_asof(
            dates.to_frame("date").sort_values("date"),
            cal_series.to_frame("trade_date").sort_index(),
            left_on="date",
            right_index=True,
            direction="nearest",
        )["trade_date"]
    elif direction == "backward":
        aligned = pd.merge_asof(
            dates.to_frame("date").sort_values("date"),
            cal_series.to_frame("trade_date").sort_index(),
            left_on="date",
            right_index=True,
            direction="backward",
        )["trade_date"]
    elif direction == "forward":
        aligned = pd.merge_asof(
            dates.to_frame("date").sort_values("date"),
            cal_series.to_frame("trade_date").sort_index(),
            left_on="date",
            right_index=True,
            direction="forward",
        )["trade_date"]
    else:
        raise ValueError(f"未知的 direction 参数: {direction}")

    # 恢复原始顺序
    aligned.index = dates.index
    return pd.Series(aligned, index=dates.index, name="trade_date")


def get_industry(stock_code: str) -> Optional[str]:
    """获取股票所属行业分类。

    使用 akshare 查询股票的申万行业分类。

    Parameters
    ----------
    stock_code : str
        股票代码，如 ``000001``。

    Returns
    -------
    Optional[str]
        行业名称。若查询失败返回 ``None``。

    Examples
    --------
    >>> get_industry("000001")
    '银行'
    >>> get_industry("600519")
    '食品饮料'
    """
    try:
        # 方法1: stock_board_industry_cons_em — 股票所属行业板块
        df = ak.stock_board_industry_cons_em(symbol=stock_code)
        if not df.empty and "板块名称" in df.columns:
            industry = df["板块名称"].iloc[0]
            logger.debug("{} -> 行业: {}", stock_code, industry)
            return str(industry)
    except Exception as e:
        logger.debug("get_industry 方法1 失败: {}", e)

    try:
        # 方法2: stock_board_industry_name_em — 全量行业列表，逐个查询成分股
        boards = ak.stock_board_industry_name_em()
        for _, row in boards.iterrows():
            board_name = row["板块名称"]
            try:
                cons = ak.stock_board_industry_cons_em(symbol=board_name)
                if stock_code in cons["代码"].values:
                    logger.debug("{} -> 行业(方法2): {}", stock_code, board_name)
                    return str(board_name)
            except Exception:
                continue
    except Exception as e:
        logger.error("get_industry 方法2 也失败: {}", e)

    logger.warning("无法获取 {} 的行业分类", stock_code)
    return None
