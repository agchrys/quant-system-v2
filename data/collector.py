"""
数据采集器 — Layer 1: 数据基础设施层

从 akshare 获取沪深 300 成分股的日线数据与财务数据，
支持本地 parquet 缓存，避免重复请求。
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional

import akshare as ak
import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm


class DataCollector:
    """A 股数据采集器。

    负责从 akshare 获取沪深 300 成分股的历史行情和财务数据，
    并以 parquet 格式缓存到本地磁盘。

    Parameters
    ----------
    config_path : str, optional
        YAML 配置文件路径，默认 ``config/config.yaml``。
    """

    def __init__(self, config_path: str = "./config/config.yaml") -> None:
        # ---- 加载配置 ----
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        data_cfg = config["data"]

        self.start_date: str = data_cfg["start_date"]
        self.end_date: str = data_cfg["end_date"]
        self.stock_universe: str = data_cfg["stock_universe"]
        self.data_source: str = data_cfg["data_source"]
        self.cache_dir: str = data_cfg["cache_dir"]
        self.financial_update: str = data_cfg.get("financial_update", "quarterly")

        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "daily"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "financial"), exist_ok=True)

        logger.info(
            "DataCollector 初始化完成 | 股票池: {} | 范围: {} ~ {}",
            self.stock_universe,
            self.start_date,
            self.end_date,
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get_stock_list(self) -> List[str]:
        """获取当前股票池的成分股代码列表。

        Returns
        -------
        List[str]
            股票代码列表，如 ``['000001', '000002', ...]``。
        """
        logger.info("正在获取「{}」成分股列表 ...", self.stock_universe)

        if self.stock_universe == "沪深300":
            # stock_zh_a_spot_em 返回全 A 实时行情，可从中筛选沪深 300
            df = ak.stock_zh_a_spot_em()
            # 尝试按沪深 300 概念板块获取；若失败则降级
            try:
                df_csi = ak.stock_board_cons_index_em(symbol="沪深300")
                codes = sorted(df_csi["代码"].tolist())
                codes = [c.strip().lstrip("'") for c in codes]
                logger.info("共获取 {} 只沪深 300 成分股", len(codes))
                return codes
            except Exception:
                pass
        elif self.stock_universe == "中证500":
            try:
                df_csi = ak.stock_board_cons_index_em(symbol="中证500")
                codes = sorted(df_csi["代码"].tolist())
                codes = [c.strip().lstrip("'") for c in codes]
                logger.info("共获取 {} 只中证 500 成分股", len(codes))
                return codes
            except Exception:
                pass

        # 降级方案：从全 A 股票中筛选
        df = ak.stock_zh_a_spot_em()
        codes = sorted(df["代码"].tolist())
        codes = [c.strip().lstrip("'") for c in codes]
        logger.warning("未能获取板块成分，降级为全 A 股票（共 {} 只）", len(codes))
        return codes

    def fetch_daily(self, stock_code: str) -> pd.DataFrame:
        """获取个股日线行情数据。

        Parameters
        ----------
        stock_code : str
            股票代码，如 ``000001``。

        Returns
        -------
        pd.DataFrame
            包含 ``开盘、收盘、最高、最低、成交量、成交额、换手率`` 等字段的 DataFrame。
        """
        cache_path = os.path.join(self.cache_dir, "daily", f"{stock_code}.parquet")
        cached = self._check_cache(cache_path)
        if cached is not None:
            return cached

        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=self.start_date.replace("-", ""),
                end_date=self.end_date.replace("-", ""),
                adjust="",
            )
            if df.empty:
                logger.warning("{} 日线数据为空", stock_code)
                return pd.DataFrame()

            # 统一列名（中文 -> 英文）
            col_map = {
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_chg",
                "涨跌额": "change",
                "换手率": "turnover",
            }
            df.rename(columns=col_map, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            df["stock_code"] = stock_code

            # 缓存
            df.to_parquet(cache_path, index=False)
            logger.debug("已缓存 {} 日线数据 -> {}", stock_code, cache_path)
            return df

        except Exception as e:
            logger.error("获取 {} 日线数据失败: {}", stock_code, e)
            return pd.DataFrame()

    def fetch_financial(self, stock_code: str) -> pd.DataFrame:
        """获取个股财务概要数据。

        Parameters
        ----------
        stock_code : str
            股票代码，如 ``000001``。

        Returns
        -------
        pd.DataFrame
            包含营收、净利润、ROE、PE、PB 等字段的 DataFrame。
        """
        cache_path = os.path.join(self.cache_dir, "financial", f"{stock_code}.parquet")
        cached = self._check_cache(cache_path)
        if cached is not None:
            return cached

        try:
            # 优先使用 stock_financial_abstract_ths
            df = ak.stock_financial_abstract_ths(symbol=stock_code)
            if df.empty:
                logger.warning("{} 财务数据为空", stock_code)
                return pd.DataFrame()

            df["stock_code"] = stock_code
            if "日期" in df.columns:
                df.rename(columns={"日期": "date"}, inplace=True)
            if "报告期" in df.columns:
                df.rename(columns={"报告期": "date"}, inplace=True)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

            df.to_parquet(cache_path, index=False)
            logger.debug("已缓存 {} 财务数据 -> {}", stock_code, cache_path)
            return df

        except Exception as e:
            logger.error("获取 {} 财务数据失败: {}", stock_code, e)
            return pd.DataFrame()

    def collect_all(self) -> None:
        """顺序采集所有股票的数据。

        先获取股票列表，再逐只采集日线数据与财务数据，
        使用 tqdm 显示进度条。已缓存且够新的数据会跳过。
        """
        codes = self.get_stock_list()
        logger.info("开始批量采集，共 {} 只股票", len(codes))

        # ---- 日线数据 ----
        logger.info("--- 采集日线数据 ---")
        daily_list: List[pd.DataFrame] = []
        for code in tqdm(codes, desc="日线数据"):
            df = self.fetch_daily(code)
            if not df.empty:
                daily_list.append(df)

        if daily_list:
            all_daily = pd.concat(daily_list, ignore_index=True)
            daily_path = os.path.join(self.cache_dir, "all_daily.parquet")
            all_daily.to_parquet(daily_path, index=False)
            logger.info(
                "日线数据汇总保存完毕 | 路径: {} | 记录数: {}",
                daily_path,
                len(all_daily),
            )

        # ---- 财务数据 ----
        logger.info("--- 采集财务数据 ---")
        fin_list: List[pd.DataFrame] = []
        for code in tqdm(codes, desc="财务数据"):
            df = self.fetch_financial(code)
            if not df.empty:
                fin_list.append(df)

        if fin_list:
            all_fin = pd.concat(fin_list, ignore_index=True)
            fin_path = os.path.join(self.cache_dir, "all_financial.parquet")
            all_fin.to_parquet(fin_path, index=False)
            logger.info(
                "财务数据汇总保存完毕 | 路径: {} | 记录数: {}",
                fin_path,
                len(all_fin),
            )

        logger.info("批量采集完成 | 日线: {} 只 | 财务: {} 只", len(daily_list), len(fin_list))

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _check_cache(
        self, cache_path: str, max_age_days: int = 1
    ) -> Optional[pd.DataFrame]:
        """检查本地 parquet 缓存是否有效。

        Parameters
        ----------
        cache_path : str
            缓存文件路径。
        max_age_days : int, optional
            缓存最大有效期（天），默认 1。

        Returns
        -------
        Optional[pd.DataFrame]
            如果缓存有效返回 DataFrame，否则返回 None。
        """
        if not os.path.exists(cache_path):
            return None
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - mtime > timedelta(days=max_age_days):
            logger.debug("缓存已过期: {} ({} 前创建)", cache_path, datetime.now() - mtime)
            return None
        try:
            df = pd.read_parquet(cache_path)
            logger.debug("使用缓存: {}", cache_path)
            return df
        except Exception as e:
            logger.warning("缓存文件损坏: {} ({})", cache_path, e)
            return None
