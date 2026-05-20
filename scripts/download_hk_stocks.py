"""
港股数据下载脚本
从 akshare 获取最近十年（2016-2026）港股日线数据，缓存到 data/hk_cache/ 目录。

用法:
    python3 scripts/download_hk_stocks.py           # 自动获取列表（优先 API，失败降级用内置列表）
    python3 scripts/download_hk_stocks.py --hsi     # 仅下载恒生指数成分股
    python3 scripts/download_hk_stocks.py --all     # 下载全部港股（数量较多，需网络稳定）
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Optional

import akshare as ak
import pandas as pd
from loguru import logger
from tqdm import tqdm

# ---- 配置 ----
HK_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hk_cache")
START_DATE = "20160520"  # 最近十年
END_DATE = "20260520"
RETRY_TIMES = 3
RETRY_SLEEP = 3

# ---- 内置备选股票列表（恒生指数 + 恒生科技 + 其他知名港股） ----
# 当 akshare 列表接口不可用时使用
FALLBACK_HK_STOCKS = [
    # 恒生指数主要成分股（蓝筹）
    "00001", "00002", "00003", "00005", "00006", "00011", "00012", "00016",
    "00017", "00019", "00027", "00066", "00083", "00101", "00175", "00241",
    "00267", "00288", "00291", "00316", "00322", "00386", "00388", "00669",
    "00688", "00700", "00762", "00823", "00857", "00868", "00883", "00939",
    "00941", "00981", "00992", "01038", "01044", "01088", "01093", "01109",
    "01113", "01177", "01209", "01299", "01378", "01398", "01810", "01876",
    "01928", "01929", "01997", "02007", "02015", "02018", "02020", "02269",
    "02313", "02318", "02319", "02331", "02382", "02388", "02628", "02688",
    "02899", "03690", "03968", "03988", "09618", "09633", "09888", "09988",
    "09992", "09999",
    # 恒生科技指数
    "00020", "00268", "00285", "00909", "01024", "01347",
    "01833", "02013", "02382", "02491", "03779", "03888",
    "06060", "06618", "09626", "09698", "09888", "09899", "09961",
    # 其他知名港股
    "00168", "00257", "00270", "00371", "00522", "00670",
    "00753", "00772", "00780", "00788", "00813", "00836",
    "00914", "00916", "00960", "00968", "01038", "01066",
    "01128", "01308", "01339", "01448", "01458", "01579",
    "01658", "01772", "01787", "01821", "01918", "01951",
    "02039", "02196", "02282", "02333", "02601", "03328",
    "03333", "03380", "03692", "03899", "06030", "06098",
    "06862", "09668",
]


def retry_api(func, name: str, times: int = RETRY_TIMES, sleep_s: int = RETRY_SLEEP):
    """带重试的 API 调用。"""
    for attempt in range(1, times + 1):
        try:
            return func()
        except Exception as e:
            if attempt < times:
                logger.warning("{} 失败 (第 {}/{} 次): {}，{} 秒后重试 ...", name, attempt, times, e, sleep_s)
                time.sleep(sleep_s)
            else:
                logger.error("{} 最终失败: {}", name, e)
                raise


class HKDataCollector:
    """港股数据采集器。"""

    def __init__(self, cache_dir: str = HK_CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "daily"), exist_ok=True)
        logger.info("港股缓存目录: {}", cache_dir)

    # ------------------------------------------------------------------
    # 获取股票列表
    # ------------------------------------------------------------------

    def get_hk_stock_list(self, prefer: str = "auto") -> List[str]:
        """获取港股代码列表。

        Parameters
        ----------
        prefer : str
            "auto"  - 优先 API 获取，失败降级为内置列表
            "hsi"   - 仅恒生蓝筹 + 恒生科技内置列表
            "all"   - 尝试从 API 获取全港股，失败用内置列表

        Returns
        -------
        List[str]
        """
        if prefer == "hsi":
            logger.info("使用内置恒生+科技成分股列表: {} 只", len(FALLBACK_HK_STOCKS))
            return FALLBACK_HK_STOCKS

        # 策略 1：港股通成分股（约 500+ 只，覆盖面广且都是可交易标的）
        try:
            logger.info("正在通过港股通接口获取成分股列表 ...")
            df = retry_api(
                lambda: ak.stock_hk_ggt_components_em(),
                name="stock_hk_ggt_components_em",
            )
            if "代码" in df.columns:
                codes = df["代码"].astype(str).tolist()
                codes = sorted([c.zfill(5) for c in codes])
                logger.info("港股通成分股: {} 只", len(codes))
                return codes
        except Exception as e:
            logger.warning("港股通接口失败: {}", e)

        # 策略 2：全港股实时行情（约 2500+ 只）
        if prefer == "all":
            try:
                logger.info("正在通过实时行情接口获取全港股列表 ...")
                df = retry_api(
                    lambda: ak.stock_hk_spot_em(),
                    name="stock_hk_spot_em",
                )
                if "代码" in df.columns:
                    codes = df["代码"].astype(str).tolist()
                    codes = sorted([c.zfill(5) for c in codes])
                    logger.info("全港股: {} 只", len(codes))
                    return codes
            except Exception as e:
                logger.warning("全港股接口失败: {}", e)

        # 策略 3：热门港股
        try:
            logger.info("正在获取热门港股列表 ...")
            df = retry_api(
                lambda: ak.stock_hk_famous_spot_em(),
                name="stock_hk_famous_spot_em",
                times=2,  # 少重试，快速失败
            )
            if "代码" in df.columns:
                codes = df["代码"].astype(str).tolist()
                codes = sorted([c.zfill(5) for c in codes])
                logger.info("热门港股: {} 只", len(codes))
                return codes
        except Exception:
            pass

        # 最终降级：内置列表
        logger.warning("所有 API 获取均失败，降级为内置列表: {} 只", len(FALLBACK_HK_STOCKS))
        return FALLBACK_HK_STOCKS

    # ------------------------------------------------------------------
    # 下载日线数据
    # ------------------------------------------------------------------

    def fetch_daily(self, stock_code: str) -> pd.DataFrame:
        """下载单只港股日线数据（带重试）。

        Parameters
        ----------
        stock_code : str
            5 位港股代码，如 ``00700``。

        Returns
        -------
        pd.DataFrame
        """
        cache_path = os.path.join(self.cache_dir, "daily", f"{stock_code}.parquet")
        cached = self._check_cache(cache_path)
        if cached is not None:
            return cached

        try:
            df = retry_api(
                lambda: ak.stock_hk_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=START_DATE,
                    end_date=END_DATE,
                    adjust="",
                ),
                name=f"stock_hk_hist({stock_code})",
            )

            if df is None or df.empty:
                logger.warning("{} 港股日线数据为空", stock_code)
                return pd.DataFrame()

            # 统一列名（中文 → 英文）
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
            rename_dict = {k: v for k, v in col_map.items() if k in df.columns}
            df.rename(columns=rename_dict, inplace=True)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.sort_values("date", inplace=True)

            df.reset_index(drop=True, inplace=True)
            df["stock_code"] = stock_code

            # 写入本地缓存
            df.to_parquet(cache_path, index=False)
            return df

        except Exception as e:
            logger.error("获取港股 {} 日线最终失败: {}", stock_code, e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 全量下载
    # ------------------------------------------------------------------

    def collect_all(self, prefer: str = "auto") -> None:
        """遍历港股并下载日线数据，汇总保存。

        Parameters
        ----------
        prefer : str
            "auto" / "hsi" / "all"
        """
        codes = self.get_hk_stock_list(prefer=prefer)
        if not codes:
            logger.error("未获取到任何港股代码，终止下载")
            return

        logger.info(
            "开始下载港股日线 | 股票: {}/只 | 范围: {} ~ {}",
            len(codes), START_DATE, END_DATE,
        )

        daily_list = []
        success_count = 0

        for code in tqdm(codes, desc="港股日线"):
            df = self.fetch_daily(code)
            if not df.empty:
                daily_list.append(df)
                success_count += 1
            # 东方财富反向爬策略较严，建议间隔 0.2~0.5s
            time.sleep(0.3)

        # 汇总保存
        if daily_list:
            all_daily = pd.concat(daily_list, ignore_index=True)
            summary_path = os.path.join(self.cache_dir, "hk_all_daily.parquet")
            all_daily.to_parquet(summary_path, index=False)
            logger.info(
                "港股日线汇总完成 | 路径: {} | 记录数: {} | 成功: {} 只",
                summary_path, len(all_daily), success_count,
            )
        else:
            logger.error("未采集到任何港股日线数据")

        # 保存股票列表
        stock_list_path = os.path.join(self.cache_dir, "hk_stock_list.csv")
        pd.DataFrame({"stock_code": codes}).to_csv(stock_list_path, index=False)
        logger.info("港股代码列表已保存 → {}", stock_list_path)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _check_cache(
        self, cache_path: str, max_age_days: int = 1
    ) -> Optional[pd.DataFrame]:
        """检查缓存有效性。"""
        if not os.path.exists(cache_path):
            return None
        if abs(datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))) > timedelta(days=max_age_days):
            return None
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            return None


# ======================================================================
# 主入口
# ======================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="下载港股最近十年日线数据")
    parser.add_argument(
        "--mode", choices=["auto", "hsi", "all"], default="auto",
        help="auto=优先API获列表,失败降级内置列表; hsi=仅内置蓝筹; all=尝试获取全部港股",
    )
    args = parser.parse_args()

    collector = HKDataCollector()
    collector.collect_all(prefer=args.mode)
