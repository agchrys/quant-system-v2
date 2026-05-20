"""
数据处理模块 — Layer 1: 数据基础设施层

对原始采集数据进行复权、去缺失、对齐、宽表构建等清洗加工，
最终产出统一的 MultiIndex 面板数据。
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class DataProcessor:
    """数据处理器。

    接收 DataCollector 采集的原始数据，完成从清洗到最终宽表的全流程加工。

    Parameters
    ----------
    daily_data : pd.DataFrame, optional
        原始日线数据，包含 ``date, stock_code, open, close, high, low, volume, ...``。
    financial_data : pd.DataFrame, optional
        原始财务数据。
    """

    def __init__(
        self,
        daily_data: Optional[pd.DataFrame] = None,
        financial_data: Optional[pd.DataFrame] = None,
    ) -> None:
        self.daily_data = daily_data
        self.financial_data = financial_data
        logger.info("DataProcessor 初始化完成")

    # ------------------------------------------------------------------
    # 复权处理
    # ------------------------------------------------------------------

    def adjust_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """对日线数据进行前复权处理。

        利用 akshare ``stock_zh_a_hist`` 的 ``adjust="qfq"`` 参数
        重新获取前复权数据，替换原始价格列。

        Parameters
        ----------
        df : pd.DataFrame
            原始日线数据（需含 ``stock_code`` 列）。

        Returns
        -------
        pd.DataFrame
            复权处理后的数据。
        """
        if df.empty or "stock_code" not in df.columns:
            logger.warning("adjust_price: 输入数据为空或缺少 stock_code 列")
            return df

        # 已存在于缓存中的日线数据可能未复权，重新获取复权版本
        import akshare as ak

        stock_code = df["stock_code"].iloc[0]
        start = df["date"].min().strftime("%Y%m%d")
        end = df["date"].max().strftime("%Y%m%d")

        try:
            adj = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq",  # 前复权
            )
            if adj.empty:
                logger.warning("{} 复权数据为空，保持原值", stock_code)
                return df

            col_map = {
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
            }
            adj.rename(columns=col_map, inplace=True)
            adj["date"] = pd.to_datetime(adj["date"])

            # 按日期合并复权价格
            price_cols = ["open", "close", "high", "low"]
            for col in price_cols:
                if col in adj.columns:
                    price_map = adj.set_index("date")[col].to_dict()
                    df[col] = df["date"].map(price_map).fillna(df[col])

            logger.debug("{} 前复权处理完成", stock_code)
        except Exception as e:
            logger.error("{} 复权失败: {}", stock_code, e)

        return df

    # ------------------------------------------------------------------
    # 缺失值处理
    # ------------------------------------------------------------------

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """中位数填充缺失值。

        仅对数值型列进行填充，非数值列保持不变。

        Parameters
        ----------
        df : pd.DataFrame
            输入数据。

        Returns
        -------
        pd.DataFrame
            缺失值填充后的数据。
        """
        if df.empty:
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        before = df[numeric_cols].isna().sum().sum()
        for col in numeric_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        after = df[numeric_cols].isna().sum().sum()
        logger.info("handle_missing: 填充 {} 个缺失值（填充前 {}）", before - after, before)
        return df

    # ------------------------------------------------------------------
    # 时间戳对齐
    # ------------------------------------------------------------------

    def align_timestamps(
        self, df_list: List[pd.DataFrame]
    ) -> List[pd.DataFrame]:
        """将多只股票的日线数据对齐到同一组交易日。

        取所有股票日期的交集作为统一交易日历，
        不在日历中的股票日期将被移除。

        Parameters
        ----------
        df_list : List[pd.DataFrame]
            多个股票的日线 DataFrame 列表。

        Returns
        -------
        List[pd.DataFrame]
            对齐后的 DataFrame 列表。
        """
        if not df_list:
            return df_list

        # 求所有股票日期的交集（取最严格的对齐）
        date_sets = [set(df["date"].unique()) for df in df_list if not df.empty]
        if not date_sets:
            return df_list

        common_dates = sorted(set.intersection(*date_sets))
        if not common_dates:
            logger.warning("align_timestamps: 无共同交易日，跳过对齐")
            return df_list

        aligned = []
        for df in df_list:
            if df.empty:
                continue
            aligned.append(df[df["date"].isin(common_dates)].copy())

        logger.info("align_timestamps: 对齐至 {} 个共同交易日", len(common_dates))
        return aligned

    # ------------------------------------------------------------------
    # 构建宽表
    # ------------------------------------------------------------------

    def build_panel(
        self,
        factor_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """构建统一的 MultiIndex 宽表面板数据。

        以 ``(日期, 股票代码)`` 为行索引，每列为一个因子/特征。

        Parameters
        ----------
        factor_data : pd.DataFrame
            必须包含 ``date`` 和 ``stock_code`` 列，其余为数值特征。

        Returns
        -------
        pd.DataFrame
            MultiIndex ``(date, stock_code)`` 的宽表。
        """
        required = {"date", "stock_code"}
        if not required.issubset(factor_data.columns):
            raise ValueError(
                f"factor_data 必须包含 {required} 列，当前列: {list(factor_data.columns)}"
            )

        df = factor_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.set_index(["date", "stock_code"], inplace=True)
        df.sort_index(inplace=True)

        # 去重：相同 (date, stock_code) 保留最后一条
        df = df[~df.index.duplicated(keep="last")]

        logger.info(
            "build_panel: 面板形状 {} | 行数 {} | 列数 {}",
            list(df.index.names),
            len(df),
            len(df.columns),
        )
        return df

    # ------------------------------------------------------------------
    # 停牌处理
    # ------------------------------------------------------------------

    def handle_suspension(self, df: pd.DataFrame) -> pd.DataFrame:
        """对停牌日的缺失数据向前填充（ffill）。

        对 MultiIndex 面板数据按每只股票分别进行向前填充。

        Parameters
        ----------
        df : pd.DataFrame
            MultiIndex 面板数据（需要包含 ``stock_code`` 在索引或列中）。

        Returns
        -------
        pd.DataFrame
            停牌日数据填充后的 DataFrame。
        """
        if df.empty:
            return df

        # 如果是 MultiIndex (date, stock_code) 形式
        if isinstance(df.index, pd.MultiIndex) and "stock_code" in df.index.names:
            # 按股票代码分组，对每组进行 ffill
            result = df.groupby(level="stock_code", group_keys=False).apply(
                lambda g: g.ffill()
            )
            logger.info("handle_suspension: 停牌向前填充完成")
            return result

        # 如果是宽表形式，按 stock_code 列分组
        if "stock_code" in df.columns:
            result = df.sort_values(["stock_code", "date"]).groupby("stock_code").apply(
                lambda g: g.ffill()
            )
            result.reset_index(drop=True, inplace=True)
            logger.info("handle_suspension: 停牌向前填充完成")
            return result

        logger.warning("handle_suspension: 无法识别数据格式，跳过")
        return df

    # ------------------------------------------------------------------
    # 全流程处理
    # ------------------------------------------------------------------

    def process_pipeline(self) -> pd.DataFrame:
        """全流程数据处理流水线。

        依次执行：复权 → 缺失值填充 → 时间对齐 → 宽表构建 → 停牌处理。

        Returns
        -------
        pd.DataFrame
            最终面板数据。
        """
        if self.daily_data is None or self.daily_data.empty:
            logger.error("process_pipeline: 缺少日线数据")
            return pd.DataFrame()

        df = self.daily_data.copy()

        # 1. 复权处理（按股票分组）
        logger.info("步骤 1/5: 复权处理 ...")
        codes = df["stock_code"].unique()
        adjusted_list = []
        for code in codes:
            sub = df[df["stock_code"] == code].copy()
            adjusted_list.append(self.adjust_price(sub))
        df = pd.concat(adjusted_list, ignore_index=True)

        # 2. 缺失值填充
        logger.info("步骤 2/5: 缺失值填充 ...")
        df = self.handle_missing(df)

        # 3. 时间戳对齐
        logger.info("步骤 3/5: 时间对齐 ...")
        code_dfs = [df[df["stock_code"] == c].copy() for c in codes]
        code_dfs = self.align_timestamps(code_dfs)
        df = pd.concat(code_dfs, ignore_index=True) if code_dfs else df

        # 4. 构建宽表面板
        logger.info("步骤 4/5: 构建宽表 ...")
        # 合并财务数据
        if self.financial_data is not None and not self.financial_data.empty:
            fin = self.financial_data.copy()
            fin["date"] = pd.to_datetime(fin["date"]) if "date" in fin.columns else None
            if "date" in fin.columns:
                # 按日期向前合并：将最新财务数据匹配到最近的交易日
                fin = fin.sort_values("date")
                df = pd.merge_asof(
                    df.sort_values("date"),
                    fin.sort_values("date"),
                    on="date",
                    by="stock_code",
                    direction="backward",
                )
            else:
                df = df.merge(fin, on="stock_code", how="left")

        panel = self.build_panel(df)

        # 5. 停牌处理
        logger.info("步骤 5/5: 停牌处理 ...")
        panel = self.handle_suspension(panel)

        logger.info("全流程处理完成 | 面板形状: {} × {}", len(panel), len(panel.columns))
        return panel

    # ------------------------------------------------------------------
    # 保存 / 加载
    # ------------------------------------------------------------------

    def save_panel(self, df: pd.DataFrame, path: str) -> None:
        """将面板数据保存为 parquet 格式。

        Parameters
        ----------
        df : pd.DataFrame
            面板数据。
        path : str
            保存路径。
        """
        if df.empty:
            logger.warning("save_panel: 面板数据为空，跳过保存")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)
        logger.info("面板数据已保存 -> {} ({} 行)", path, len(df))

    def load_panel(self, path: str) -> pd.DataFrame:
        """从 parquet 文件加载面板数据。

        Parameters
        ----------
        path : str
            parquet 文件路径。

        Returns
        -------
        pd.DataFrame
            加载的面板数据。
        """
        if not os.path.exists(path):
            logger.error("load_panel: 文件不存在 {}", path)
            return pd.DataFrame()
        df = pd.read_parquet(path)
        logger.info("面板数据已加载 <- {} ({} 行)", path, len(df))
        return df
