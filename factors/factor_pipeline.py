"""
因子处理流水线模块

提供 FactorPipeline 类，对因子数据进行标准化处理:
    1. 去极值 (winsorize)
    2. 标准化 (standardize)
    3. 中性化 (neutralize)
    4. 缺失值填充 (fillna)
    5. 完整流水线 (run)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


class FactorPipeline:
    """因子处理流水线: 去极值 -> 标准化 -> 中性化 -> 填充缺失值。

    Parameters
    ----------
    config : dict or str, optional
        配置字典或 yaml 文件路径。为 None 时使用默认配置。
    """

    DEFAULT_CONFIG: Dict = {
        "winsorize": {
            "enabled": True,
            "method": "mad",
            "std": 5,
        },
        "standardize": {
            "enabled": True,
            "method": "zscore",
        },
        "neutralize": {
            "enabled": True,
        },
        "fillna": {
            "enabled": True,
            "method": "median",
        },
    }

    def __init__(self, config: Optional[Union[Dict, str]] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config is not None:
            if isinstance(config, str):
                self._load_config(config)
            elif isinstance(config, dict):
                self._merge_config(config)

    def _load_config(self, path: str) -> None:
        """从 yaml 文件加载配置。"""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "读取 yaml 配置需要安装 PyYAML: pip install pyyaml"
            )

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)

        pipeline_config = yaml_config.get("factor_pipeline", yaml_config)
        if isinstance(pipeline_config, dict):
            self._merge_config(pipeline_config)

    def _merge_config(self, user_config: Dict) -> None:
        """将用户配置合并到默认配置中。"""
        for key in self.DEFAULT_CONFIG:
            if key in user_config:
                if isinstance(self.DEFAULT_CONFIG[key], dict) and isinstance(user_config[key], dict):
                    self.config[key].update(user_config[key])
                else:
                    self.config[key] = user_config[key]

    # ------------------------------------------------------------------
    # 去极值
    # ------------------------------------------------------------------

    @staticmethod
    def winsorize(factor_df: pd.DataFrame,
                  method: str = "mad",
                  std: float = 5.0,
                  lower: float = 0.01,
                  upper: float = 0.99) -> pd.DataFrame:
        """对因子去极值。

        Parameters
        ----------
        factor_df : pd.DataFrame
            因子数据, index=date, columns=stock_code
        method : str, default='mad'
            去极值方法:
            - 'mad': 基于中位数绝对偏差, 将超出 median ± std * MAD 的值截断
            - 'percentile': 百分位截断, 将超出 [lower, upper] 分位数的值截断
        std : float, default=5.0
            MAD 倍数 (仅 method='mad' 时使用)
        lower : float, default=0.01
            下分位数 (仅 method='percentile' 时使用)
        upper : float, default=0.99
            上分位数 (仅 method='percentile' 时使用)

        Returns
        -------
        pd.DataFrame
            去极值后的因子数据
        """
        df = factor_df.copy()

        if method == "mad":
            for col in df.columns:
                series = df[col]
                median = series.median()
                # MAD = median(|x - median|)
                mad = (series - median).abs().median()
                if mad == 0 or pd.isna(mad):
                    continue
                lower_bound = median - std * mad
                upper_bound = median + std * mad
                df[col] = series.clip(lower_bound, upper_bound)

        elif method == "percentile":
            for col in df.columns:
                series = df[col]
                lo = series.quantile(lower)
                hi = series.quantile(upper)
                df[col] = series.clip(lo, hi)

        else:
            raise ValueError(f"未知去极值方法: {method}, 可选: 'mad', 'percentile'")

        return df

    # ------------------------------------------------------------------
    # 标准化
    # ------------------------------------------------------------------

    @staticmethod
    def standardize(factor_df: pd.DataFrame,
                    method: str = "zscore") -> pd.DataFrame:
        """对因子做截面标准化。

        Parameters
        ----------
        factor_df : pd.DataFrame
            因子数据, index=date, columns=stock_code
        method : str, default='zscore'
            标准化方法:
            - 'zscore': (x - mean) / std
            - 'rank':   cross-sectional rank, 映射到 [-1, 1]

        Returns
        -------
        pd.DataFrame
            标准化后的因子数据
        """
        df = factor_df.copy()

        if method == "zscore":
            for col in df.columns:
                series = df[col]
                mean = series.mean()
                std_val = series.std(ddof=0)
                if std_val == 0 or pd.isna(std_val):
                    df[col] = 0.0
                else:
                    df[col] = (series - mean) / std_val

        elif method == "rank":
            for col in df.columns:
                ranks = df[col].rank(method="average")
                n = ranks.count()
                if n == 0:
                    continue
                # 映射到 [-1, 1]
                df[col] = (ranks / (n + 1)) * 2 - 1

        else:
            raise ValueError(f"未知标准化方法: {method}, 可选: 'zscore', 'rank'")

        return df

    # ------------------------------------------------------------------
    # 中性化
    # ------------------------------------------------------------------

    @staticmethod
    def neutralize(factor_df: pd.DataFrame,
                   industry_map: pd.Series,
                   market_cap: pd.DataFrame) -> pd.DataFrame:
        """对行业和市值做回归中性化。

        对每个截面的每个因子, 用 OLS 回归剔除行业哑变量与对数市值的影响,
        取残差作为中性化后的因子值。

        Parameters
        ----------
        factor_df : pd.DataFrame
            因子数据, index=date, columns=stock_code
        industry_map : pd.Series
            行业映射, index=stock_code, values=行业分类 (如申万一级行业代码)
        market_cap : pd.DataFrame
            市值数据, index=date, columns=stock_code (单位: 元)

        Returns
        -------
        pd.DataFrame
            中性化后的因子数据, 结构同 factor_df
        """
        df = factor_df.copy()

        for date in df.index:
            # 当前日期的截面数据
            factors_t = df.loc[date]
            valid_mask = factors_t.notna()
            valid_codes = factors_t.index[valid_mask]

            if len(valid_codes) < 20:
                # 样本太少, 不做中性化
                continue

            y = factors_t[valid_codes]

            # --- 构建行业哑变量 ---
            ind_map_t = industry_map.reindex(valid_codes)
            dummy = pd.get_dummies(ind_map_t, prefix="ind", drop_first=True)

            # --- 对数市值 ---
            cap_t = market_cap.loc[date, valid_codes]
            log_cap = np.log(cap_t.clip(lower=1))
            log_cap.name = "log_market_cap"

            # --- 合并自变量 ---
            X = pd.concat([dummy, log_cap], axis=1)
            X = X.apply(pd.to_numeric, errors="coerce")
            X = X.fillna(0.0)

            # --- OLS 回归 ---
            try:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression(fit_intercept=False)
                model.fit(X, y)
                residuals = y - model.predict(X)
            except ImportError:
                # 无 sklearn 时用 numpy 最小二乘
                X_mat = X.values.astype(float)
                y_vec = y.values.astype(float)
                try:
                    coeff, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
                    residuals_vec = y_vec - X_mat @ coeff
                    residuals = pd.Series(residuals_vec, index=valid_codes)
                except np.linalg.LinAlgError:
                    continue

            df.loc[date, valid_codes] = residuals

        return df

    # ------------------------------------------------------------------
    # 缺失值处理
    # ------------------------------------------------------------------

    @staticmethod
    def fillna(factor_df: pd.DataFrame,
               method: str = "median",
               fill_value: float = 0.0) -> pd.DataFrame:
        """填充因子中的缺失值。

        Parameters
        ----------
        factor_df : pd.DataFrame
            因子数据, index=date, columns=stock_code
        method : str, default='median'
            填充方法:
            - 'median': 用截面中位数填充
            - 'mean':   用截面均值填充
            - 'zero':   用 0 填充
            - 'ffill':  用前值填充
            - 'value':  用 fill_value 填充
            - 'drop':   丢弃含 NaN 的列
        fill_value : float, default=0.0
            仅 method='value' 时使用的填充值

        Returns
        -------
        pd.DataFrame
            填充后的因子数据
        """
        df = factor_df.copy()

        if method == "drop":
            return df.dropna(axis=1, how="any")

        if method == "ffill":
            return df.fillna(method="ffill").fillna(method="bfill")

        if method == "zero":
            return df.fillna(0.0)

        if method == "value":
            return df.fillna(fill_value)

        if method == "median":
            for col in df.columns:
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df[col] = df[col].fillna(median_val)
            return df

        if method == "mean":
            for col in df.columns:
                mean_val = df[col].mean()
                if pd.isna(mean_val):
                    mean_val = 0.0
                df[col] = df[col].fillna(mean_val)
            return df

        raise ValueError(f"未知填充方法: {method}")

    # ------------------------------------------------------------------
    # 完整流水线
    # ------------------------------------------------------------------

    def run(self, factor_df: pd.DataFrame,
            industry_map: Optional[pd.Series] = None,
            market_cap: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """按配置执行全套因子处理流程。

        流程: 去极值 -> 标准化 -> 中性化 -> 缺失值填充

        Parameters
        ----------
        factor_df : pd.DataFrame
            原始因子数据
        industry_map : pd.Series, optional
            行业映射 (中性化时需要)
        market_cap : pd.DataFrame, optional
            市值数据 (中性化时需要)

        Returns
        -------
        pd.DataFrame
            处理后的因子数据
        """
        df = factor_df.copy()

        # Step 1: 去极值
        ws_config = self.config.get("winsorize", {})
        if ws_config.get("enabled", True):
            method = ws_config.get("method", "mad")
            std_val = ws_config.get("std", 5.0)
            lower = ws_config.get("lower", 0.01)
            upper = ws_config.get("upper", 0.99)
            df = self.winsorize(df, method=method, std=std_val, lower=lower, upper=upper)

        # Step 2: 标准化
        std_config = self.config.get("standardize", {})
        if std_config.get("enabled", True):
            method = std_config.get("method", "zscore")
            df = self.standardize(df, method=method)

        # Step 3: 中性化
        neut_config = self.config.get("neutralize", {})
        if neut_config.get("enabled", True):
            if industry_map is None or market_cap is None:
                warnings.warn(
                    "中性化已启用但未提供 industry_map 或 market_cap, 跳过该步骤"
                )
            else:
                df = self.neutralize(df, industry_map, market_cap)

        # Step 4: 缺失值填充
        fill_config = self.config.get("fillna", {})
        if fill_config.get("enabled", True):
            method = fill_config.get("method", "median")
            fill_value = fill_config.get("fill_value", 0.0)
            df = self.fillna(df, method=method, fill_value=fill_value)

        return df
