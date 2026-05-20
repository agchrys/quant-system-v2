"""
组合优化模块 — Layer 4: Portfolio Optimizer

实现多种组合优化方法，包括风险平价、均值-方差优化，
以及统一的调仓计算和行业约束处理。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.optimize import minimize


class PortfolioOptimizer:
    """组合优化器，负责权重分配、调仓计算与约束检查。

    Parameters
    ----------
    config_path : str
        项目 config.yaml 的路径。优化器读取其中 portfolio 段的配置。
    """

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        with open(config_path, encoding="utf-8") as f:
            full_cfg: Dict[str, Any] = yaml.safe_load(f)

        cfg = full_cfg.get("portfolio", {})

        self.max_positions: int = cfg.get("max_positions", 20)
        self.max_position_weight: float = cfg.get("max_position_weight", 0.10)
        self.max_industry_weight: float = cfg.get("max_industry_weight", 0.30)
        self.target_volatility: float = cfg.get("target_volatility", 0.15)
        self.rebalance_frequency: str = cfg.get("rebalance_frequency", "weekly")

        opt_cfg = cfg.get("optimization", {})
        self.method: str = opt_cfg.get("method", "risk_parity")
        self.risk_aversion: float = opt_cfg.get("risk_aversion", 2.0)

        logger.info(
            "PortfolioOptimizer initialized | max_positions={}, max_weight={}, method={}",
            self.max_positions,
            self.max_position_weight,
            self.method,
        )

    # ------------------------------------------------------------------
    # 风险平价权重
    # ------------------------------------------------------------------

    def risk_parity_weights(
        self,
        cov_matrix: pd.DataFrame,
        max_weight: float = 0.10,
    ) -> pd.Series:
        """风险平价权重 —— 使每个资产的边际风险贡献（MRC）相等。

        使用 SLSQP 求解以下无约束等价问题：
            min  Σ (MRC_i - avg_MRC)²

        Parameters
        ----------
        cov_matrix : pd.DataFrame
            资产协方差矩阵，index / columns 为资产代码。
        max_weight : float, default=0.10
            单资产权重上限。

        Returns
        -------
        pd.Series
            index 为资产代码，value 为权重。
        """
        assets = cov_matrix.index.tolist()
        n = len(assets)

        if n == 0:
            logger.warning("空协方差矩阵，返回空权重序列")
            return pd.Series(dtype=float)

        bounds = [(0.0, max_weight)] * n
        constraints: List[Dict[str, Any]] = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]

        # 初始值：等权
        w0 = np.array([1.0 / n] * n)

        def _risk_parity_objective(weights: np.ndarray) -> float:
            """风险平价目标函数：MRC 差异平方和。"""
            port_var = weights @ cov_matrix.values @ weights
            if port_var <= 0:
                return 0.0
            # 边际风险贡献
            mrc = (cov_matrix.values @ weights) / np.sqrt(port_var) * weights
            target = np.mean(mrc)
            return float(np.sum((mrc - target) ** 2))

        result = minimize(
            _risk_parity_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "maxiter": 1000},
        )

        if not result.success:
            logger.warning("风险平价优化未收敛: {}", result.message)

        weights = result.x
        # 小权重归零后重归一化
        weights[weights < 1e-6] = 0.0
        weights = weights / weights.sum()

        return pd.Series(weights, index=assets, name="risk_parity_weight")

    # ------------------------------------------------------------------
    # 均值-方差优化
    # ------------------------------------------------------------------

    def mean_variance_optimize(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_aversion: float = 2.0,
    ) -> pd.Series:
        """均值-方差优化 —— 最大化风险调整后收益。

        目标函数:
            max  w'μ - λ · w'Σw
        其中 λ = risk_aversion。

        Parameters
        ----------
        expected_returns : pd.Series
            预期收益序列，index 为资产代码。
        cov_matrix : pd.DataFrame
            协方差矩阵。
        risk_aversion : float, default=2.0
            风险厌恶系数 λ（越大越保守）。

        Returns
        -------
        pd.Series
            最优权重序列。
        """
        assets = expected_returns.index.tolist()
        n = len(assets)
        max_w = self.max_position_weight

        bounds = [(0.0, max_w)] * n
        constraints: List[Dict[str, Any]] = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]

        w0 = np.array([1.0 / n] * n)
        mu = expected_returns.values.astype(np.float64)
        sigma = cov_matrix.loc[assets, assets].values.astype(np.float64)

        def _mv_objective(weights: np.ndarray) -> float:
            port_ret = weights @ mu
            port_var = weights @ sigma @ weights
            return float(-(port_ret - risk_aversion * port_var))

        result = minimize(
            _mv_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "maxiter": 1000},
        )

        if not result.success:
            logger.warning("均值-方差优化未收敛: {}", result.message)

        weights = result.x
        weights[weights < 1e-6] = 0.0
        weights = weights / weights.sum()

        return pd.Series(weights, index=assets, name="mv_weight")

    # ------------------------------------------------------------------
    # 统一优化接口
    # ------------------------------------------------------------------

    def optimize(
        self,
        scores: pd.Series,
        cov_matrix: pd.DataFrame,
        method: str = "risk_parity",
    ) -> Dict[str, float]:
        """统一优化接口。

        流程：
        1. 按分数降序选出 Top K 股票
        2. 在选中的股票上执行权重优化
        3. 返回仓位字典

        Parameters
        ----------
        scores : pd.Series
            股票评分，index 为股票代码，value 为分数。
        cov_matrix : pd.DataFrame
            协方差矩阵。
        method : str, default='risk_parity'
            优化方法，支持 'risk_parity' / 'mean_variance' / 'equal_weight'。

        Returns
        -------
        Dict[str, float]
            {股票代码: 权重} 字典。
        """
        # Step 1: Top K 选股
        top_k = min(self.max_positions, len(scores))
        if top_k == 0:
            logger.warning("无可选股票，返回空仓位")
            return {}

        sorted_scores = scores.sort_values(ascending=False)
        selected = sorted_scores.head(top_k)
        selected_assets = selected.index.tolist()

        logger.info("Top {} 候选股票: {}", top_k, selected_assets)

        # Step 2: 子协方差矩阵
        sub_cov = cov_matrix.loc[selected_assets, selected_assets]

        if method == "risk_parity":
            weights_series = self.risk_parity_weights(sub_cov, self.max_position_weight)
        elif method == "mean_variance":
            expected_ret = selected / selected.abs().max() * 0.01  # 分数映射为预期收益
            weights_series = self.mean_variance_optimize(
                expected_ret, sub_cov, self.risk_aversion
            )
        elif method == "equal_weight":
            n = len(selected_assets)
            w = 1.0 / n
            weights_series = pd.Series(
                [w] * n, index=selected_assets, name="equal_weight"
            )
        else:
            logger.warning("未知优化方法 '{}'，回退至等权", method)
            n = len(selected_assets)
            w = 1.0 / n
            weights_series = pd.Series(
                [w] * n, index=selected_assets, name="equal_weight"
            )

        # Step 3: 转字典
        pos_dict: Dict[str, float] = weights_series.to_dict()
        logger.info("优化完成 | method={}, positions={}", method, len(pos_dict))
        return pos_dict

    # ------------------------------------------------------------------
    # 调仓方案
    # ------------------------------------------------------------------

    def rebalance(
        self,
        current_positions: Dict[str, float],
        target_weights: Dict[str, float],
        trade_cost: float = 0.001,
    ) -> List[Tuple[str, str, float]]:
        """计算从当前持仓调整到目标权重的调仓指令。

        交易成本会降低调仓净收益，仅在预期收益 > 成本×阈值 时执行调仓。

        Parameters
        ----------
        current_positions : Dict[str, float]
            当前持仓，{股票代码: 权重}。
        target_weights : Dict[str, float]
            目标权重，{股票代码: 权重}。
        trade_cost : float, default=0.001
            单边交易成本比例。

        Returns
        -------
        List[Tuple[str, str, float]]
            调仓指令列表，每项为 (股票代码, 操作类型, 调整比例)。
            操作类型: 'buy' / 'sell' / 'hold'。
        """
        all_stocks = set(current_positions.keys()) | set(target_weights.keys())
        orders: List[Tuple[str, str, float]] = []

        for stock in all_stocks:
            current_w = current_positions.get(stock, 0.0)
            target_w = target_weights.get(stock, 0.0)
            diff = target_w - current_w

            if abs(diff) < trade_cost:
                orders.append((stock, "hold", 0.0))
                continue

            if diff > 0:
                # 买入 — 需确认收益足以覆盖成本
                if diff > trade_cost:
                    orders.append((stock, "buy", round(diff, 6)))
            else:
                # 卖出
                orders.append((stock, "sell", round(abs(diff), 6)))

        total_adjust = sum(abs(o[2]) for o in orders if o[1] in ("buy", "sell"))
        estimated_cost = total_adjust * trade_cost
        logger.info(
            "Rebalance plan | orders={}, estimated_cost={:.4f}",
            len(orders),
            estimated_cost,
        )

        return orders

    # ------------------------------------------------------------------
    # 行业约束
    # ------------------------------------------------------------------

    def industry_constraint(
        self,
        weights: pd.Series,
        industry_map: Dict[str, str],
        max_industry_weight: float = 0.30,
    ) -> pd.Series:
        """行业集中度约束检查与修正。

        若某行业权重超过阈值，将超限部分的权重按比例重新分配给未超限的行业。

        Parameters
        ----------
        weights : pd.Series
            当前权重序列，index 为股票代码。
        industry_map : Dict[str, str]
            股票 -> 行业 映射字典。
        max_industry_weight : float, default=0.30
            单个行业最大权重。

        Returns
        -------
        pd.Series
            修正后的权重序列。
        """
        if industry_map is None:
            return weights

        # 计算各行业权重
        industry_weight: Dict[str, float] = {}
        for stock in weights.index:
            ind = industry_map.get(stock, "未知")
            industry_weight[ind] = industry_weight.get(ind, 0.0) + weights[stock]

        # 检查超限行业
        over_limit_industries: Dict[str, float] = {}
        total_excess = 0.0

        for ind, w in industry_weight.items():
            if w > max_industry_weight:
                excess = w - max_industry_weight
                over_limit_industries[ind] = excess
                total_excess += excess
                logger.warning("行业 '{}' 权重 {:.2%} 超限 {:.2%}", ind, w, excess)

        if total_excess == 0:
            return weights

        # 找出未超限行业的总权重，作为重新分配基数
        safe_industries = {
            ind: w
            for ind, w in industry_weight.items()
            if ind not in over_limit_industries
        }
        safe_total = sum(safe_industries.values())

        if safe_total <= 0:
            logger.warning("所有行业均超限，无法重新分配，返回原权重")
            return weights

        result = weights.copy()

        for ind, excess in over_limit_industries.items():
            # 该行业内的股票等比例削减
            ind_stocks = [s for s in weights.index if industry_map.get(s) == ind]
            if not ind_stocks:
                continue
            reduction = excess / len(ind_stocks)
            for s in ind_stocks:
                result[s] = max(0.0, result[s] - reduction)

        # 将超限部分按安全行业权重比例分配回去
        for ind, safe_w in safe_industries.items():
            ind_stocks = [s for s in weights.index if industry_map.get(s) == ind]
            if not ind_stocks:
                continue
            allocate_ratio = safe_w / safe_total
            add_amount = total_excess * allocate_ratio / len(ind_stocks)
            for s in ind_stocks:
                result[s] += add_amount

        # 确保权重和 = 1
        result = result / result.sum()
        logger.info("行业约束修正完成 | overlapped_industries={}", len(over_limit_industries))

        return result
