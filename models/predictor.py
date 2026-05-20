"""
Predictor — 推理预测与选股排序
===================================
对因子数据进行模型推理，输出个股预期收益排序分数，
支持多模型集成预测与分数权重转换。

依赖: lightgbm, pandas, numpy, loguru
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger


class Predictor:
    """模型推理预测器。

    提供单模型预测、股票排序、多模型集成预测及分数权重转换功能。

    Attributes:
        recent_performance: 记录各模型近期表现（如 NDCG），用于动态加权。
    """

    def __init__(self) -> None:
        """初始化预测器。"""
        self.recent_performance: Dict[str, float] = {}
        logger.info("Predictor 初始化完成")

    # ------------------------------------------------------------------
    # 单模型预测
    # ------------------------------------------------------------------

    def predict(
        self,
        model: lgb.Booster,
        factor_df: pd.DataFrame,
    ) -> pd.Series:
        """使用单模型对因子数据进行推理，返回每只股票的预期收益排序分数。

        Args:
            model: 训练好的 LightGBM Booster 实例。
            factor_df: 因子数据，形状 (n_stocks, n_features)。
                       列名需与训练时的特征列一致。

        Returns:
            Series，索引与 ``factor_df`` 相同，值为模型预测的排序分数。
            分数越高表示模型认为该股票未来相对表现越好。

        Raises:
            ValueError: 特征列不匹配。
        """
        # 检查特征一致性
        expected_features = model.feature_name()
        missing = set(expected_features) - set(factor_df.columns)
        if missing:
            raise ValueError(
                f"因子数据缺少以下特征列: {missing}。"
                f"请确保 factor_df 包含模型训练时使用的所有特征。"
            )

        # 只取模型需要的特征，按模型顺序排列
        X = factor_df[expected_features]

        scores = model.predict(X)
        result = pd.Series(scores, index=factor_df.index, name="pred_score")

        logger.info(
            f"单模型预测完成: {len(result)} 只股票, "
            f"分数范围 [{scores.min():.4f}, {scores.max():.4f}]"
        )

        return result

    # ------------------------------------------------------------------
    # 股票排序
    # ------------------------------------------------------------------

    def rank_stocks(
        self,
        scores: pd.Series,
        top_k: int = 20,
    ) -> List[Tuple[Any, float]]:
        """对预测分数排序，返回得分最高的 Top K 股票列表。

        Args:
            scores: ``predict()`` 输出的分数 Series，索引为股票标识。
            top_k: 返回前 K 只股票，默认 20。

        Returns:
            列表，每个元素为 ``(stock_identifier, score)``，
            按分数降序排列。
        """
        sorted_scores = scores.sort_values(ascending=False)
        top = sorted_scores.head(top_k)

        result = [(idx, round(score, 6)) for idx, score in top.items()]

        logger.info(f"股票排序完成: Top-1={result[0]}, Top-{top_k} 已返回")
        return result

    # ------------------------------------------------------------------
    # 多模型集成预测
    # ------------------------------------------------------------------

    def ensemble_predict(
        self,
        models_dict: Dict[str, lgb.Booster],
        factor_df: pd.DataFrame,
    ) -> pd.Series:
        """多模型集成预测。

        使用近期表现作为权重对各模型预测分数进行加权平均。
        若无近期表现记录，则等权平均。

        Args:
            models_dict: 模型字典，``{模型名称: Booster 实例}``。
            factor_df: 因子数据 DataFrame。

        Returns:
            集成后的预测分数 Series。
        """
        if not models_dict:
            raise ValueError("models_dict 不能为空，请提供至少一个模型。")

        predictions: Dict[str, pd.Series] = {}

        for name, model in models_dict.items():
            try:
                pred = self.predict(model, factor_df)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"模型 '{name}' 预测失败: {e}，跳过")
                continue

        if not predictions:
            raise RuntimeError("所有模型预测均失败，无法进行集成。")

        # 确定权重
        weights = self._get_ensemble_weights(list(predictions.keys()))

        # 加权平均
        ensemble_score: pd.Series = sum(
            predictions[name] * weights[name] for name in predictions
        )

        logger.info(
            f"集成预测完成: {len(predictions)} 个模型参与, "
            f"权重分布: {weights}"
        )

        return ensemble_score

    # ------------------------------------------------------------------
    # 分数转权重
    # ------------------------------------------------------------------

    def score_to_weights(
        self,
        scores: pd.Series,
        method: str = "linear",
    ) -> pd.Series:
        """将预测分数转换为持仓权重。

        Args:
            scores: 预测分数 Series，索引为股票标识。
            method: 权重计算方法:

                - ``'linear'``: 线性归一化，权重与分数成正比。
                - ``'rank'``: 排名权重，将分数排名后等距映射到 [0, 1]。
                - ``'softmax'``: Softmax 变换，指数归一化，放大差异。

        Returns:
            权重 Series，所有权重之和为 1.0。

        Raises:
            ValueError: 不支持的 method 参数。
        """
        if scores.empty:
            logger.warning("scores 为空，返回空权重")
            return scores

        if method == "linear":
            # 线性归一化: (score - min) / (max - min), 再归一化到和为 1
            min_s, max_s = scores.min(), scores.max()
            if max_s == min_s:
                weights = pd.Series(1.0 / len(scores), index=scores.index)
            else:
                weights = (scores - min_s) / (max_s - min_s)

        elif method == "rank":
            # 排名权重: 排名越高权重越大，等距映射
            ranks = scores.rank(method="average", ascending=True)
            weights = ranks / ranks.sum()

        elif method == "softmax":
            # Softmax: 指数归一化
            exp_scores = np.exp(scores - scores.max())  # 防止溢出
            weights = exp_scores / exp_scores.sum()

        else:
            raise ValueError(
                f"不支持的权重计算方法: '{method}'。"
                f"可选值: 'linear', 'rank', 'softmax'。"
            )

        # 确保所有权重之和为 1
        weights = weights / weights.sum()

        logger.info(f"分数转权重完成: method={method}, 非零权重数={(weights > 0).sum()}")
        return weights

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def update_performance(self, model_name: str, metric_value: float) -> None:
        """更新模型的近期表现指标。

        由外部回测或验证流程调用，记录各模型最新 NDCG / IC 等指标。

        Args:
            model_name: 模型名称，需与 ``models_dict`` 的键一致。
            metric_value: 近期表现指标值（如 NDCG, IC）。
        """
        self.recent_performance[model_name] = metric_value
        logger.debug(f"模型 '{model_name}' 近期表现已更新: {metric_value:.4f}")

    def _get_ensemble_weights(
        self,
        model_names: List[str],
    ) -> Dict[str, float]:
        """根据近期表现计算各模型的集成权重。

        若某模型无近期表现记录，使用所有模型权重的中位数代替。

        Args:
            model_names: 参与集成的模型名称列表。

        Returns:
            模型名称到权重的映射字典。
        """
        n = len(model_names)
        if n == 0:
            return {}

        if not self.recent_performance:
            # 无记录，等权平均
            return {name: 1.0 / n for name in model_names}

        # 获取各模型表现值
        values = []
        for name in model_names:
            val = self.recent_performance.get(name)
            if val is None:
                # 无记录则使用其他模型的中位数
                other_vals = [
                    self.recent_performance[m]
                    for m in model_names
                    if m in self.recent_performance
                ]
                val = np.median(other_vals) if other_vals else 0.0
            values.append(max(val, 0.0))  # 截断负值

        values_arr = np.array(values, dtype=float)
        total = values_arr.sum()
        if total <= 0:
            return {name: 1.0 / n for name in model_names}

        weights_arr = values_arr / total
        return {name: float(w) for name, w in zip(model_names, weights_arr)}
