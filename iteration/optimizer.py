"""
自动参数优化 (AutoOptimizer)
===========================
搜索最优模型超参数：网格搜索 + 随机搜索 + 自适应细化。
"""

import itertools
import random
import json
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class AutoOptimizer:
    """自动参数优化器"""

    def __init__(self, config: dict = None):
        """
        参数:
            config: 配置字典（iteration 部分）
        """
        self.config = config or {}
        iter_cfg = self.config.get("iteration", {})

        self.max_iterations = iter_cfg.get("max_iterations", 50)
        self.improvement_threshold = iter_cfg.get("improvement_threshold", 0.01)
        self.search_method = iter_cfg.get("search_method", "grid")
        self.search_space = iter_cfg.get("search_space", {})

        self.metric_target = iter_cfg.get("metric_target", {
            "win_rate": 0.60,
            "annual_return": 0.20,
            "max_drawdown": 0.12,
        })

        self.search_history = []
        self.best_score = -float("inf")
        self.best_params = None
        self.no_improve_count = 0

        logger.info(
            f"AutoOptimizer 初始化: "
            f"方法={self.search_method}, "
            f"最大迭代={self.max_iterations}"
        )

    def grid_search(self, param_grid: dict = None,
                    X_train=None, y_train=None, X_val=None, y_val=None,
                    groups=None) -> dict:
        """
        网格搜索：遍历所有参数组合

        参数:
            param_grid: 参数网格（从 config 读取）
            X_train, y_train, X_val, y_val, groups: 训练数据

        返回:
            最佳参数组合
        """
        param_grid = param_grid or self.search_space
        if not param_grid:
            logger.warning("参数网格为空，返回默认参数")
            return {}

        # 生成所有参数组合
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        logger.info(f"网格搜索: {len(combinations)} 种参数组合")

        best_score = -float("inf")
        best_params = None

        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))

            if X_train is not None and y_train is not None:
                score = self._eval_params(params, X_train, y_train, X_val, y_val, groups)
            else:
                score = self._mock_score(params)

            self.search_history.append({"params": params, "score": score})

            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"  新最优: 组合{i + 1}, score={score:.4f}, params={params}")

        self.best_score = best_score
        self.best_params = best_params

        logger.success(f"网格搜索完成，最优 score={best_score:.4f}")
        return best_params

    def random_search(self, param_dist: dict = None, n_iter: int = 30,
                      X_train=None, y_train=None, X_val=None, y_val=None,
                      groups=None) -> dict:
        """
        随机搜索：从参数分布中随机采样

        参数:
            param_dist: 参数分布（list of values 或 range）
            n_iter: 采样次数
            X_train, y_train, X_val, y_val, groups: 训练数据

        返回:
            最佳参数组合
        """
        param_dist = param_dist or self.search_space
        if not param_dist:
            logger.warning("参数分布为空，返回默认参数")
            return {}

        keys = list(param_dist.keys())
        logger.info(f"随机搜索: {n_iter} 次采样")

        best_score = -float("inf")
        best_params = None

        for i in range(n_iter):
            params = {}
            for k in keys:
                choices = param_dist[k]
                if isinstance(choices, (list, tuple)):
                    params[k] = random.choice(choices)
                elif isinstance(choices, range):
                    params[k] = random.choice(list(choices))

            if X_train is not None and y_train is not None:
                score = self._eval_params(params, X_train, y_train, X_val, y_val, groups)
            else:
                score = self._mock_score(params)

            self.search_history.append({"params": params, "score": score})

            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"  新最优: 采样{i + 1}, score={score:.4f}")

        self.best_score = best_score
        self.best_params = best_params

        logger.success(f"随机搜索完成，最优 score={best_score:.4f}")
        return best_params

    def suggest_next_params(self, best_params: dict,
                            search_history: list = None) -> dict:
        """
        基于历史建议下一组参数（自适应细化）

        策略:
        - 连续 3 次无改善 → 扩大搜索范围
        - 在最优参数附近细化

        参数:
            best_params: 当前最优参数
            search_history: 搜索历史列表

        返回:
            建议的新参数组合
        """
        history = search_history or self.search_history

        if not best_params:
            return {}

        new_params = deepcopy(best_params)

        # 检查是否无改善
        self.no_improve_count += 1
        if len(history) >= 2:
            if history[-1]["score"] > history[-2]["score"]:
                self.no_improve_count = 0

        # 如果在最优参数附近做随机扰动
        for key in best_params:
            val = best_params[key]
            if isinstance(val, (int, float)):
                if self.no_improve_count >= 3:
                    # 扩大范围 ±50%
                    scale = 1.5
                else:
                    # 小范围扰动 ±10%
                    scale = 0.1

                if isinstance(val, int):
                    delta = max(1, int(val * scale))
                    new_val = val + random.randint(-delta, delta)
                    new_params[key] = max(1, new_val)
                else:
                    delta = val * scale
                    new_val = val + random.uniform(-delta, delta)
                    new_params[key] = max(0.001, new_val)

        return new_params

    def run_search(self, X_train=None, y_train=None,
                   X_val=None, y_val=None, groups=None) -> dict:
        """
        执行完整参数搜索流程

        策略: 先粗搜 → 区域细化 → 检查是否达标

        参数:
            X_train, y_train, X_val, y_val, groups: 训练数据

        返回:
            最佳参数组合
        """
        logger.info("=" * 40)
        logger.info("自动参数搜索开始")
        logger.info("=" * 40)

        # Phase 1: 粗搜（网格或随机）
        if self.search_method == "grid":
            params = self.grid_search(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val, groups=groups
            )
        else:
            n_iter = min(self.max_iterations, 30)
            params = self.random_search(
                n_iter=n_iter,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val, groups=groups
            )

        # Phase 2: 细化搜索（在最优参数周围做随机扰动）
        refinement_rounds = 3
        for r in range(refinement_rounds):
            logger.info(f"细化搜索第 {r + 1} 轮")

            # 在最优参数附近做多次尝试
            candidates = [self.suggest_next_params(params) for _ in range(5)]

            best_local_score = -float("inf")
            best_local_params = None

            for candidate in candidates:
                if X_train is not None and y_train is not None:
                    score = self._eval_params(candidate, X_train, y_train,
                                              X_val, y_val, groups)
                else:
                    score = self._mock_score(candidate)

                self.search_history.append({"params": candidate, "score": score})

                if score > best_local_score:
                    best_local_score = score
                    best_local_params = candidate

            if best_local_score > self.best_score:
                self.best_score = best_local_score
                self.best_params = best_local_params
                params = best_local_params
                logger.info(f"  细化改进: score={best_local_score:.4f}")

        logger.success(
            f"参数搜索完成: "
            f"共 {len(self.search_history)} 次评估, "
            f"最优 score={self.best_score:.4f}"
        )

        return self.best_params

    def _eval_params(self, params: dict,
                     X_train, y_train, X_val, y_val, groups) -> float:
        """
        评估一组参数的效果
        使用 LightGBM 快速训练，返回验证集 NDCG

        参数:
            params: 模型参数
            X_train, y_train, X_val, y_val, groups: 数据

        返回:
            评估分数 (越高越好)
        """
        if not HAS_LGB or X_train is None:
            return self._mock_score(params)

        try:
            lgb_params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5],
                "verbosity": -1,
                "num_leaves": params.get("num_leaves", 31),
                "learning_rate": params.get("learning_rate", 0.05),
                "subsample": params.get("subsample", 0.8),
                "colsample_bytree": params.get("colsample_bytree", 0.8),
                "reg_alpha": params.get("reg_alpha", 0.1),
                "reg_lambda": params.get("reg_lambda", 0.1),
                "min_child_samples": params.get("min_child_samples", 20),
            }

            train_data = lgb.Dataset(X_train, label=y_train, group=groups)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=100,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
            )

            # 用验证集的 NDCG 作为分数
            bst_score = model.best_score.get("valid_0", {}).get("ndcg@5", 0)
            return float(bst_score)

        except Exception as e:
            logger.warning(f"参数评估失败: {e}")
            return 0.0

    def _mock_score(self, params: dict) -> float:
        """
        模拟评估（无真实数据时使用）
        基于参数组合的合理性给出模拟分数

        参数:
            params: 参数组合

        返回:
            模拟评分 (0-1)
        """
        score = 0.5

        # 简单规则：适中的参数组合得分更高
        lr = params.get("learning_rate", 0.05)
        nl = params.get("num_leaves", 31)
        ss = params.get("subsample", 0.8)

        # 学习率适中
        if 0.03 <= lr <= 0.08:
            score += 0.1
        if lr > 0.1:
            score -= 0.1

        # 叶子数适中
        if 15 <= nl <= 63:
            score += 0.1
        if nl > 127:
            score -= 0.1

        # 采样率
        if 0.7 <= ss <= 0.9:
            score += 0.05

        # 加一点随机性
        score += random.uniform(-0.05, 0.05)

        return max(0, min(1, score))

    def get_best_params(self) -> dict:
        """获取当前最优参数"""
        return self.best_params

    def get_search_summary(self) -> str:
        """获取搜索过程摘要"""
        if not self.search_history:
            return "尚未执行搜索"

        lines = [
            "参数搜索摘要",
            "=" * 40,
            f"评估次数: {len(self.search_history)}",
            f"最优分数: {self.best_score:.4f}",
            f"最优参数: {self.best_params}",
            "",
            "搜索历史 (Top 5):",
        ]

        sorted_history = sorted(
            self.search_history, key=lambda x: x["score"], reverse=True
        )
        for i, h in enumerate(sorted_history[:5]):
            lines.append(f"  {i + 1}. score={h['score']:.4f}  params={h['params']}")

        return "\n".join(lines)
