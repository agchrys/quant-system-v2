"""
ModelTrainer — LightGBM 选股模型训练器
==========================================
使用 LightGBM LambdaRank 训练排序模型，支持特征筛选、时序交叉验证、
模型保存与加载。

依赖: lightgbm, sklearn, pandas, numpy, pyyaml, loguru
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder


class ModelTrainer:
    """LightGBM 排序模型训练器。

    负责读取配置、准备时序数据、训练 LambdaRank 模型、
    特征筛选、时序交叉验证及模型持久化。

    Attributes:
        config: 从 YAML 加载的完整配置字典。
        model: 训练完成的 LightGBM Booster 实例。
        feature_importance: 训练后记录的特征重要性（原始值）。
        selected_features: 特征筛选后保留的特征列名列表。
        label_encoder: 用于股票代码编码的 LabelEncoder。
    """

    def __init__(self, config_path: Union[str, Path]) -> None:
        """初始化训练器并加载配置文件。

        Args:
            config_path: YAML 配置文件的路径。

        Raises:
            FileNotFoundError: 配置文件不存在。
            yaml.YAMLError: 配置文件解析失败。
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.model: Optional[lgb.Booster] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.selected_features: Optional[List[str]] = None
        self.label_encoder = LabelEncoder()

        logger.info(f"ModelTrainer 初始化完成，配置文件: {config_path}")

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        factor_df: pd.DataFrame,
        return_df: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
        np.ndarray,
    ]:
        """按时间划分训练/验证/测试集，并生成 group 参数。

        factor_df 与 return_df 的索引需为 ``(date, stock)`` MultiIndex。

        Args:
            factor_df: 因子数据，形状 (n_samples, n_features)，MultiIndex 为 (date, stock)。
            return_df: 未来 N 日收益率，形状 (n_samples,)，MultiIndex 同 factor_df。

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test, groups):
                训练、验证、测试的特征/标签，以及按日期分组的 group 数组。
                groups 用于 LightGBM 的 ``group`` 参数，确保同日期样本在同一组。

        Raises:
            ValueError: 索引不匹配或日期划分配置缺失。
        """
        train_start = self.config["model"]["training"]["train_start"]
        train_end = self.config["model"]["training"]["train_end"]
        val_start = self.config["model"]["training"]["val_start"]
        val_end = self.config["model"]["training"]["val_end"]
        test_start = self.config["model"]["training"]["test_start"]
        test_end = self.config["model"]["training"]["test_end"]

        # 对齐索引
        common_idx = factor_df.index.intersection(return_df.index)
        factor_df = factor_df.loc[common_idx]
        return_df = return_df.loc[common_idx]

        # 提取日期层级
        dates = factor_df.index.get_level_values(0)

        # 按时间划分
        train_mask = (dates >= train_start) & (dates <= train_end)
        val_mask = (dates >= val_start) & (dates <= val_end)
        test_mask = (dates >= test_start) & (dates <= test_end)

        X_train = factor_df[train_mask].copy()
        y_train = return_df[train_mask].copy()
        X_val = factor_df[val_mask].copy()
        y_val = return_df[val_mask].copy()
        X_test = factor_df[test_mask].copy()
        y_test = return_df[test_mask].copy()

        # 生成 group 参数：每个日期作为一个 group
        groups_train = self._make_groups(X_train)

        logger.info(
            f"数据划分完成: "
            f"train={X_train.shape[0]} samples, "
            f"val={X_val.shape[0]} samples, "
            f"test={X_test.shape[0]} samples"
        )

        return X_train, y_train, X_val, y_val, X_test, y_test, groups_train

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        groups_train: np.ndarray,
    ) -> lgb.Booster:
        """使用 LightGBM LambdaRank 训练排序模型。

        Args:
            X_train: 训练特征。
            y_train: 训练标签（未来收益率）。
            X_val: 验证特征。
            y_val: 验证标签。
            groups_train: 训练集 group 数组，表示每个日期组的大小。

        Returns:
            训练完成的 LightGBM Booster 实例。

        Raises:
            RuntimeError: 训练过程中发生异常。
        """
        lgb_config = self.config["model"]["lightgbm"]

        params = {
            "objective": lgb_config["objective"],
            "metric": lgb_config["metric"],
            "ndcg_eval_at": lgb_config["ndcg_eval_at"],
            "num_leaves": lgb_config["num_leaves"],
            "learning_rate": lgb_config["learning_rate"],
            "n_estimators": lgb_config["n_estimators"],
            "subsample": lgb_config["subsample"],
            "colsample_bytree": lgb_config["colsample_bytree"],
            "reg_alpha": lgb_config["reg_alpha"],
            "reg_lambda": lgb_config["reg_lambda"],
            "min_child_samples": lgb_config["min_child_samples"],
            "verbosity": lgb_config["verbosity"],
        }

        early_stop = lgb_config["early_stopping_rounds"]

        # 移除 n_estimators，使用 callbacks 替代
        n_estimators = params.pop("n_estimators")
        eval_at = params.pop("ndcg_eval_at")

        train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
        # 验证集的 group 同样按日期划分
        groups_val = self._make_groups(X_val)
        val_data = lgb.Dataset(X_val, label=y_val, group=groups_val, reference=train_data)

        logger.info("开始训练 LightGBM LambdaRank 模型...")
        self.model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=[val_data],
            num_boost_round=n_estimators,
            callbacks=[
                lgb.early_stopping(early_stop, verbose=True),
                lgb.log_evaluation(period=50),
            ],
        )
        logger.info(f"模型训练完成，最佳迭代次数: {self.model.best_iteration}")

        # 记录特征重要性
        self._record_feature_importance(X_train.columns)

        return self.model

    # ------------------------------------------------------------------
    # 特征筛选
    # ------------------------------------------------------------------

    def feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 30,
    ) -> List[str]:
        """使用 RandomForest 初步筛选 + 相关性去冗余，选出重要特征。

        步骤:
            1. 用 RandomForestRegressor 训练并获取特征重要性。
            2. 保留重要性排名前 ``2 * top_k`` 的特征。
            3. 计算剩余特征的 Pearson 相关系数矩阵。
            4. 对相关系数 > 0.7 的特征对，保留重要性较高的一个。
            5. 返回最终 ``top_k`` 个特征。

        Args:
            X: 因子特征 DataFrame。
            y: 目标标签 Series。
            top_k: 最终保留的特征数量，默认 30。

        Returns:
            选中的特征列名列表。
        """
        logger.info(f"开始特征筛选，目标保留 {top_k} 个特征，输入特征数 {X.shape[1]}")

        # Step 1: RandomForest 粗筛
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X, y)

        importance = pd.DataFrame(
            {"feature": X.columns, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        # 保留 2*top_k 个候选
        candidate_features = importance.head(top_k * 2)["feature"].tolist()
        logger.info(f"RandomForest 粗筛后保留 {len(candidate_features)} 个候选特征")

        # Step 2: 相关性去冗余
        corr_matrix = X[candidate_features].corr(method="pearson")

        selected = []
        dropped = set()

        for feat in candidate_features:
            if feat in dropped:
                continue
            selected.append(feat)
            # 找到与该特征高度相关的其他特征
            high_corr = corr_matrix[feat][
                (corr_matrix[feat].abs() > 0.7) & (corr_matrix[feat].abs() < 1.0)
            ]
            for correlated_feat in high_corr.index:
                if correlated_feat not in dropped:
                    dropped.add(correlated_feat)
                    logger.debug(f"  剔除冗余特征: {correlated_feat} (与 {feat} 相关系数 {corr_matrix.loc[feat, correlated_feat]:.3f})")

        # Step 3: 取 top_k
        self.selected_features = selected[:top_k]
        logger.info(
            f"特征筛选完成: 共保留 {len(self.selected_features)} 个特征，"
            f"剔除了 {len(dropped)} 个冗余特征"
        )

        return self.selected_features

    # ------------------------------------------------------------------
    # 时序交叉验证
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: np.ndarray,
        n_folds: int = 5,
    ) -> Dict[str, List[float]]:
        """按时序做 GroupKFold 交叉验证。

        GroupKFold 确保同一天的样本不会被分割到不同 fold，避免未来信息泄露。

        Args:
            X: 特征 DataFrame。
            y: 标签 Series。
            groups: group 数组（按日期编码的整型）。
            n_folds: 折数，默认 5。

        Returns:
            各 fold 的评估指标::

                {
                    "ndcg@5":  [ndcg5_fold1, ndcg5_fold2, ...],
                    "ndcg@10": [ndcg10_fold1, ...],
                    "val_size": [size_fold1, ...],
                }
        """
        lgb_config = self.config["model"]["lightgbm"]
        eval_at = lgb_config.get("ndcg_eval_at", [5, 10, 20])

        params = {
            "objective": lgb_config["objective"],
            "metric": lgb_config["metric"],
            "num_leaves": lgb_config["num_leaves"],
            "learning_rate": lgb_config["learning_rate"],
            "n_estimators": 100,  # CV 时使用较少的迭代次数
            "subsample": lgb_config["subsample"],
            "colsample_bytree": lgb_config["colsample_bytree"],
            "reg_alpha": lgb_config["reg_alpha"],
            "reg_lambda": lgb_config["reg_lambda"],
            "min_child_samples": lgb_config["min_child_samples"],
            "verbosity": -1,
        }

        # 按日期排序，确保时序
        unique_dates = np.unique(groups)

        cv = GroupKFold(n_splits=min(n_folds, len(unique_dates)))

        results: Dict[str, List[float]] = {f"ndcg@{k}": [] for k in eval_at}
        results["val_size"] = []

        logger.info(f"开始 {n_folds}-Fold 时序交叉验证...")

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            groups_fold_train = groups[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]

            # 构建 group 长度
            group_sizes = self._compute_group_sizes(groups_fold_train)

            train_data = lgb.Dataset(
                X_fold_train, label=y_fold_train, group=group_sizes
            )
            val_data = lgb.Dataset(
                X_fold_val, label=y_fold_val, reference=train_data
            )

            model = lgb.train(
                params=params,
                train_set=train_data,
                valid_sets=[val_data],
                num_boost_round=params["n_estimators"],
                callbacks=[lgb.early_stopping(10, verbose=False)],
            )

            # 评估
            y_pred = model.predict(X_fold_val)
            y_true = y_fold_val.values.reshape(1, -1)
            y_pred_scores = y_pred.reshape(1, -1)

            for k in eval_at:
                try:
                    ndcg = ndcg_score(y_true, y_pred_scores, k=k)
                except Exception:
                    ndcg = 0.0
                results[f"ndcg@{k}"].append(round(ndcg, 4))

            results["val_size"].append(len(y_fold_val))

            logger.info(
                f"  Fold {fold_idx + 1}/{n_folds}: "
                f"val_size={len(y_fold_val)}, "
                + ", ".join(
                    f"ndcg@{k}={results[f'ndcg@{k}'][-1]:.4f}"
                    for k in eval_at
                )
            )

        # 汇总
        for k in eval_at:
            vals = results[f"ndcg@{k}"]
            logger.info(
                f"CV ndcg@{k}: mean={np.mean(vals):.4f} ± {np.std(vals):.4f}"
            )

        return results

    # ------------------------------------------------------------------
    # 模型持久化
    # ------------------------------------------------------------------

    def save_model(self, path: Union[str, Path]) -> None:
        """将训练好的 LightGBM 模型保存到磁盘。

        Args:
            path: 保存路径。若路径以 ``.txt`` 结尾则使用 LightGBM 原生格式，
                  否则使用 ``joblib`` 格式（含元数据）。

        Raises:
            RuntimeError: 模型尚未训练。
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练，无法保存。请先调用 train()。")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(path))
        logger.info(f"模型已保存至: {path}")

    def load_model(self, path: Union[str, Path]) -> lgb.Booster:
        """从磁盘加载 LightGBM 模型。

        Args:
            path: 模型文件路径。

        Returns:
            加载的 LightGBM Booster 实例。

        Raises:
            FileNotFoundError: 模型文件不存在。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")

        self.model = lgb.Booster(model_file=str(path))
        logger.info(f"模型已加载: {path}")
        return self.model

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _make_groups(self, df: pd.DataFrame) -> np.ndarray:
        """从 MultiIndex DataFrame 提取 group 数组。

        每个日期为一个 group，返回每个样本对应的日期编号。

        Args:
            df: 索引为 (date, stock) MultiIndex 的 DataFrame。

        Returns:
            group 数组，用于 LightGBM 的 ``group`` 参数。长度等于样本数。
        """
        dates = df.index.get_level_values(0)
        return self.label_encoder.fit_transform(dates)

    @staticmethod
    def _compute_group_sizes(groups: np.ndarray) -> np.ndarray:
        """计算每个 group 内的样本数。

        Args:
            groups: group 编号数组。

        Returns:
            每个 group 的大小数组。
        """
        unique, counts = np.unique(groups, return_counts=True)
        return counts

    def _record_feature_importance(self, feature_names: pd.Index) -> None:
        """记录训练完成的模型的特征重要性。

        Args:
            feature_names: 特征列名。
        """
        if self.model is None:
            return

        importance = self.model.feature_importance(importance_type="gain")
        self.feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance_gain": importance}
        ).sort_values("importance_gain", ascending=False)

        logger.info(
            f"Top-10 重要特征: "
            + ", ".join(
                f"{row['feature']}({row['importance_gain']:.2f})"
                for _, row in self.feature_importance.head(10).iterrows()
            )
        )
