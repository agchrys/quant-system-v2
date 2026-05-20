"""
策略版本管理模块

提供 VersionManager 类，用于管理量化策略的版本迭代、回滚与对比。
"""

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class VersionManager:
    """管理策略版本的生命周期。

    负责创建版本、列出历史、对比差异、回滚操作，以及追踪最优版本。
    """

    def __init__(self, base_dir: str = "./") -> None:
        """初始化 VersionManager。

        Args:
            base_dir: 项目根目录，versions/ 和 version.json 将在此目录下管理。
        """
        self.base_dir = os.path.abspath(base_dir)
        self.versions_dir = os.path.join(self.base_dir, "versions")
        self.version_file = os.path.join(self.base_dir, "version.json")

        os.makedirs(self.versions_dir, exist_ok=True)

        if os.path.exists(self.version_file):
            with open(self.version_file, "r") as f:
                self._data: Dict[str, Any] = json.load(f)
            logger.info(f"已加载版本信息，当前版本: {self._data.get('current_version', 'N/A')}")
        else:
            self._data = {
                "current_version": "v0.0.0",
                "versions": {},
            }
            self._save_version_file()
            logger.info("已初始化版本文件，初始版本 v0.0.0")

    def _save_version_file(self) -> None:
        """将版本数据持久化到 version.json。"""
        with open(self.version_file, "w") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def _version_tuple(self, version: str) -> Tuple[int, int, int]:
        """将版本号字符串转换为可比较的元组。

        Args:
            version: 版本号字符串，格式为 v{major}.{minor}.{patch}。

        Returns:
            包含 (major, minor, patch) 的整型元组。
        """
        parts = version.lstrip("v").split(".")
        return tuple(int(p) for p in parts)  # type: ignore[return-value]

    def get_current_version(self) -> str:
        """返回当前版本号。

        Returns:
            当前版本号字符串，如 v1.2.3。
        """
        return self._data["current_version"]

    def create_version(
        self,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        model_path: Optional[str] = None,
        factor_importance: Optional[Dict[str, float]] = None,
        is_major_improvement: bool = False,
    ) -> str:
        """创建新版本。

        每次调用 patch+1，若 is_major_improvement=True 则 minor+1 并重置 patch。

        Args:
            metrics: 回测性能指标字典，如 {'annual_return': 0.15, 'sharpe': 1.5}。
            params: 该版本使用的参数配置。
            model_path: 模型文件路径，若提供则复制到版本目录。
            factor_importance: 因子重要性排名，格式为 {因子名: 重要性分数}。
            is_major_improvement: 是否为重大改进，若为 True 则 minor+1。

        Returns:
            新创建的版本号。
        """
        current = self.get_current_version()
        major, minor, patch = self._version_tuple(current)

        if is_major_improvement:
            minor += 1
            patch = 0
        else:
            patch += 1

        new_version = f"v{major}.{minor}.{patch}"
        version_dir = os.path.join(self.versions_dir, new_version)
        os.makedirs(version_dir, exist_ok=True)

        # 复制模型文件
        saved_model_path = None
        if model_path and os.path.exists(model_path):
            ext = os.path.splitext(model_path)[1]
            dest_model = os.path.join(version_dir, f"model{ext}")
            shutil.copy2(model_path, dest_model)
            saved_model_path = dest_model
            logger.info(f"模型文件已复制: {dest_model}")

        # 保存参数
        params_path = os.path.join(version_dir, "params.json")
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)

        # 保存指标
        metrics_path = os.path.join(version_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # 保存因子重要性
        if factor_importance:
            fi_path = os.path.join(version_dir, "factor_importance.json")
            with open(fi_path, "w") as f:
                json.dump(factor_importance, f, indent=2, ensure_ascii=False)

        # 记录版本元信息
        version_info = {
            "version": new_version,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "params_path": params_path,
            "metrics_path": metrics_path,
            "model_path": saved_model_path,
            "factor_importance_path": fi_path if factor_importance else None,
        }

        self._data["versions"][new_version] = version_info
        self._data["current_version"] = new_version
        self._save_version_file()

        logger.info(
            f"已创建版本 {new_version}，指标: {metrics.get('annual_return', 'N/A')}"
        )
        return new_version

    def list_versions(self) -> List[Dict[str, Any]]:
        """列出所有版本及其性能指标。

        Returns:
            按创建时间排序的版本信息列表。
        """
        versions = []
        for ver, info in self._data["versions"].items():
            versions.append(
                {
                    "version": ver,
                    "created_at": info["created_at"],
                    "metrics": info["metrics"],
                    "has_model": info["model_path"] is not None,
                    "has_factor_importance": info["factor_importance_path"] is not None,
                }
            )
        versions.sort(key=lambda x: x["created_at"])
        return versions

    def get_best_version(
        self, metric: str = "annual_return"
    ) -> Optional[Dict[str, Any]]:
        """获取指定指标上表现最优的版本。

        Args:
            metric: 用于比较的指标名称，默认为 'annual_return'。

        Returns:
            最优版本的信息字典，若无版本则返回 None。
        """
        best_version = None
        best_value = float("-inf")

        for ver, info in self._data["versions"].items():
            value = info["metrics"].get(metric)
            if value is not None and value > best_value:
                best_value = value
                best_version = {
                    "version": ver,
                    "created_at": info["created_at"],
                    "metrics": info["metrics"],
                }

        if best_version:
            logger.info(f"最优版本 ({metric}): {best_version['version']} = {best_value}")
        else:
            logger.warning(f"未找到包含指标 {metric} 的版本")
        return best_version

    def compare_versions(
        self, v1: str, v2: str
    ) -> Dict[str, Any]:
        """对比两个版本的性能差异。

        Args:
            v1: 第一个版本号。
            v2: 第二个版本号。

        Returns:
            包含对比信息的字典，包括指标差异和改进/退化状态。
        """
        if v1 not in self._data["versions"]:
            raise ValueError(f"版本 {v1} 不存在")
        if v2 not in self._data["versions"]:
            raise ValueError(f"版本 {v2} 不存在")

        info1 = self._data["versions"][v1]
        info2 = self._data["versions"][v2]
        metrics1 = info1["metrics"]
        metrics2 = info2["metrics"]

        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        diffs = {}

        for m in sorted(all_metrics):
            val1 = metrics1.get(m)
            val2 = metrics2.get(m)
            if val1 is not None and val2 is not None:
                diff = val2 - val1
                pct = diff / abs(val1) * 100 if val1 != 0 else float("inf")
                status = "improved" if diff > 0 else ("degraded" if diff < 0 else "unchanged")
            else:
                diff = None
                pct = None
                status = "incomplete"
            diffs[m] = {
                v1: val1,
                v2: val2,
                "diff": diff,
                "diff_pct": pct,
                "status": status,
            }

        report = {
            "v1": v1,
            "v2": v2,
            "v1_created_at": info1["created_at"],
            "v2_created_at": info2["created_at"],
            "metrics": diffs,
        }

        logger.info(f"版本对比完成: {v1} vs {v2}")
        return report

    def rollback(self, version: str) -> bool:
        """回滚到指定版本，恢复模型文件和配置。

        Args:
            version: 要回滚到的目标版本号。

        Returns:
            回滚是否成功。
        """
        if version not in self._data["versions"]:
            raise ValueError(f"版本 {version} 不存在")

        version_info = self._data["versions"][version]

        # 恢复参数
        params_path = version_info["params_path"]
        if os.path.exists(params_path):
            dest_params = os.path.join(self.base_dir, "config", "active_params.json")
            os.makedirs(os.path.dirname(dest_params), exist_ok=True)
            shutil.copy2(params_path, dest_params)
            logger.info(f"参数已恢复: {dest_params}")

        # 恢复模型
        model_path = version_info["model_path"]
        if model_path and os.path.exists(model_path):
            dest_model = os.path.join(self.base_dir, "models", "active_model")
            ext = os.path.splitext(model_path)[1]
            if ext:
                dest_model += ext
            os.makedirs(os.path.dirname(dest_model), exist_ok=True)
            shutil.copy2(model_path, dest_model)
            logger.info(f"模型已恢复: {dest_model}")

        # 更新当前版本
        self._data["current_version"] = version
        self._save_version_file()

        logger.info(f"已回滚到版本 {version}")
        return True

    def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """获取指定版本的详细信息。

        Args:
            version: 版本号。

        Returns:
            版本详细信息字典，若版本不存在则返回 None。
        """
        info = self._data["versions"].get(version)
        if info is None:
            logger.warning(f"版本 {version} 不存在")
            return None
        return {
            "version": version,
            "created_at": info["created_at"],
            "metrics": info["metrics"],
            "params_path": info["params_path"],
            "model_path": info["model_path"],
            "factor_importance_path": info["factor_importance_path"],
        }

    def delete_version(self, version: str) -> bool:
        """删除指定版本及其文件。

        Args:
            version: 要删除的版本号。

        Returns:
            删除是否成功。
        """
        if version not in self._data["versions"]:
            raise ValueError(f"版本 {version} 不存在")

        # 删除版本目录
        version_dir = os.path.join(self.versions_dir, version)
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
            logger.info(f"已删除版本目录: {version_dir}")

        # 从记录中移除
        del self._data["versions"][version]

        # 如果删除的是当前版本，将当前版本指回最后一个版本或 v0.0.0
        if self._data["current_version"] == version:
            remaining = sorted(self._data["versions"].keys(), key=self._version_tuple)
            self._data["current_version"] = remaining[-1] if remaining else "v0.0.0"

        self._save_version_file()
        logger.info(f"已删除版本 {version}")
        return True
