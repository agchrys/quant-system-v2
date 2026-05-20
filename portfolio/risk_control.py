"""
多层级风险控制 (RiskController)
==============================
四层嵌套风控机制：个股止损 → 组合回撤 → 波动率目标 → 异常检测
"""

import numpy as np
import pandas as pd
from loguru import logger


class RiskController:
    """风险控制器"""

    def __init__(self, config: dict = None):
        """
        参数:
            config: 配置字典 (config.yaml 中 risk 部分)
        """
        self.config = config or {}
        risk_cfg = self.config.get("risk", {})

        # 个股止损
        sl_cfg = risk_cfg.get("stop_loss", {})
        self.stop_loss_enabled = sl_cfg.get("enabled", True)
        self.loss_pct = sl_cfg.get("loss_pct", 0.03)
        self.time_stop_days = sl_cfg.get("time_stop_days", 5)
        self.time_stop_pct = sl_cfg.get("time_stop_pct", 0.0)

        # 组合回撤
        dd_cfg = risk_cfg.get("drawdown", {})
        self.drawdown_enabled = dd_cfg.get("enabled", True)
        self.dd_levels = [
            {"threshold": dd_cfg.get("level1", {}).get("threshold", 0.05),
             "reduce_to": dd_cfg.get("level1", {}).get("reduce_to", 0.60)},
            {"threshold": dd_cfg.get("level2", {}).get("threshold", 0.07),
             "reduce_to": dd_cfg.get("level2", {}).get("reduce_to", 0.30)},
            {"threshold": dd_cfg.get("level3", {}).get("threshold", 0.10),
             "reduce_to": dd_cfg.get("level3", {}).get("reduce_to", 0.00),
             "cooldown": dd_cfg.get("level3", {}).get("cooldown_days", 3)},
        ]

        # 波动率目标
        vt_cfg = risk_cfg.get("vol_target", {})
        self.vol_target_enabled = vt_cfg.get("enabled", True)
        self.target_vol = vt_cfg.get("annual_target", 0.12)
        self.vol_lookback = vt_cfg.get("lookback_days", 60)
        self.max_leverage = vt_cfg.get("max_leverage", 1.0)

        # 冷却状态
        self.cooldown_until = None

        logger.info(
            f"RiskController 初始化: "
            f"止损={self.stop_loss_enabled}({self.loss_pct:.0%}), "
            f"回撤控制={self.drawdown_enabled}"
        )

    def check_stop_loss(self, stock_code: str, current_price: float,
                        entry_price: float, days_held: int,
                        highest_price: float = None) -> dict:
        """
        个股止损检查

        参数:
            stock_code: 股票代码
            current_price: 当前价格
            entry_price: 买入价格
            days_held: 持有天数
            highest_price: 持仓期间最高价（用于移动止盈）

        返回:
            {'action': 'hold'/'stop_loss'/'time_stop', 'reason': str}
        """
        if not self.stop_loss_enabled:
            return {"action": "hold", "reason": "止损未启用"}

        result = {"action": "hold", "reason": ""}

        # 1. 固定比例止损
        loss_ratio = (current_price - entry_price) / entry_price
        if loss_ratio <= -self.loss_pct:
            result["action"] = "stop_loss"
            result["reason"] = f"亏损{loss_ratio:.2%}，触发止损线{self.loss_pct:.0%}"
            logger.warning(f"[风控] {stock_code}: {result['reason']}")
            return result

        # 2. 时间止损（买入后 N 日不涨）
        if days_held >= self.time_stop_days and loss_ratio <= self.time_stop_pct:
            result["action"] = "time_stop"
            result["reason"] = f"持有{days_held}日涨幅{loss_ratio:.2%}，触发时间止损"
            logger.warning(f"[风控] {stock_code}: {result['reason']}")
            return result

        # 3. 移动止盈（从最高点回落）
        if highest_price is not None and highest_price > entry_price * 1.05:
            drawdown_from_peak = (current_price - highest_price) / highest_price
            if drawdown_from_peak <= -0.05:
                result["action"] = "trailing_stop"
                result["reason"] = f"从最高点回落{drawdown_from_peak:.2%}，触发移动止盈"
                logger.info(f"[风控] {stock_code}: {result['reason']}")
                return result

        return result

    def check_drawdown(self, current_value: float, peak_value: float) -> dict:
        """
        组合回撤检查

        参数:
            current_value: 当前组合价值
            peak_value: 组合历史峰值

        返回:
            {'action': str, 'reduce_to': float, 'reason': str}
        """
        if not self.drawdown_enabled or peak_value <= 0:
            return {"action": "none", "reduce_to": 1.0, "reason": ""}

        drawdown = (peak_value - current_value) / peak_value
        result = {"action": "normal", "reduce_to": 1.0, "reason": ""}

        # 按阈值从高到低匹配
        for level in reversed(self.dd_levels):
            if drawdown >= level["threshold"]:
                reduce_to = level["reduce_to"]
                is_cooldown = level.get("cooldown", 0) > 0
                cooldown_days = level.get("cooldown", 0)

                result["action"] = "cooldown" if is_cooldown else "reduce"
                result["reduce_to"] = reduce_to

                if is_cooldown:
                    result["reason"] = (
                        f"回撤{drawdown:.2%}≥{level['threshold']:.0%}，"
                        f"清仓冷却{cooldown_days}天"
                    )
                    self.cooldown_until = cooldown_days
                else:
                    result["reason"] = (
                        f"回撤{drawdown:.2%}≥{level['threshold']:.0%}，"
                        f"减仓至{reduce_to:.0%}"
                    )

                logger.warning(f"[风控] 组合: {result['reason']}")
                return result

        return result

    def check_volatility_target(self, portfolio_returns: np.ndarray,
                                annual_target: float = None) -> dict:
        """
        波动率目标检查

        参数:
            portfolio_returns: 近期组合日收益率数组
            annual_target: 目标年化波动率（默认 12%）

        返回:
            {'action': str, 'adjustment': float, 'reason': str}
        """
        if not self.vol_target_enabled or len(portfolio_returns) < 20:
            return {"action": "none", "adjustment": 1.0, "reason": ""}

        target = annual_target or self.target_vol
        realized_vol = np.std(portfolio_returns) * np.sqrt(252)

        if realized_vol <= target:
            return {"action": "normal", "adjustment": 1.0,
                    "reason": f"波动率{realized_vol:.2%}在目标{target:.0%}以内"}

        # 计算调整系数
        adjustment = target / realized_vol
        adjustment = min(adjustment, 1.0 / (1 - self.max_leverage + 0.01))
        adjustment = max(adjustment, 0.1)  # 最低保留 10%

        result = {
            "action": "reduce_leverage",
            "adjustment": adjustment,
            "reason": f"波动率{realized_vol:.2%}超目标{target:.0%}，杠杆降至{adjustment:.1%}"
        }
        logger.warning(f"[风控] 波动率: {result['reason']}")
        return result

    def check_cooldown(self, current_date) -> bool:
        """检查是否处于冷却期"""
        if self.cooldown_until is None:
            return False
        if self.cooldown_until > 0:
            self.cooldown_until -= 1
            logger.info(f"[风控] 冷却中，剩余{self.cooldown_until}天")
            return True
        else:
            self.cooldown_until = None
            return False

    def check_all(self, portfolio_state: dict) -> dict:
        """
        全面风控检查

        参数:
            portfolio_state: 组合状态字典，包含:
                - current_value: 当前价值
                - peak_value: 历史峰值
                - positions: {stock_code: {price, entry_price, days_held, ...}}
                - portfolio_returns: 近期收益率数组（可选）
                - date: 当前日期

        返回:
            {
                'action': str,
                'reduce_to': float,
                'explanations': [str],
                'triggered_events': [dict]
            }
        """
        triggered_events = []
        explanations = []
        current_reduce = 1.0

        # 1. 冷却期检查
        if self.check_cooldown(portfolio_state.get("date")):
            triggered_events.append({"type": "cooldown", "detail": "冷却期内"})
            explanations.append("冷却期：不进行任何操作")
            return {
                "action": "cooldown",
                "reduce_to": 0.0,
                "explanations": explanations,
                "triggered_events": triggered_events,
            }

        # 2. 组合回撤检查
        dd_check = self.check_drawdown(
            portfolio_state.get("current_value", 1),
            portfolio_state.get("peak_value", 1)
        )
        current_reduce = min(current_reduce, dd_check.get("reduce_to", 1.0))
        if dd_check["action"] != "none":
            triggered_events.append(dd_check)
            explanations.append(dd_check["reason"])

        # 3. 波动率目标检查
        returns = portfolio_state.get("portfolio_returns", [])
        if isinstance(returns, list) and len(returns) > 20:
            vol_check = self.check_volatility_target(np.array(returns))
            current_reduce = min(current_reduce, vol_check.get("adjustment", 1.0))
            if vol_check["action"] != "none":
                triggered_events.append(vol_check)
                explanations.append(vol_check["reason"])

        # 4. 个股止损（遍历所有持仓）
        positions = portfolio_state.get("positions", {})
        stop_loss_signals = []
        for code, pos in positions.items():
            sl = self.check_stop_loss(
                code,
                pos.get("price", 0),
                pos.get("entry_price", 0),
                pos.get("days_held", 0),
                pos.get("highest_price")
            )
            if sl["action"] != "hold":
                stop_loss_signals.append({"stock": code, **sl})
                triggered_events.append({**sl, "stock": code})
                explanations.append(f"[{code}] {sl['reason']}")

        # 汇总
        overall_action = "normal"
        if current_reduce <= 0:
            overall_action = "clear"
        elif current_reduce < 1.0:
            overall_action = "reduce"

        return {
            "action": overall_action,
            "reduce_to": current_reduce,
            "stop_loss_signals": stop_loss_signals,
            "explanations": explanations,
            "triggered_events": triggered_events,
        }

    def get_risk_report(self, history: list) -> str:
        """
        生成风控事件报告

        参数:
            history: 风控事件历史记录列表

        返回:
            格式化的报告字符串
        """
        if not history:
            return "无风控事件触发"

        lines = ["=" * 50, "风控事件报告", "=" * 50]
        lines.append(f"总计触发: {len(history)} 次")

        # 按类型统计
        from collections import Counter
        type_counts = Counter(e.get("action", e.get("type", "unknown")) for e in history)
        for t, c in type_counts.most_common():
            lines.append(f"  {t}: {c} 次")

        lines.append("")
        for i, event in enumerate(history[-10:], 1):  # 最近 10 条
            lines.append(f"  {i}. [{event.get('date', '?')}] "
                         f"{event.get('action', event.get('type', '?'))}: "
                         f"{event.get('reason', '')}")

        return "\n".join(lines)
