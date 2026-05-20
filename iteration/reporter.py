"""
可视化报告生成器 (Reporter)
============================
生成完整的 HTML 性能报告，包含净值曲线、回撤图、月度收益矩阵等。
使用 matplotlib 生成图片并嵌入 HTML（无需外部 web 服务器）。
"""

import os
import base64
import io
import json
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC",
                                    "PingFang SC", "STHeiti", "SimHei",
                                    "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class Reporter:
    """HTML 报告生成器"""

    def __init__(self):
        logger.info("Reporter 初始化")

    def generate_html_report(self, metrics: dict = None,
                             nav_series: pd.Series = None,
                             trades: list = None,
                             risk_events: list = None,
                             factor_importance: dict = None) -> str:
        """
        生成完整的 HTML 性能报告

        参数:
            metrics: 性能指标字典
            nav_series: 净值序列
            trades: 交易记录列表
            risk_events: 风控事件列表
            factor_importance: 因子重要性字典

        返回:
            HTML 字符串
        """
        metrics = metrics or {}
        trades = trades or []
        risk_events = risk_events or []
        factor_importance = factor_importance or {}

        # 生成图表
        chart_images = []
        if nav_series is not None and len(nav_series) > 5:
            chart_images.append(
                ("净值曲线", self._plot_nav(nav_series))
            )
            chart_images.append(
                ("回撤曲线", self._plot_drawdown(nav_series))
            )
            chart_images.append(
                ("月度收益热力图", self._plot_monthly_returns(nav_series))
            )

        # 构建 HTML
        html = self._build_html(
            metrics, chart_images, trades, risk_events, factor_importance
        )
        return html

    def save_report(self, html_content: str, path: str):
        """
        保存报告到文件

        参数:
            html_content: HTML 字符串
            path: 保存路径
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"报告已保存: {path}")

    def compare_reports(self, metrics_list: list) -> str:
        """
        多版本对比报告

        参数:
            metrics_list: 各版本的指标字典列表

        返回:
            对比 HTML 字符串
        """
        rows = []
        headers = set()
        for m in metrics_list:
            headers.update(m.keys())

        for i, m in enumerate(metrics_list):
            version = m.get("version", f"v{i}")
            row = f"<tr><td><strong>{version}</strong></td>"
            for h in sorted(headers):
                if h == "version":
                    continue
                val = m.get(h, "-")
                if isinstance(val, float):
                    if abs(val) < 1:
                        row += f"<td>{val * 100:.2f}%</td>"
                    else:
                        row += f"<td>{val:.2f}</td>"
                else:
                    row += f"<td>{val}</td>"
            row += "</tr>"
            rows.append(row)

        header_html = "<th>版本</th>"
        for h in sorted(headers):
            if h == "version":
                continue
            display = h.replace("_", " ").title()
            header_html += f"<th>{display}</th>"

        html = f"""<div style="overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;font-size:13px;">
<thead><tr style="background:#1a2332;">{header_html}</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table></div>"""
        return html

    # ========== 图表绘制 ==========

    def _plot_nav(self, nav_series: pd.Series) -> str:
        """绘制净值曲线"""
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#0a0f1a")
        ax.set_facecolor("#111827")

        dates = nav_series.index
        values = nav_series.values

        ax.fill_between(dates, values, 1, alpha=0.15, color="#22c55e")
        ax.plot(dates, values, color="#22c55e", linewidth=1.5)

        # 基准线
        ax.axhline(y=1, color="#5a6680", linewidth=0.5, linestyle="--", alpha=0.5)

        ax.set_title("净值曲线", color="#e2e8f0", fontsize=14, pad=12)
        ax.tick_params(colors="#8892a8", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#2d3a50")
        ax.spines["left"].set_color("#2d3a50")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=30)
        plt.grid(alpha=0.1, color="#5a6680")
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _plot_drawdown(self, nav_series: pd.Series) -> str:
        """绘制回撤曲线"""
        fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#0a0f1a")
        ax.set_facecolor("#111827")

        rolling_max = nav_series.expanding().max()
        drawdown = (nav_series - rolling_max) / rolling_max

        dates = drawdown.index
        values = drawdown.values * 100

        ax.fill_between(dates, values, 0, alpha=0.3, color="#ef4444")
        ax.plot(dates, values, color="#ef4444", linewidth=1)

        # 标注最大回撤
        min_idx = drawdown.idxmin()
        min_val = drawdown.min() * 100
        ax.annotate(
            f"最大回撤 {min_val:.1f}%",
            xy=(min_idx, min_val),
            xytext=(min_idx, min_val * 1.3),
            color="#ef4444",
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1),
        )

        ax.set_title("回撤曲线", color="#e2e8f0", fontsize=14, pad=12)
        ax.tick_params(colors="#8892a8", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#2d3a50")
        ax.spines["left"].set_color("#2d3a50")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=30)
        plt.grid(alpha=0.1, color="#5a6680")
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _plot_monthly_returns(self, nav_series: pd.Series) -> str:
        """绘制月度收益热力图"""
        daily_returns = nav_series.pct_change().dropna()
        if len(daily_returns) < 20:
            return ""

        monthly = daily_returns.groupby([
            daily_returns.index.year,
            daily_returns.index.month
        ]).apply(lambda x: (1 + x).prod() - 1) * 100

        # 转为矩阵
        monthly_df = monthly.unstack(level=0)
        if monthly_df.empty:
            return ""

        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0a0f1a")
        ax.set_facecolor("#111827")

        # 绘制热力图
        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad("#1a2332")

        im = ax.imshow(
            monthly_df.values,
            aspect="auto",
            cmap=cmap,
            vmin=-10,
            vmax=10,
        )

        # 标注数值
        for i in range(monthly_df.shape[0]):
            for j in range(monthly_df.shape[1]):
                val = monthly_df.values[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 5 else "#e2e8f0"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                            fontsize=8, color=color)

        ax.set_xticks(range(len(monthly_df.columns)))
        ax.set_xticklabels([str(int(c)) for c in monthly_df.columns], color="#8892a8")
        ax.set_yticks(range(len(monthly_df.index)))
        ax.set_yticklabels([f"{int(m)}月" for m in monthly_df.index], color="#8892a8")

        ax.set_title("月度收益率 (%)", color="#e2e8f0", fontsize=14, pad=12)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """将 matplotlib figure 转为 base64 字符串"""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    # ========== HTML 构建 ==========

    def _build_html(self, metrics: dict, chart_images: list,
                    trades: list, risk_events: list,
                    factor_importance: dict) -> str:
        """构建完整的 HTML 报告"""
        version = metrics.get("version", "latest")

        # 指标卡片
        cards_html = self._build_metric_cards(metrics)

        # 图表
        charts_html = ""
        for title, img_data in chart_images:
            if img_data:
                charts_html += f"""
<div style="margin:20px 0;">
  <h3 style="color:#f59e0b;font-size:16px;margin-bottom:8px;">📊 {title}</h3>
  <img src="{img_data}" style="width:100%;border-radius:8px;border:1px solid #2d3a50;">
</div>"""

        # 因子重要性
        factor_html = ""
        if factor_importance:
            factor_html = self._build_factor_importance(factor_importance)

        # 交易记录
        trades_html = ""
        if trades:
            trades_html = self._build_trades_table(trades)

        # 风控事件
        risk_html = ""
        if risk_events:
            risk_html = self._build_risk_events(risk_events)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>量化系统报告 v{version}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans SC', sans-serif;
  background: #0a0f1a; color: #e2e8f0; line-height: 1.6;
}}
.container {{ max-width: 960px; margin: 0 auto; padding: 20px; }}
.header {{
  text-align: center; padding: 40px 20px 30px;
  border-bottom: 1px solid #2d3a50; margin-bottom: 30px;
}}
.header h1 {{ font-size: 24px; color: #f59e0b; margin-bottom: 6px; }}
.header p {{ color: #8892a8; font-size: 14px; }}
.metrics-grid {{
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px; margin: 20px 0;
}}
.metric-card {{
  background: #1e293b; border: 1px solid #2d3a50;
  border-radius: 10px; padding: 16px; text-align: center;
}}
.metric-card .value {{ font-size: 22px; font-weight: 700; }}
.metric-card .label {{ font-size: 12px; color: #8892a8; margin-top: 4px; }}
.metric-card .green {{ color: #22c55e; }}
.metric-card .red {{ color: #ef4444; }}
.metric-card .yellow {{ color: #f59e0b; }}
.metric-card .blue {{ color: #3b82f6; }}
h2 {{ color: #f59e0b; font-size: 18px; margin: 24px 0 12px;
     padding-left: 0; border-left: none; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ background: #1a2332; padding: 8px 12px; text-align: left;
     font-weight: 600; color: #8892a8; border-bottom: 1px solid #2d3a50; }}
td {{ padding: 8px 12px; border-bottom: 1px solid rgba(45,58,80,.3); }}
tr:hover td {{ background: rgba(245,158,11,.03); }}
.footer {{ text-align: center; padding: 30px; color: #5a6680; font-size: 12px; }}
</style></head>
<body>
<div class="container">
  <div class="header">
    <h1>📈 量化系统性能报告</h1>
    <p>版本: {version} · 生成时间: {now}</p>
  </div>

  {cards_html}
  {charts_html}
  {factor_html}
  {trades_html}
  {risk_html}

  <div class="footer">
    Quant-System Personal Edition · 数据驱动 · 机器学习 · 严格风控
  </div>
</div>
</body>
</html>"""

        return html

    def _build_metric_cards(self, metrics: dict) -> str:
        """构建指标卡片"""
        if not metrics:
            return "<p style='color:#8892a8;'>暂无指标数据</p>"

        def fmt(v, is_pct=True):
            if v is None or v == "-":
                return "-"
            try:
                v = float(v)
                if is_pct and abs(v) < 1:
                    return f"{v * 100:.1f}%"
                return f"{v:.2f}"
            except (ValueError, TypeError):
                return str(v)

        card_config = [
            ("年化收益", metrics.get("annual_return", "-"), "green"),
            ("胜率", metrics.get("win_rate", "-"), "blue"),
            ("最大回撤", metrics.get("max_drawdown", "-"), "red"),
            ("夏普比率", metrics.get("sharpe_ratio", "-"), "yellow"),
            ("总收益", metrics.get("total_return", "-"), "green"),
            ("盈亏比", metrics.get("profit_factor", "-"), "blue"),
            ("卡玛比率", metrics.get("calmar_ratio", "-"), "yellow"),
            ("年化波动率", metrics.get("volatility", "-"), "red"),
        ]

        cards = ""
        for label, value, color_class in card_config:
            display = fmt(value)
            cards += f"""
<div class="metric-card">
  <div class="value {color_class}">{display}</div>
  <div class="label">{label}</div>
</div>"""

        return f'<div class="metrics-grid">{cards}</div>'

    def _build_factor_importance(self, importance: dict) -> str:
        """构建因子重要性表格"""
        sorted_items = sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        )[:20]

        rows = ""
        for rank, (name, score) in enumerate(sorted_items, 1):
            bar_width = score / sorted_items[0][1] * 100 if sorted_items else 0
            rows += f"""<tr>
<td>{rank}</td>
<td style="font-family:monospace;">{name}</td>
<td>
  <div style="display:flex;align-items:center;gap:8px;">
    <div style="flex:1;height:6px;background:#1a2332;border-radius:3px;">
      <div style="height:100%;width:{bar_width:.0f}%;background:linear-gradient(90deg,#f59e0b,#ef4444);
           border-radius:3px;"></div>
    </div>
    <span style="color:#8892a8;font-size:11px;">{score:.4f}</span>
  </div>
</td>
</tr>"""

        return f"""<h2>🔬 因子重要性 Top 20</h2>
<div style="overflow-x:auto;margin-bottom:20px;">
<table><thead><tr><th>#</th><th>因子名称</th><th>重要性得分</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""

    def _build_trades_table(self, trades: list) -> str:
        """构建交易记录表"""
        if not trades:
            return ""
        rows = ""
        for t in trades[:50]:
            action = t.get("action", "")
            action_class = "green" if action == "buy" else "red"
            rows += f"""<tr>
<td>{t.get("date", "")}</td>
<td>{t.get("stock_code", "")}</td>
<td class="{action_class}">{action}</td>
<td>{t.get("price", 0):.2f}</td>
<td>{t.get("shares", 0)}</td>
<td>{t.get("value", 0):.0f}</td>
</tr>"""

        return f"""<h2>📋 交易记录（最近 50 笔）</h2>
<div style="overflow-x:auto;margin-bottom:20px;">
<table><thead><tr><th>日期</th><th>股票</th><th>操作</th><th>价格</th><th>数量</th><th>金额</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""

    def _build_risk_events(self, events: list) -> str:
        """构建风控事件列表"""
        if not events:
            return ""
        items = ""
        for e in events[-20:]:
            date = e.get("date", e.get("time", ""))
            rtype = e.get("action", e.get("type", ""))
            reason = e.get("reason", "")
            items += f"""<div style="background:#1e293b;border-left:3px solid #ef4444;
padding:8px 14px;margin:4px 0;border-radius:0 6px 6px 0;">
<strong style="font-size:13px;">[{rtype}]</strong>
<span style="color:#8892a8;font-size:12px;margin-left:8px;">{date}</span>
<span style="color:#e2e8f0;font-size:12px;margin-left:8px;">{reason}</span>
</div>"""

        return f"""<h2>⚠️ 风控事件（最近 20 条）</h2>
<div style="margin:12px 0 20px;">{items}</div>"""
