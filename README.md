# Quant-System v2 — 策略插件版

> **个人版量化交易系统** · 策略插件架构 + 五层框架
>
> 在 Quant-System 基础上扩展了策略插件系统。所有策略以插件形式独立注册，
> 通过 `--strategy` 参数选择，新增策略只需在 `strategies/` 目录下添加文件。

## 🎯 已注册策略

```bash
# 查看所有可用策略
python main.py --mode list-strategies

# 运行 v8.15 策略（2024-2026 年化 20.66%）
python main.py --strategy v8_15
```

| 策略 | 版本 | 胜率 | 年化 | 描述 |
|------|------|------|------|------|
| v8_15 | 8.15 | 51.55% | 20.66% | 信号持续性+量能确认+ATR动态止损 |

## 🧩 策略插件架构

```
strategies/
  __init__.py    # StrategyBase + StrategyRegistry
  base.py        # 策略抽象基类
  v8_15.py       # v8.15 策略插件实现
  <your>.py      # 你的新策略（自动发现）
```

**创建新策略：**
```python
from strategies import StrategyBase, StrategyRegistry

@StrategyRegistry.register
class MyStrategy(StrategyBase):
    name = "my_strategy"
    version = "1.0"
    description = "描述"
    
    def run_pipeline(self, config, **kwargs):
        # 实现完整流水线
        return result
```

策略自动发现，零配置注册。

## 🏗️ 系统架构

```
┌────────────────────────────────────────────┐
│  Layer 5: 策略执行层                        │
│  (backtest/engine.py + scripts/)            │
├────────────────────────────────────────────┤
│  Layer 4: 组合管理层                        │
│  (portfolio/optimizer.py + risk_control.py) │
├────────────────────────────────────────────┤
│  Layer 3: 信号生成层                        │
│  (models/trainer.py + predictor.py +        │
│   market_timer.py)                          │
├────────────────────────────────────────────┤
│  Layer 2: 因子工程层                        │
│  (factors/traditional_factors.py +          │
│   factor_pipeline.py)                       │
├────────────────────────────────────────────┤
│  Layer 1: 数据基础设施层                    │
│  (data/collector.py + processor.py)         │
└────────────────────────────────────────────┘
```

## 🎯 目标

| 指标 | 目标 |
|------|------|
| 胜率 (Win Rate) | ≥ 60% |
| 年化收益率 (Annual Return) | ≥ 20% |
| 最大回撤 (Max Drawdown) | < 12% |
| 夏普比率 (Sharpe Ratio) | > 1.5 |

## 🚀 快速开始

### 1. 安装依赖

```bash
cd quant-system
pip install -r requirements.txt
```

### 2. 运行完整流程

一键运行数据采集、因子计算、模型训练、回测、报告生成：

```bash
python main.py --mode full
```

### 3. 自动迭代优化

系统会自动搜索最优参数，多轮迭代直至达到目标：

```bash
python main.py --mode iterate
```

或者使用脚本：

```bash
bash scripts/run_pipeline.sh       # 单次运行
bash scripts/auto_iterate.sh       # 自动迭代优化
```

### 4. 查看报告

在 `logs/` 目录下查看生成的 HTML 性能报告。

## 📁 项目结构

```
quant-system/
├── main.py                       # 主入口（全流程控制）
├── config/
│   └── config.yaml               # 全局配置
├── data/
│   ├── collector.py              # 数据采集（AKShare）
│   ├── processor.py              # 数据清洗与对齐
│   └── data_utils.py             # 数据工具函数
├── factors/
│   ├── traditional_factors.py    # 传统因子计算（20+ 因子）
│   └── factor_pipeline.py        # 因子处理流水线
├── models/
│   ├── trainer.py                # LightGBM 模型训练
│   ├── predictor.py              # 预测推理
│   └── market_timer.py           # 市场状态判断
├── portfolio/
│   ├── optimizer.py              # 组合优化（风险平价等）
│   └── risk_control.py           # 多层级风控
├── backtest/
│   ├── engine.py                 # 事件驱动回测引擎
│   ├── metrics.py                # 性能指标计算
│   └── result.py                 # 回测结果数据类
├── iteration/
│   ├── version_manager.py        # 版本管理
│   ├── optimizer.py              # 自动参数搜索
│   └── reporter.py               # 可视化报告生成
├── scripts/
│   ├── run_pipeline.sh           # 一键运行脚本
│   └── auto_iterate.sh           # 自动迭代脚本
├── logs/                         # 日志与报告
├── docs/                         # 文档
└── requirements.txt              # Python 依赖
```

## 💡 各层详解

### Layer 1: 数据基础设施

数据源使用 **AKShare**（免费开源 A 股数据接口），采集内容包括：

- **日线行情**：开高低收、成交量、成交额、换手率
- **财务数据**：营收、净利润、ROE、PE、PB
- **资金流数据**：主力资金流向（可选）
- **指数数据**：沪深300、中证500 等

数据清洗包括：复权处理、缺失值填充、停牌处理、时间对齐。

> 首次运行会自动创建 `data/cache/` 目录缓存数据，后续运行会自动增量更新。

### Layer 2: 因子工程

**传统因子（20+ 个核心因子）：**

| 类别 | 因子 | 说明 |
|------|------|------|
| 动量 | momentum_1m/3m/6m | 过去 N 日收益率 |
| 反转 | reversal_5d | 短期反转效应 |
| 均线 | ma_deviation | 收盘价偏离均线程度 |
| 波动率 | volatility_20d/60d | 历史波动率 |
| 流动性 | turnover_rate/change | 换手率及其变化 |
| 量比 | volume_ratio | 近期成交量相对水平 |
| 估值 | pe_rank, pb_rank | PE/PB 历史分位数 |
| 质量 | roe_factor | 净资产收益率 |
| 成长 | revenue_growth, profit_growth | 营收/利润增速 |
| 非流动性 | amihud | Amihud 指标 |

**因子处理流水线：**
1. 去极值（MAD 或分位数法）
2. 标准化（Z-score 或 Rank 排序）
3. 行业/市值中性化
4. 缺失值填充

### Layer 3: 信号生成

**模型：LightGBM Lambdarank**
- 排序学习任务，直接优化 NDCG 指标
- 支持早停机制，防止过拟合
- 自动特征重要性排序

**市场择时模块：**
- 基于技术规则判断市场状态（趋势上涨/震荡/趋势下跌/高波动）
- 不同状态动态调整仓位上限

### Layer 4: 组合管理

**仓位分配：**
- 风险平价 (Risk Parity) 或均值-方差优化
- 单股权重 ≤ 10%，单行业 ≤ 30%
- 定期再平衡（按周/月）

**多层级风控：**

```
第一层：个股止损         亏损 3% 止损，N 日不涨减仓
第二层：组合回撤控制     回撤 5%/7%/10% 依次降仓
第三层：波动率目标       超标自动降杠杆
第四层：大模型预警       (预留) 新闻舆情检测
```

### Layer 5: 回测与评估

**回测引擎特性：**
- 事件驱动，按日推进
- 支持滑点、佣金、印花税模拟
- A 股交易规则（T+1、整百股交易）

**性能指标：**
- 年化收益率、胜率
- 最大回撤（含峰值/谷值/恢复日期）
- 夏普比率、卡玛比率、索提诺比率
- 盈亏比、月度收益矩阵
- 滚动指标分析

## 🔄 自动迭代机制

系统内置了自动迭代优化流程：

```
初始化 → 数据采集 → 因子计算 → 模型训练
  ↓                              ↓
  参数搜索 ← 回测评估 ← 组合优化 ← 信号生成
  ↓
  达到目标？→ 是 → 保存版本 + 生成报告
  ↓ 否
  继续下一轮迭代
```

**迭代参数：**
- 网格搜索或随机搜索
- 参数空间涵盖学习率、树深度、采样率、正则化等 8 个维度
- 早停机制：连续 3 次无改善自动终止

## 📊 版本管理

- 每次迭代自动创建新版本（v{major}.{minor}.{patch}）
- 保存模型、配置、性能指标
- 支持版本回滚和对比

```bash
# 查看版本列表
from iteration.version_manager import VersionManager
vm = VersionManager('./')
vm.list_versions()

# 版本对比
vm.compare_versions('v1.0.0', 'v1.1.0')

# 回滚
vm.rollback('v1.0.0')
```

## 🔧 自定义配置

编辑 `config/config.yaml` 可调整：

- **选股池**：沪深300 / 中证500 / 全A
- **回测周期**：训练/验证/测试区间
- **因子权重**：各组因子的配比
- **模型参数**：LightGBM 超参数
- **风控阈值**：止损线、回撤控制线
- **迭代参数**：搜索空间、目标阈值

## 📈 个人投资者低配方案

如果不使用 ML 模型，可使用简化规则策略：

```
选股规则：
  1. PE 处于历史 20%-80% 分位数
  2. ROE > 10%
  3. 过去 20 日收益率 > 0 (动量筛选)
  4. 换手率 > 市场平均 (流动性)

择时规则：
  沪深 300 在 20 日均线上方 → 做多
  沪深 300 在 20 日均线下方 → 空仓

风控：
  单股止损 5%，总回撤 10% 清仓

预期：年化 10-18%，最大回撤 15-25%
```

## ❓ 常见问题

**Q: 需要 AKShare token 吗？**
A: 不需要，AKShare 是免费开源接口。

**Q: 回测时间多长？**
A: 首次运行约 10-30 分钟（取决于股票数量和历史长度），后续使用缓存会快很多。

**Q: 支持实盘吗？**
A: 本系统主要用于研究和回测。如需实盘，可对接 QMT/Ptrade 等券商接口。

**Q: 能跑在 Windows 上吗？**
A: 可以，但建议使用 Linux/Mac 获得更好的性能。

## 📚 引用

- [AKShare](https://github.com/akfamily/akshare) - 免费开源 A 股数据接口
- [LightGBM](https://github.com/microsoft/LightGBM) - 梯度提升框架
- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) - 组合优化库

---

> **免责声明**: 本项目仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。
