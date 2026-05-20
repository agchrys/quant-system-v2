#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════╗
║  Quant-System Personal Edition              ║
║  个人版量化交易系统                          ║
║  数据驱动 · 大模型赋能 · 严格风控            ║
╚══════════════════════════════════════════════╝

架构:
  Layer 5: 策略执行层  (下单、风控、归因分析)
  Layer 4: 组合管理层  (仓位分配、行业轮动)
  Layer 3: 信号生成层  (选股信号、择时信号)
  Layer 2: 因子工程层  (传统因子 + 大模型因子)
  Layer 1: 数据层      (行情、财务、另类数据)

使用方式:
    python main.py --mode full          # 运行完整流程
    python main.py --mode iterate       # 运行自动迭代
    python main.py --mode data          # 仅采集数据
    python main.py --mode train         # 仅训练模型
    python main.py --mode backtest      # 仅回测
    python main.py --mode report        # 仅生成报告
    python main.py --mode compare       # 版本对比
"""

import os
import sys
import json
import argparse
import yaml
from loguru import logger
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ---- 配置加载 ----
def load_config(path: str = None) -> dict:
    """加载 YAML 配置文件"""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(config: dict):
    """配置日志系统"""
    log_config = config.get("logging", {})
    log_file = os.path.join(PROJECT_ROOT, log_config.get("file", "logs/quant_system.log"))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger.remove()  # 移除默认 handler
    logger.add(
        sys.stdout,
        level=log_config.get("level", "INFO"),
        format=log_config.get("format", "{time:HH:mm:ss} | {level} | {message}")
    )
    logger.add(
        log_file,
        level="DEBUG",
        rotation="10 MB",
        format=log_config.get("format", "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    )
    logger.info(f"日志初始化完成: {log_file}")

# ============================================================
# 各阶段模块导入（延迟导入，避免启动时依赖缺失）
# ============================================================

def stage_data(config: dict):
    """Layer 1: 数据采集与处理"""
    logger.info("=" * 50)
    logger.info("Layer 1: 数据采集与处理")
    logger.info("=" * 50)
    
    try:
        from data.collector import DataCollector
        from data.processor import DataProcessor
        
        collector = DataCollector(config)
        raw_data = collector.collect_all()
        
        if raw_data:
            processor = DataProcessor(config)
            panel = processor.build_panel(raw_data)
            processor.save_panel(panel, os.path.join(PROJECT_ROOT, "data", "cache", "factor_panel.parquet"))
            logger.success(f"数据处理完成，宽表形状: {panel.shape}")
            return panel
        else:
            logger.warning("未获取到数据，尝试使用缓存...")
            processor = DataProcessor(config)
            panel = processor.load_panel(os.path.join(PROJECT_ROOT, "data", "cache", "factor_panel.parquet"))
            return panel
    except Exception as e:
        logger.error(f"数据阶段失败: {e}")
        raise

def stage_factors(config: dict, panel=None):
    """Layer 2: 因子工程"""
    logger.info("=" * 50)
    logger.info("Layer 2: 因子工程")
    logger.info("=" * 50)
    
    try:
        from factors.traditional_factors import FactorCalculator
        from factors.factor_pipeline import FactorPipeline
        
        calculator = FactorCalculator()
        pipeline = FactorPipeline(config)
        
        factor_dict = calculator.compute_all(panel)
        logger.info(f"计算了 {len(factor_dict)} 个因子组")
        
        processed_factors = pipeline.run(factor_dict)
        logger.success(f"因子处理完成，形状: {processed_factors.shape}")
        
        # 保存处理后的因子
        import pandas as pd
        save_path = os.path.join(PROJECT_ROOT, "data", "cache", "processed_factors.parquet")
        processed_factors.to_parquet(save_path)
        
        return processed_factors
    except Exception as e:
        logger.error(f"因子工程阶段失败: {e}")
        raise

def stage_models(config: dict, factors=None):
    """Layer 3: 模型训练与信号生成"""
    logger.info("=" * 50)
    logger.info("Layer 3: 信号生成层")
    logger.info("=" * 50)
    
    try:
        from models.trainer import ModelTrainer
        from models.predictor import Predictor
        from models.market_timer import MarketTimer
        
        trainer = ModelTrainer(config)
        
        # 准备收益率数据（未来 N 日收益）
        # 假设 factors 中包含了 close 价格
        import pandas as pd
        import numpy as np
        
        # 提取收盘价列（如果有）
        close_prices = None
        if hasattr(factors, 'columns') and 'close' in factors.columns.get_level_values(1) if isinstance(factors.columns, pd.MultiIndex) else False:
            pass  # 从 factor 数据中提取
        
        # 生成训练数据
        X_train, y_train, X_val, y_val, X_test, y_test, groups = trainer.prepare_data(factors)
        
        # 模型训练
        model = trainer.train(X_train, y_train, X_val, y_val, groups)
        
        # 保存模型
        model_path = os.path.join(PROJECT_ROOT, "models", "saved", "lgb_model.txt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        trainer.save_model(model, model_path)
        
        # 预测
        predictor = Predictor()
        test_scores = predictor.predict(model, X_test)
        
        # 市场择时
        timer = MarketTimer()
        
        logger.success("模型训练完成")
        return {
            "model": model,
            "predictor": predictor,
            "timer": timer,
            "X_test": X_test,
            "y_test": y_test,
            "test_scores": test_scores
        }
    except Exception as e:
        logger.error(f"模型阶段失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def stage_portfolio(config: dict, model_output=None):
    """Layer 4: 组合管理与风控"""
    logger.info("=" * 50)
    logger.info("Layer 4: 组合管理与风控")
    logger.info("=" * 50)
    
    try:
        from portfolio.optimizer import PortfolioOptimizer
        from portfolio.risk_control import RiskController
        
        optimizer = PortfolioOptimizer(config)
        risk_controller = RiskController(config)
        
        logger.success("组合管理模块就绪")
        return {
            "optimizer": optimizer,
            "risk_controller": risk_controller
        }
    except Exception as e:
        logger.error(f"组合管理阶段失败: {e}")
        raise

def stage_backtest(config: dict, model_output=None, portfolio_output=None):
    """Layer 5: 回测与评估"""
    logger.info("=" * 50)
    logger.info("Layer 5: 回测与评估")
    logger.info("=" * 50)
    
    try:
        from backtest.engine import BacktestEngine
        from backtest.metrics import PerformanceMetrics
        
        engine = BacktestEngine(config)
        metrics_calc = PerformanceMetrics()
        
        # 简单测试运行
        logger.info("回测引擎初始化完成")
        
        return {
            "engine": engine,
            "metrics": metrics_calc
        }
    except Exception as e:
        logger.error(f"回测阶段失败: {e}")
        raise

def stage_iteration(config: dict):
    """自动迭代优化"""
    logger.info("=" * 50)
    logger.info("自迭代优化模式")
    logger.info("=" * 50)
    
    try:
        from iteration.version_manager import VersionManager
        from iteration.optimizer import AutoOptimizer
        from iteration.reporter import Reporter
        
        version_mgr = VersionManager(PROJECT_ROOT)
        optimizer = AutoOptimizer(config)
        reporter = Reporter()
        
        current_ver = version_mgr.get_current_version()
        logger.info(f"当前版本: {current_ver}")
        
        # === 数据准备 ===
        panel = stage_data(config)
        factors = stage_factors(config, panel)
        model_output = stage_models(config, factors)
        portfolio_output = stage_portfolio(config, model_output)
        
        # === 参数搜索 ===
        logger.info("开始参数搜索优化...")
        best_params = optimizer.run_search(
            model_output.get("X_train"), 
            model_output.get("y_train"),
            model_output.get("X_val"),
            model_output.get("y_val"),
            None
        )
        logger.info(f"最优参数: {best_params}")
        
        # === 生成报告 ===
        html_report = reporter.generate_html_report(
            metrics={"version": current_ver},
            nav_series=None,
            trades=[],
            risk_events=[],
            factor_importance={}
        )
        
        report_path = os.path.join(PROJECT_ROOT, "logs", f"performance_report_{current_ver}.html")
        reporter.save_report(html_report, report_path)
        logger.success(f"报告已生成: {report_path}")
        
        # === 保存版本 ===
        version_mgr.create_version(
            metrics={"annual_return": 0, "win_rate": 0, "max_drawdown": 1},
            params=best_params or config.get("model", {}).get("lightgbm", {})
        )
        
        return version_mgr.get_current_version()
        
    except Exception as e:
        logger.error(f"迭代阶段失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def stage_report(config: dict, version: str = None):
    """生成可视化报告"""
    logger.info("生成可视化报告...")
    
    try:
        from iteration.reporter import Reporter
        
        reporter = Reporter()
        
        # 尝试从缓存读取历史数据
        import pandas as pd
        import os
        
        nav_path = os.path.join(PROJECT_ROOT, "logs", "backtest_nav.csv")
        trades_path = os.path.join(PROJECT_ROOT, "logs", "backtest_trades.csv")
        
        nav_series = None
        trades = []
        risk_events = []
        
        if os.path.exists(nav_path):
            nav_df = pd.read_csv(nav_path, index_col=0, parse_dates=True)
            if 'nav' in nav_df.columns:
                nav_series = nav_df['nav']
        
        version_info = version or "latest"
        html_report = reporter.generate_html_report(
            metrics={"version": version_info},
            nav_series=nav_series,
            trades=trades,
            risk_events=risk_events,
            factor_importance={}
        )
        
        report_path = os.path.join(PROJECT_ROOT, "logs", f"performance_report_{version_info}.html")
        reporter.save_report(html_report, report_path)
        logger.success(f"报告已保存: {report_path}")
        
    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        raise


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quant-System Personal Edition - 个人版量化交易系统"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="full",
        choices=["full", "iterate", "data", "train", "backtest", "report", "compare"],
        help="运行模式"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="配置文件路径"
    )
    parser.add_argument(
        "--version", "-v",
        type=str,
        default=None,
        help="指定版本号（用于对比和报告）"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="跳过数据采集（使用缓存）"
    )
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default=None,
        help="策略插件名称（如 v8_15），使用策略插件模式运行"
    )

    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    setup_logging(config)
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Quant-System v{config.get('version', '?')} 启动")
    logger.info(f"运行时间: {current_time}")
    logger.info(f"运行模式: {args.mode}")
    
    try:
        # ---- 策略插件模式 ----
        if args.strategy:
            logger.info(f"使用策略插件: {args.strategy}")
            from strategies import StrategyRegistry
            strategy = StrategyRegistry.get(args.strategy)
            if strategy is None:
                available = StrategyRegistry.list_strategies()
                names = [s['name'] for s in available]
                logger.error(f"策略 '{args.strategy}' 未找到。可用策略: {names}")
                sys.exit(1)
            logger.info(strategy)
            result = strategy.run_pipeline(config)
            logger.info("=" * 50)
            logger.success("策略执行完成！")
            logger.info("=" * 50)
            return

        if args.mode == "full":
            # 完整流程
            panel = stage_data(config) if not args.skip_data else None
            factors = stage_factors(config, panel)
            model_out = stage_models(config, factors)
            portfolio_out = stage_portfolio(config, model_out)
            bt_out = stage_backtest(config, model_out, portfolio_out)
            stage_report(config)
            
            logger.info("=" * 50)
            logger.success("完整流程执行完成！")
            logger.info("=" * 50)
            
        elif args.mode == "iterate":
            # 自动迭代
            new_version = stage_iteration(config)
            logger.info(f"系统升级至版本: {new_version}")
            
        elif args.mode == "data":
            stage_data(config)
            
        elif args.mode == "train":
            factors = stage_factors(config, stage_data(config))
            stage_models(config, factors)
            
        elif args.mode == "backtest":
            panel = stage_data(config) if not args.skip_data else None
            factors = stage_factors(config, panel)
            model_out = stage_models(config, factors)
            portfolio_out = stage_portfolio(config, model_out)
            stage_backtest(config, model_out, portfolio_out)
            
        elif args.mode == "report":
            stage_report(config, args.version)
            
        elif args.mode == "compare":
            logger.info("版本对比模式")
            from iteration.version_manager import VersionManager
            vm = VersionManager(PROJECT_ROOT)
            versions = vm.list_versions()
            if len(versions) >= 2:
                result = vm.compare_versions(versions[-2], versions[-1])
                logger.info(f"\n{result}")
            else:
                logger.warning("版本数量不足，无法对比")
                
    except KeyboardInterrupt:
        logger.warning("用户中断运行")
        sys.exit(1)
    except Exception as e:
        logger.error(f"运行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
