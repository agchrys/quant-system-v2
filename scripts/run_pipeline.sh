#!/bin/bash
# ============================================================
# Quant-System — 全流程运行脚本
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo " Quant-System Personal Edition"
echo " 版本: $(grep '^version:' config/config.yaml | awk '{print $2}')"
echo "=========================================="

# 1. 检查依赖
echo ""
echo "[1/4] 检查依赖..."
python -c "import lightgbm; import akshare; import pandas" 2>/dev/null || {
    echo "  ⚠  部分依赖缺失，执行安装..."
    pip install -r requirements.txt -q
}

# 2. 数据采集
echo ""
echo "[2/4] 数据采集与处理..."
python -c "
from data.collector import DataCollector
c = DataCollector()
c.collect_all()
print('  ✓ 数据采集完成')
"

# 3. 运行完整流程（训练 + 回测 + 报告）
echo ""
echo "[3/4] 运行完整策略流程..."
python main.py --mode full

# 4. 生成报告
echo ""
echo "[4/4] 迭代报告已生成至 logs/ 目录"
echo ""
echo "=========================================="
echo " 运行完成!"
echo " 报告: logs/performance_report_latest.html"
echo "=========================================="
