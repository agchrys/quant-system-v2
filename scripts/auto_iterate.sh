#!/bin/bash
# ============================================================
# Quant-System — 自动迭代优化脚本
# 每次迭代自动调参，直至达到目标（胜率≥60%, 年化≥20%, 回撤<12%）
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

MAX_ITERATIONS=50
IMPROVEMENT_THRESHOLD=0.01
BEST_METRICS_FILE="logs/best_metrics.json"

echo "=========================================="
echo " 自动迭代优化启动"
echo " 目标: 胜率≥60% | 年化≥20% | 回撤<12%"
echo "=========================================="

for i in $(seq 1 $MAX_ITERATIONS); do
    echo ""
    echo "══════════════════════════════════════════"
    echo "  迭代 #${i}"
    echo "══════════════════════════════════════════"
    
    # 运行一次完整策略
    python main.py --mode iterate
    
    # 读取当前指标
    if [ -f "$BEST_METRICS_FILE" ]; then
        WIN_RATE=$(python -c "import json; d=json.load(open('$BEST_METRICS_FILE')); print(d.get('win_rate', 0))")
        ANN_RET=$(python -c "import json; d=json.load(open('$BEST_METRICS_FILE')); print(d.get('annual_return', 0))")
        MAX_DD=$(python -c "import json; d=json.load(open('$BEST_METRICS_FILE')); print(d.get('max_drawdown', 1))")
        
        echo ""
        echo "  当前指标:"
        echo "    胜率:     $(python -c "print(f'{float($WIN_RATE)*100:.1f}%')")  (目标: 60%)"
        echo "    年化收益: $(python -c "print(f'{float($ANN_RET)*100:.1f}%')")  (目标: 20%)"
        echo "    最大回撤: $(python -c "print(f'{float($MAX_DD)*100:.1f}%')")  (目标: <12%)"
        
        # 检查是否达标
        GOALS_MET=0
        python -c "
w = float($WIN_RATE)
a = float($ANN_RET)
m = float($MAX_DD)
if w >= 0.60 and a >= 0.20 and m < 0.12:
    exit(0)
else:
    exit(1)
" && GOALS_MET=1 || GOALS_MET=0
        
        if [ "$GOALS_MET" -eq 1 ]; then
            echo ""
            echo "  🎉 所有目标已达成！停止迭代。"
            break
        fi
        
        # 检查是否有实质改进
        python -c "
import json, os
best = json.load(open('$BEST_METRICS_FILE'))
# 简化检查：只要在继续就继续
" 2>/dev/null || true
        
        echo ""
        echo "  继续下一轮迭代..."
    else
        echo "  ⚠  未检测到指标文件，继续..."
    fi
done

echo ""
echo "=========================================="
echo " 迭代完成"
echo " 总计迭代: ${i} 次"
echo " 最终报告: logs/performance_report_latest.html"
echo "=========================================="
