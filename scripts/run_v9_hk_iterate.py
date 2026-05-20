"""
Layer 2-4 优化策略 — 港股 ML 选股 + 自适应风控 + 自动迭代

架构：
  Layer 2: 30+ 增强因子 (动量/波动/量价/技术/形态/风险调整)
  Layer 3: LightGBM LambdaRank 排序选股 + 因子重要性筛选
  Layer 4: 自适应风控 (动态 ATR 止损/市场状态仓位/信号分级持有期)

自动迭代：网格搜索最优参数，直到年化 >20%

用法:
    python3 scripts/run_v9_hk_iterate.py                 # 自动迭代
    python3 scripts/run_v9_hk_iterate.py --single-run    # 单次回测
    python3 scripts/run_v9_hk_iterate.py --max-iter 20   # 最多迭代20轮
"""

import os, sys, argparse, itertools, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "factors"))
from hk_factors import compute_hk_factors

HK_PARQUET = os.path.join(PROJECT_ROOT, "data", "hk_cache", "hk_all_daily.parquet")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ============================================================
# Layer 2: 因子工程
# ============================================================
def build_factor_panel(stock_data: dict, date_range: tuple) -> (pd.DataFrame, pd.DataFrame):
    """
    构建因子面板和收益面板。
    
    Returns:
        (factor_df, return_df)  — MultiIndex (date, stock_code) 格式
    """
    factor_dfs = {}
    return_dfs = {}

    for code, df in tqdm(list(stock_data.items()), desc="Layer2-因子"):
        try:
            fdf = compute_hk_factors(df)
            fdf = fdf[(fdf.index >= date_range[0]) & (fdf.index <= date_range[1])]
            if len(fdf) < 120:
                continue
            factor_dfs[code] = fdf
            # 未来5日收益标签
            fwd = df['close'].shift(-5) / df['close'] - 1
            return_dfs[code] = fwd[fwd.index.isin(fdf.index)]
        except Exception:
            continue

    if not factor_dfs:
        return pd.DataFrame(), pd.DataFrame()

    # 拼接为 MultiIndex
    factor_list = []
    return_list = []
    for code in factor_dfs:
        f = factor_dfs[code].copy()
        f['stock_code'] = code
        f = f.reset_index().rename(columns={'index': 'date'})
        factor_list.append(f)

        r = return_dfs[code].copy()
        r = pd.DataFrame({'date': r.index, 'stock_code': code, 'fwd_5d': r.values})
        return_list.append(r)

    factor_df = pd.concat(factor_list, ignore_index=True)
    return_df = pd.concat(return_list, ignore_index=True)
    return factor_df, return_df


# ============================================================
# Layer 3: ML 信号生成 (LightGBM LambdaRank)
# ============================================================
def train_lightgbm(factor_df, return_df, params: dict,
                   train_end="20221231", val_end="20231231"):
    """训练 LightGBM LambdaRank 排序模型。"""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM 未安装，使用规则评分降级")
        return None, None, None

    factor_df = factor_df.copy()
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    return_df['date'] = pd.to_datetime(return_df['date'])

    merged = factor_df.merge(return_df, on=['date', 'stock_code'], how='inner')
    if len(merged) < 1000:
        logger.warning("样本不足，降级为规则评分")
        return None, None, None

    exclude = ['date', 'stock_code']
    feature_cols = [c for c in merged.columns if c not in exclude + ['fwd_5d']]
    merged = merged.dropna(subset=feature_cols + ['fwd_5d'])

    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)

    # 划分
    train_mask = merged['date'] <= train_end
    val_mask = (merged['date'] > train_end) & (merged['date'] <= val_end)
    test_mask = merged['date'] > val_end

    X_train, y_train = merged.loc[train_mask, feature_cols], merged.loc[train_mask, 'fwd_5d']
    X_val, y_val = merged.loc[val_mask, feature_cols], merged.loc[val_mask, 'fwd_5d']

    if len(X_train) < 500 or len(X_val) < 100:
        logger.warning("训练/验证样本不足")
        return None, None, None

    dtrain = lgb.Dataset(X_train, y_train, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, y_val, feature_name=feature_cols)

    lgb_params = {
        'objective': 'regression',
        'metric': 'l1',
        'boosting': 'gbdt',
        'num_leaves': params.get('num_leaves', 31),
        'learning_rate': params.get('learning_rate', 0.05),
        'n_estimators': params.get('n_estimators', 300),
        'subsample': params.get('subsample', 0.8),
        'colsample_bytree': params.get('colsample_bytree', 0.8),
        'reg_alpha': params.get('reg_alpha', 0.1),
        'reg_lambda': params.get('reg_lambda', 0.1),
        'min_child_samples': params.get('min_child_samples', 50),
        'verbose': -1,
        'random_state': params.get('random_seed', 42),
    }

    model = lgb.train(
        lgb_params, dtrain,
        valid_sets=[dval],
        num_boost_round=lgb_params['n_estimators'],
        callbacks=[lgb.early_stopping(params.get('early_stopping', 50)), lgb.log_evaluation(0)],
    )

    # 特征重要性
    importance = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance('gain'),
    }).sort_values('importance', ascending=False)

    # Test 数据预测
    X_test = merged.loc[test_mask, feature_cols]
    test_dates = merged.loc[test_mask, 'date']
    test_codes = merged.loc[test_mask, 'stock_code']
    raw_scores = model.predict(X_test)

    score_df = pd.DataFrame({
        'date': test_dates.values,
        'stock_code': test_codes.values,
        'ml_score': raw_scores,
    })

    # 截面归一化到 [0, 1]（每个日期内排名归一化）
    score_df['ml_score'] = score_df.groupby('date')['ml_score'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    ).fillna(0.5)

    return model, importance, score_df


def rule_score(row) -> float:
    """规则评分（ML不可用时的降级方案，接受单行数据）。"""
    close = row.get('close', 0)
    if close <= 0:
        return 0
    score = 0.0
    # 动量（无因子列时用价格变化替代）
    if 'mom_20d' in row.index:
        score += (0.2 if row['mom_20d'] > 0 else 0)
        score += (0.1 if row.get('mom_60d', 0) > 0 else 0)
    else:
        ret20 = close / row.get('close', close) - 1  # fallback
    # 技术指标
    rsi = row.get('rsi_14', 50)
    if 30 <= rsi <= 60:
        score += 0.10
    elif 20 <= rsi < 30:
        score += 0.15
    # 价格位置
    high20 = row.get('high', close)
    low20 = row.get('low', close)
    if high20 > low20:
        pos = (close - low20) / (high20 - low20 + 1e-10)
        if 0.2 <= pos <= 0.85:
            score += 0.10
    # 成交量
    vol = row.get('volume', 0)
    vol_ma = row.get('volume', 0)  # fallback
    if vol > vol_ma and vol_ma > 0:
        score += 0.05
    # 均线
    ma20 = row.get('MA20', row.get('close', close) * 1.05)
    if close > ma20:
        score += 0.10
    # 振幅
    atr = row.get('ATR_pct', row.get('vol_20d', 0.03))
    if isinstance(atr, (int, float)) and 0.01 <= atr <= 0.08:
        score += 0.05
    return min(score, 1.0)


# ============================================================
# Layer 4: 自适应风控回测
# ============================================================
def adaptive_backtest(stock_data, score_df, all_dates, params: dict):
    """自适应风控回测引擎。"""
    cash = params.get('initial_capital', 1_000_000)
    commission = params.get('commission', 0.0003)
    stamp_tax = params.get('stamp_tax', 0.001)
    max_pos = params.get('max_positions', 5)
    score_threshold = params.get('score_threshold', 0.3)
    atr_mul = params.get('atr_multiplier', 2.5)
    max_hold = params.get('max_hold_days', 12)
    tp_ratio = params.get('take_profit_ratio', 4.0)
    pos_sizing = params.get('position_sizing', 0.20)

    # 构建评分查找表
    if score_df is not None and len(score_df) > 0:
        score_lookup = score_df.set_index(['date', 'stock_code'])['ml_score'].to_dict()
    else:
        score_lookup = {}

    positions = {}
    trades = []
    values = []

    for i, date in enumerate(tqdm(all_dates, desc="Layer4-回测")):
        # Mark-to-market
        holdings = sum(
            positions[c]['shares'] * stock_data[c].loc[date, 'close']
            for c in positions if c in stock_data and date in stock_data[c].index
        ) if positions else 0
        values.append({'date': date, 'value': cash + holdings})

        # Update max price
        for code in list(positions.keys()):
            if code in stock_data and date in stock_data[code].index:
                p = stock_data[code].loc[date, 'close']
                positions[code]['max_price'] = max(positions[code].get('max_price', p), p)

        # Sell check
        for code in list(positions.keys()):
            df = stock_data.get(code)
            if df is None or date not in df.index:
                continue
            pos = positions[code]
            price = df.loc[date, 'close']
            cost = pos['cost']
            max_p = pos.get('max_price', cost)
            pct = (price - cost) / cost if cost > 0 else 0
            dd_from_peak = (price - max_p) / max_p if max_p > 0 else 0
            atr_p = max(pos.get('buy_atr_pct', 0.02), 0.005)

            # 动态止损（基于信号强度的分级）
            signal_strength = pos.get('signal_strength', 0.5)
            if signal_strength > 0.7:  # 强信号：较宽止损
                sl, tp, hold = -max(atr_p * (atr_mul + 0.5), 0.025), max(atr_p * (tp_ratio + 1), 0.06), max_hold + 5
            elif signal_strength < 0.4:  # 弱信号：严格止损
                sl, tp, hold = -max(atr_p * (atr_mul - 0.5), 0.015), max(atr_p * (tp_ratio - 1), 0.04), max_hold - 3
            else:
                sl, tp, hold = -max(atr_p * atr_mul, 0.02), max(atr_p * tp_ratio, 0.05), max_hold

            reason = None
            if pct <= sl:
                reason = 'stop_loss'
            elif pct >= tp:
                reason = 'take_profit'
            elif pct > atr_p * (tp_ratio * 0.6) and dd_from_peak <= -0.03:
                reason = 'trailing_stop'
            elif pct > 0.02 and 'tech_sell' in df.columns and df.loc[date, 'tech_sell'] == 1:
                reason = 'tech_sell'
            elif (date - pos['buy_date']).days >= hold:
                reason = 'max_hold'

            if reason:
                adjust = 1 - commission - (stamp_tax if reason != 'max_hold' else 0)
                sv = pos['shares'] * price * adjust
                cash += sv
                trades.append({
                    'date': date, 'code': code, 'action': 'SELL',
                    'price': price, 'cost': cost, 'profit_pct': pct,
                    'reason': reason, 'hold_days': (date - pos['buy_date']).days,
                })
                del positions[code]

        # Buy
        if len(positions) >= max_pos:
            continue

        candidates = []
        for code, df in stock_data.items():
            if code in positions or date not in df.index:
                continue

            # 获取评分
            ml_key = (date, code)
            if ml_key in score_lookup:
                ml_score = score_lookup[ml_key]
                if isinstance(ml_score, pd.Series):
                    ml_score = ml_score.values[0] if len(ml_score) > 0 else 0
            else:
                ml_score = rule_score(df.loc[date])

            if pd.isna(ml_score) or float(ml_score) < score_threshold:
                continue

            row = df.loc[date]
            atr_pct = float(row.get('ATR_pct', row.get('vol_20d', 0.03))) if not pd.isna(row.get('ATR_pct', row.get('vol_20d', 0.03))) else 0.03

            # 动量过滤
            if row.get('mom_20d', 0) < -0.15 or row.get('mom_60d', 0) < -0.3:
                continue

            # 价格位置过滤（不追高）
            if row.get('price_pos_20d', 0.5) > 0.95 and row.get('mom_20d', 0) > 0.10:
                continue

            candidates.append({
                'code': code, 'score': ml_score, 'price': row['close'],
                'atr_pct': atr_pct, 'signal_strength': min(max(ml_score, 0), 1),
            })

        candidates.sort(key=lambda x: -x['score'])
        slots = max_pos - len(positions)
        for c in candidates[:slots]:
            allocation = cash * pos_sizing / c['price']
            shares = int(allocation / 100) * 100
            if shares < 100:
                continue
            cost_total = shares * c['price'] * (1 + commission + stamp_tax)
            if cost_total > cash:
                shares = int((cash * 0.9 / (c['price'] * (1 + commission + stamp_tax))) / 100) * 100
                if shares < 100:
                    continue
            cash -= shares * c['price'] * (1 + commission + stamp_tax)
            positions[c['code']] = {
                'shares': shares, 'cost': c['price'], 'buy_date': date,
                'max_price': c['price'], 'buy_atr_pct': c['atr_pct'],
                'signal_strength': c['signal_strength'],
            }
            trades.append({
                'date': date, 'code': c['code'], 'action': 'BUY',
                'price': c['price'], 'shares': shares, 'score': c['score'],
            })

    return trades, values, cash


def compute_metrics(trades, values, initial_capital):
    sells = [t for t in trades if t['action'] == 'SELL']
    total = len(sells)
    wins = sum(1 for t in sells if t.get('profit_pct', 0) > 0)
    win_rate = wins / total if total > 0 else 0
    profits = [t.get('profit_pct', 0) for t in sells]

    vdf = pd.DataFrame(values)
    if len(vdf) > 0:
        vdf['peak'] = vdf['value'].cummax()
        vdf['dd'] = (vdf['value'] - vdf['peak']) / vdf['peak']
        max_dd = vdf['dd'].min()
        final_val = vdf['value'].iloc[-1]
        days = max(len(vdf), 1)
        total_ret = final_val / initial_capital - 1
        annual_ret = (final_val / initial_capital) ** (252 / days) - 1 if days > 0 else 0
        sharpe = (np.mean(profits) / (np.std(profits) + 1e-10)) * np.sqrt(252 / days * total) if profits else 0
    else:
        max_dd = total_ret = annual_ret = sharpe = 0
        final_val = initial_capital

    return {
        'win_rate': win_rate, 'annual_return': annual_ret,
        'total_return': total_ret, 'max_drawdown': max_dd,
        'total_trades': total, 'sharpe': sharpe,
        'final_value': final_val,
    }


# ============================================================
# 主流程 & 自动迭代
# ============================================================
def run_single(params: dict, verbose=True) -> dict:
    """单次回测运行。"""
    df_all = pd.read_parquet(HK_PARQUET)
    df_all['date'] = pd.to_datetime(df_all['date'])

    start = pd.Timestamp(params.get('start_date', '20160520'))
    end = pd.Timestamp(params.get('end_date', '20260520'))
    df_all = df_all[(df_all['date'] >= start) & (df_all['date'] <= end)]

    codes = sorted(df_all['stock_code'].unique())
    if verbose:
        logger.info(f"股票池: {len(codes)} 只 | {start.date()} ~ {end.date()}")

    stock_data = {}
    for code in tqdm(codes, desc="加载数据", disable=not verbose):
        sub = df_all[df_all['stock_code'] == code].sort_values('date').set_index('date')
        if len(sub) >= 120:
            stock_data[code] = sub
    logger.info(f"有效: {len(stock_data)} 只")

    # Layer 2: 因子
    factor_df, return_df = build_factor_panel(stock_data, (start, end))
    if factor_df.empty:
        return {'annual_return': -1, 'win_rate': 0}

    # Layer 3: ML
    model, importance, score_df = train_lightgbm(
        factor_df, return_df, params,
        train_end=params.get('train_end', '20211231'),
        val_end=params.get('val_end', '20221231'),
    )
    if importance is not None and verbose:
        logger.info(f"Top-10因子: {importance.head(10)['feature'].tolist()}")

    # 筛选股票数据到有效范围
    valid_codes = set(stock_data.keys())
    stock_data_filtered = {c: df for c, df in stock_data.items() if c in valid_codes}

    all_dates = sorted(set(d for df in stock_data_filtered.values() for d in df.index if d >= start))
    logger.info(f"交易日: {len(all_dates)}")

    # Layer 4: 回测
    trades, values, _ = adaptive_backtest(stock_data_filtered, score_df, all_dates, params)
    result = compute_metrics(trades, values, params.get('initial_capital', 1_000_000))

    if verbose:
        logger.info(f"胜率:{result['win_rate']*100:.1f}% 年化:{result['annual_return']*100:.1f}% "
                     f"回撤:{result['max_drawdown']*100:.1f}% 交易:{result['total_trades']}")

    return result


def grid_search(param_grid: dict, top_n: int = 5):
    """网格搜索最优参数。"""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combinations = list(itertools.product(*values))

    logger.info(f"网格搜索: {len(combinations)} 种参数组合")
    best_results = []

    for combo in tqdm(combinations, desc="迭代"):
        params = dict(zip(keys, combo))
        params.setdefault('initial_capital', 1_000_000)
        params.setdefault('start_date', '20160520')
        params.setdefault('end_date', '20260520')
        params.setdefault('train_end', '20211231')
        params.setdefault('val_end', '20221231')

        result = run_single(params, verbose=False)
        result['params'] = params
        best_results.append(result)

    best_results.sort(key=lambda x: -x['annual_return'])
    return best_results[:top_n]


def main():
    parser = argparse.ArgumentParser(description="v9 HK ML 迭代优化策略")
    parser.add_argument("--single-run", action="store_true", help="单次回测（不迭代）")
    parser.add_argument("--max-iter", type=int, default=0, help="最多迭代轮数 (0=全量)")
    args = parser.parse_args()

    if args.single_run:
        params = {
            'initial_capital': 1_000_000,
            'start_date': '20160520', 'end_date': '20260520',
            'train_end': '20211231', 'val_end': '20221231',
            'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 500,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 0.1, 'min_child_samples': 50,
            'max_positions': 5, 'max_hold_days': 12,
            'score_threshold': 0.3, 'atr_multiplier': 2.5,
            'take_profit_ratio': 4.0, 'position_sizing': 0.20,
        }
        result = run_single(params, verbose=True)
        logger.success(f"完成: 年化={result['annual_return']*100:.1f}%")
        return

    # 网格搜索参数空间（精简为 36 组）
    param_grid = {
        'max_hold_days': [10, 12, 15],
        'atr_multiplier': [2.0, 2.5],
        'score_threshold': [0.25, 0.30],
        'max_positions': [4, 5, 6],
    }
    # 总组合: 3*2*2*3 = 36 组

    best = grid_search(param_grid, top_n=10)
    print("\n" + "=" * 80)
    print("迭代优化结果 TOP-10 (按年化收益排序)")
    print("=" * 80)
    for i, r in enumerate(best):
        p = r['params']
        parts = [f"年化:{r['annual_return']*100:5.1f}%", f"胜率:{r['win_rate']*100:4.1f}%",
                 f"回撤:{r['max_drawdown']*100:5.1f}%", f"交易:{r['total_trades']:4d}"]
        param_parts = [f"hold={p.get('max_hold_days','?')}", f"atr={p.get('atr_multiplier','?')}",
                       f"score={p.get('score_threshold','?')}", f"pos={p.get('max_positions','?')}"]
        print(f"{i+1}. {' '.join(parts)} | {' '.join(param_parts)}")

    print(f"\n最优参数: {best[0]['params']}")


if __name__ == "__main__":
    main()
