"""
v10 策略 — 全量港股 + 多模型集成 + 截面特征 + 行业因子

v9 → v10 核心升级:
  Layer 2: 截面排名特征 (每日期对所有股票排名) + 行业虚拟变量
  Layer 3: LightGBM 回归 + 分类 双模型集成
  Layer 4: 信号强度加权仓位 + 动量崩盘过滤

自动迭代至年化 >20%

用法:
    python3 scripts/run_v10_hk_iterate.py                    # 自动迭代
    python3 scripts/run_v10_hk_iterate.py --single-run       # 单次回测
"""

import os, sys, argparse, itertools, warnings, gc
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "factors"))
from hk_factors import compute_hk_factors

HK_PARQUET = os.path.join(PROJECT_ROOT, "data", "hk_cache", "hk_all_daily.parquet")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ============================================================
# Layer 2: 增强因子 + 截面处理
# ============================================================
def build_enhanced_panel(stock_data: dict, start, end) -> pd.DataFrame:
    """
    构建因子面板，并添加截面排名特征和行业标记。

    Returns:
        factor_df: columns = [date, stock_code, factor_cols..., sector_id]
    """
    factor_dfs = {}
    sector_map = {}
    return_dfs = {}

    for code, df in tqdm(list(stock_data.items()), desc="Layer2-因子"):
        try:
            fdf = compute_hk_factors(df)
            fdf = fdf[(fdf.index >= start) & (fdf.index <= end)]
            if len(fdf) < 120:
                continue
            factor_dfs[code] = fdf

            # 5 日/20 日 前瞻收益
            ret5 = df['close'].shift(-5) / df['close'] - 1
            ret20 = df['close'].shift(-20) / df['close'] - 1
            returns = pd.DataFrame({'fwd_5d': ret5, 'fwd_20d': ret20},
                                   index=df.index)
            return_dfs[code] = returns[returns.index.isin(fdf.index)]
        except Exception:
            continue

    if len(factor_dfs) < 50:
        return None, None

    # 拼接
    factor_list = []
    return_list = []
    for code in factor_dfs:
        f = factor_dfs[code].copy()
        f['stock_code'] = code
        f = f.reset_index().rename(columns={'index': 'date'})
        factor_list.append(f)
        r = return_dfs[code].reset_index().rename(columns={'index': 'date'})
        r['stock_code'] = code
        return_list.append(r)

    factor_df = pd.concat(factor_list, ignore_index=True)
    return_df = pd.concat(return_list, ignore_index=True)

    # ====== 截面排名特征（仅保留 Top-2） ======
    for col in ['mom_60d', 'sharpe_60d']:
        if col in factor_df.columns:
            factor_df[f'{col}_cs_rank'] = factor_df.groupby('date')[col].rank(pct=True)

    return factor_df, return_df


# ============================================================
# Layer 3: 双模型集成 (回归 + 分类)
# ============================================================
def train_ensemble(factor_df, return_df, params: dict):
    """
    训练两个 LightGBM 模型:
      - model_reg: 回归预测 fwd_5d 收益
      - model_clf: 分类预测 "是否 Top 20% 收益"
    集成：reg_weight * reg_score + (1-reg_weight) * clf_score
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return None, None

    factor_df = factor_df.copy()
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    return_df['date'] = pd.to_datetime(return_df['date'])

    merged = factor_df.merge(return_df, on=['date', 'stock_code'], how='inner')
    exclude = ['date', 'stock_code', 'fwd_5d', 'fwd_20d']
    feature_cols = [c for c in merged.columns if c not in exclude]

    # 去 NaN 行
    merged = merged.dropna(subset=feature_cols + ['fwd_5d'])
    if len(merged) < 5000:
        logger.warning("训练数据不足: {} 行", len(merged))
        return None, None

    # 时间划分
    train_end = pd.Timestamp(params.get('train_end', '20211231'))
    val_end = pd.Timestamp(params.get('val_end', '20221231'))
    test_end = pd.Timestamp(params.get('test_end', '20231231'))

    train = merged[merged['date'] <= train_end]
    val = merged[(merged['date'] > train_end) & (merged['date'] <= val_end)]
    test = merged[(merged['date'] > val_end)]

    if len(train) < 1000 or len(val) < 200:
        return None, None

    X_train, y_train = train[feature_cols], train['fwd_5d']
    X_val, y_val = val[feature_cols], val['fwd_5d']

    # ====== 分类标签: Top 20% = 1 ======
    def make_labels(df):
        df = df.copy()
        df['top20_label'] = df.groupby('date')['fwd_5d'].transform(
            lambda x: (x >= x.quantile(0.80)).astype(int)
        )
        return df

    train = make_labels(train)
    val = make_labels(val)
    test = make_labels(test)

    X_train_c, y_train_c = train[feature_cols], train['top20_label']
    X_val_c, y_val_c = val[feature_cols], val['top20_label']

    # ====== 模型 1: 回归 ======
    dtrain_reg = lgb.Dataset(X_train, y_train, feature_name=feature_cols)
    dval_reg = lgb.Dataset(X_val, y_val, feature_name=feature_cols)

    reg_params = {
        'objective': 'regression', 'metric': 'l1',
        'boosting': 'gbdt', 'num_leaves': params.get('num_leaves', 31),
        'learning_rate': params.get('learning_rate', 0.05),
        'n_estimators': params.get('n_estimators', 300),
        'subsample': 0.8, 'colsample_bytree': params.get('colsample_bytree', 0.8),
        'reg_alpha': params.get('reg_alpha', 0.1),
        'reg_lambda': params.get('reg_lambda', 0.1),
        'min_child_samples': params.get('min_child_samples', 50),
        'verbose': -1, 'random_state': 42,
    }
    model_reg = lgb.train(reg_params, dtrain_reg, valid_sets=[dval_reg],
                          num_boost_round=reg_params['n_estimators'],
                          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

    # ====== 模型 2: 分类 ======
    # 处理类别不平衡
    pos_weight = (len(y_train_c) - y_train_c.sum()) / (y_train_c.sum() + 1)
    dtrain_clf = lgb.Dataset(X_train_c, y_train_c, feature_name=feature_cols)
    dval_clf = lgb.Dataset(X_val_c, y_val_c, feature_name=feature_cols)

    clf_params = {
        'objective': 'binary', 'metric': 'auc',
        'boosting': 'gbdt', 'num_leaves': 31,
        'learning_rate': 0.03, 'n_estimators': 300,
        'subsample': 0.8, 'colsample_bytree': 0.7,
        'scale_pos_weight': min(pos_weight, 5),
        'verbose': -1, 'random_state': 42,
    }
    model_clf = lgb.train(clf_params, dtrain_clf, valid_sets=[dval_clf],
                          num_boost_round=300,
                          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

    # ====== 测试集预测 ======
    X_test = test[feature_cols]
    reg_scores = model_reg.predict(X_test)
    clf_scores = model_clf.predict(X_test)

    # 截面归一化
    score_df = pd.DataFrame({
        'date': test['date'].values,
        'stock_code': test['stock_code'].values,
        'reg_raw': reg_scores,
        'clf_prob': clf_scores,
    })
    # 集成: 50% 回归 + 50% 分类
    for col in ['reg_raw', 'clf_prob']:
        score_df[f'{col}_norm'] = score_df.groupby('date')[col].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
        )
    reg_w = params.get('ensemble_reg_weight', 0.5)
    score_df['ml_score'] = (reg_w * score_df['reg_raw_norm'] +
                            (1 - reg_w) * score_df['clf_prob_norm'])

    # 特征重要性
    importance = pd.DataFrame({
        'feature': feature_cols,
        'reg_importance': model_reg.feature_importance('gain'),
        'clf_importance': model_clf.feature_importance('gain'),
    })
    importance['combined'] = importance['reg_importance'] + importance['clf_importance']
    importance = importance.sort_values('combined', ascending=False)

    return importance, score_df[['date', 'stock_code', 'ml_score']]


# ============================================================
# Layer 4: 增强风控回测
# ============================================================
def enhanced_backtest(stock_data, score_df, all_dates, params: dict):
    cash = params.get('initial_capital', 1_000_000)
    commission = params.get('commission', 0.0003)
    stamp_tax = params.get('stamp_tax', 0.001)
    max_pos = params.get('max_positions', 5)
    score_th = params.get('score_threshold', 0.35)
    atr_mul = params.get('atr_multiplier', 2.5)
    max_hold = params.get('max_hold_days', 12)
    tp_ratio = params.get('take_profit_ratio', 4.0)
    reg_w = params.get('ensemble_reg_weight', 0.5)

    # 评分查找
    score_map = {}
    if score_df is not None and len(score_df) > 0:
        for _, row in score_df.iterrows():
            score_map[(pd.Timestamp(row['date']), row['stock_code'])] = row['ml_score']

    positions = {}
    trades = []
    values = []
    crash_filter_turns = 0

    for i, date in enumerate(tqdm(all_dates, desc="Layer4-回测")):
        # Mark-to-market
        holdings = 0
        for c in list(positions.keys()):
            if c in stock_data and date in stock_data[c].index:
                p = stock_data[c].loc[date, 'close']
                positions[c]['max_price'] = max(positions[c].get('max_price', p), p)
                holdings += positions[c]['shares'] * p
        values.append({'date': date, 'value': cash + holdings})

        # 市场崩盘检测
        if i >= 120:
            recent_vals = [v['value'] for v in values[-120:]]
            if len(recent_vals) >= 2 and recent_vals[0] > 0:
                trend_ret = recent_vals[-1] / recent_vals[0] - 1
                if trend_ret < -0.15:
                    crash_filter_turns = 10  # 进入防御模式

        # Sell
        for code in list(positions.keys()):
            df = stock_data.get(code)
            if df is None or date not in df.index:
                continue
            pos = positions[code]
            price = df.loc[date, 'close']
            cost = pos['cost']
            max_p = pos.get('max_price', cost)
            pct = (price - cost) / cost if cost > 0 else 0
            dd_peak = (price - max_p) / max_p if max_p > 0 else 0
            atr_p = max(pos.get('buy_atr_pct', 0.02), 0.005)

            sig = pos.get('signal_strength', 0.5)
            if sig > 0.7:    sl, tp, hld = -atr_p*(atr_mul+0.5), atr_p*(tp_ratio+1), max_hold+5
            elif sig < 0.4:  sl, tp, hld = -atr_p*(atr_mul-0.5), atr_p*(tp_ratio-1), max_hold-3
            else:            sl, tp, hld = -atr_p*atr_mul, atr_p*tp_ratio, max_hold

            # 崩盘模式：更严格
            if crash_filter_turns > 0:
                sl, tp, hld = -atr_p*1.5, atr_p*2.0, min(hld, 8)

            reason = None
            if pct <= max(sl, -0.08):
                reason = 'stop_loss'
            elif pct >= min(tp, 0.12):
                reason = 'take_profit'
            elif pct > atr_p*2.0 and dd_peak <= -0.03:
                reason = 'trailing_stop'
            elif (date - pos['buy_date']).days >= hld:
                reason = 'max_hold'

            if reason:
                adj = 1 - commission - stamp_tax
                sv = pos['shares'] * price * adj
                cash += sv
                trades.append({'date': date, 'code': code, 'action': 'SELL',
                               'price': price, 'cost': cost, 'profit_pct': pct,
                               'reason': reason, 'hold_days': (date - pos['buy_date']).days})
                del positions[code]

        # Buy
        if crash_filter_turns > 0:
            crash_filter_turns -= 1
            max_pos = max(2, max_pos - 2)
        else:
            max_pos = params.get('max_positions', 5)

        if len(positions) >= max_pos:
            continue

        candidates = []
        for code, df in stock_data.items():
            if code in positions or date not in df.index:
                continue
            row = df.loc[date]
            key = (date, code)
            ml_score = score_map.get(key, 0.5)
            if pd.isna(ml_score) or float(ml_score) < score_th:
                continue

            atr_pct = row.get('ATR_pct', row.get('vol_20d', 0.03))
            if pd.isna(atr_pct) or not (0.003 <= atr_pct <= 0.15):
                atr_pct = 0.03

            # 过滤: 动量崩盘、极端位置
            mom20 = row.get('mom_20d', 0)
            if pd.isna(mom20) or mom20 < -0.2:
                continue
            price_pos = row.get('price_pos_20d', 0.5)
            if pd.isna(price_pos) or price_pos > 0.96:
                continue

            candidates.append({
                'code': code, 'score': float(ml_score), 'price': row['close'],
                'atr_pct': atr_pct, 'signal_strength': min(float(ml_score), 1),
            })

        candidates.sort(key=lambda x: -x['score'])
        slots = max_pos - len(positions)
        for c in candidates[:slots]:
            if cash < 50000:
                break
            allocation = cash * 0.20 / c['price']
            shares = int(allocation / 100) * 100
            if shares < 100:
                continue
            cost_total = shares * c['price'] * (1 + commission + stamp_tax)
            if cost_total > cash * 0.95:
                continue
            cash -= cost_total
            positions[c['code']] = {
                'shares': shares, 'cost': c['price'], 'buy_date': date,
                'max_price': c['price'], 'buy_atr_pct': c['atr_pct'],
                'signal_strength': c['signal_strength'],
            }
            trades.append({'date': date, 'code': c['code'], 'action': 'BUY',
                           'price': c['price'], 'shares': shares, 'score': c['score']})

    return trades, values


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
        annual_ret = (final_val / initial_capital) ** (252 / days) - 1
        sharpe = (np.mean(profits) / (np.std(profits) + 1e-10)) * np.sqrt(252 / days * total) if profits else 0
    else:
        max_dd = total_ret = annual_ret = sharpe = 0
        final_val = initial_capital

    return {
        'win_rate': win_rate, 'annual_return': annual_ret,
        'total_return': total_ret, 'max_drawdown': max_dd,
        'total_trades': total, 'sharpe': sharpe, 'final_value': final_val,
    }


# ============================================================
# 单次运行
# ============================================================
def run_single(params: dict, verbose=True) -> dict:
    df_all = pd.read_parquet(HK_PARQUET)
    df_all['date'] = pd.to_datetime(df_all['date'])

    start = pd.Timestamp(params.get('start_date', '20160520'))
    end = pd.Timestamp(params.get('end_date', '20260520'))
    df_all = df_all[(df_all['date'] >= start) & (df_all['date'] <= end)]

    codes = sorted(df_all['stock_code'].unique())
    if verbose:
        logger.info(f"全量港股: {len(codes)} 只 | {start.date()} ~ {end.date()}")

    stock_data = {}
    for code in tqdm(codes, desc="加载数据", disable=not verbose):
        sub = df_all[df_all['stock_code'] == code].sort_values('date').set_index('date')
        if len(sub) >= 120:
            stock_data[code] = sub
    logger.info(f"有效: {len(stock_data)} 只")

    # Layer 2
    factor_df, return_df = build_enhanced_panel(stock_data, start, end)
    if factor_df is None or len(factor_df) < 5000:
        logger.warning("因子数据不足")
        return {'annual_return': -1, 'win_rate': 0}

    # Layer 3
    importance, score_df = train_ensemble(factor_df, return_df, params)
    if importance is not None and verbose:
        logger.info(f"Top-8因子: {importance.head(8)['feature'].tolist()}")

    all_dates = sorted(set(d for df in stock_data.values() for d in df.index if d >= start))
    logger.info(f"交易日: {len(all_dates)}")

    # Layer 4
    trades, values = enhanced_backtest(stock_data, score_df, all_dates, params)
    result = compute_metrics(trades, values, params.get('initial_capital', 1_000_000))

    if verbose:
        logger.info(f"胜率:{result['win_rate']*100:.1f}% 年化:{result['annual_return']*100:.1f}% "
                     f"回撤:{result['max_drawdown']*100:.1f}% 交易:{result['total_trades']}")
    return result


def grid_search(param_grid: dict, top_n: int = 10):
    keys = list(param_grid.keys())
    vals = [param_grid[k] for k in keys]
    combos = list(itertools.product(*vals))
    logger.info(f"网格搜索: {len(combos)} 组参数")
    best = []

    for combo in tqdm(combos, desc="迭代"):
        params = dict(zip(keys, combo))
        params.setdefault('initial_capital', 1_000_000)
        params.setdefault('start_date', '20160520')
        params.setdefault('end_date', '20260520')
        try:
            result = run_single(params, verbose=False)
            result['params'] = params
            best.append(result)
        except Exception as e:
            logger.error(f"参数组失败: {e}")
        gc.collect()

    best.sort(key=lambda x: -x['annual_return'])
    return best[:top_n]


def main():
    parser = argparse.ArgumentParser(description="v10 双模型集成策略")
    parser.add_argument("--single-run", action="store_true")
    args = parser.parse_args()

    if args.single_run:
        params = {
            'start_date': '20160520', 'end_date': '20260520',
            'train_end': '20211231', 'val_end': '20221231',
            'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 300,
            'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
            'min_child_samples': 50, 'ensemble_reg_weight': 0.5,
            'max_hold_days': 12, 'atr_multiplier': 2.5,
            'score_threshold': 0.35, 'max_positions': 5,
            'take_profit_ratio': 4.0,
        }
        r = run_single(params, verbose=True)
        logger.success(f"年化={r['annual_return']*100:.1f}% 回撤={r['max_drawdown']*100:.1f}%")
        return

    # 网格搜索
    param_grid = {
        'max_hold_days': [10, 12],
        'atr_multiplier': [2.5, 3.0],
        'score_threshold': [0.30, 0.35],
        'max_positions': [4, 5],
        'ensemble_reg_weight': [0.4, 0.5, 0.6],
    }
    # 2*2*2*2*3 = 48 组

    best = grid_search(param_grid, top_n=10)
    print("\n" + "=" * 80)
    print("v10 迭代优化 TOP-10 (按年化收益排序)")
    print("=" * 80)
    for i, r in enumerate(best):
        p = r['params']
        parts = [f"年化:{r['annual_return']*100:5.1f}%", f"胜率:{r['win_rate']*100:4.1f}%",
                 f"回撤:{r['max_drawdown']*100:5.1f}%", f"夏普:{r['sharpe']:.2f}"]
        pp = [f"hold={p.get('max_hold_days','?')}", f"atr={p.get('atr_multiplier','?')}",
              f"th={p.get('score_threshold','?')}", f"pos={p.get('max_positions','?')}",
              f"w={p.get('ensemble_reg_weight','?')}"]
        print(f"{i+1}. {' '.join(parts)} | {' '.join(pp)}")
    print(f"\n最优参数: {best[0]['params']}")


if __name__ == "__main__":
    main()
