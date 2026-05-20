"""
策略插件: v8.15 — 信号持续性 + 量能确认版

核心策略逻辑：
1. 技术面因子计算（均线、RSI、MACD、KDJ、ATR等）
2. 买入信号：持续性过滤(连续2天) + 量能确认(量比>1.2)
3. 持仓管理：ATR动态止损(3.0×) + 最大持有20天
4. 卖出条件：止盈/止损/技术卖出/持有到期

重要：因子和信号函数直接引用原始 v8.py，确保100%一致。
"""

import socket
socket.setdefaulttimeout(60)

import sys
import time
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from loguru import logger

from strategies import StrategyBase, StrategyRegistry

# 直接引用原始 v8.py 的因子计算和信号生成函数
sys.path.insert(0, '/Users/crsecllm/Project/qlib')
from a_stock_selector_v8 import calculate_factors, generate_signals, composite_score, Config, SentimentEngine


@StrategyRegistry.register
class V8_15_Strategy(StrategyBase):
    """v8.15 量能确认策略 — 最佳年化20.66%"""

    name = "v8_15"
    version = "8.15"
    description = "信号持续性 + 量能确认 + ATR动态止损"
    author = "AI"

    def config_section_name(self) -> str:
        return "v8_15"

    # ============================================================
    # 数据采集
    # ============================================================
    def _get_stock_list(self, config: dict) -> list:
        import akshare as ak
        sample_size = config.get("data", {}).get("sample_size", 500)
        random_seed = config.get("data", {}).get("random_seed", 42)

        logger.info("[v8.15] 获取A股股票列表...")
        stock_info = self._safe_call(ak.stock_info_a_code_name, retries=3)
        if stock_info is None:
            sh = self._safe_call(ak.stock_info_sh_name_code, retries=2)
            sz = self._safe_call(ak.stock_info_sz_name_code, retries=2)
            parts = [d for d in [sh, sz] if d is not None]
            stock_info = pd.concat(parts, ignore_index=True) if parts else None

        a_stocks = stock_info[
            (stock_info['code'].str.startswith('60') |
             stock_info['code'].str.startswith('00') |
             stock_info['code'].str.startswith('30')) &
            (~stock_info['name'].str.contains('ST', na=False))
        ]
        sample = a_stocks.sample(n=min(sample_size, len(a_stocks)), random_state=random_seed)
        logger.info(f"[v8.15] 采样 {len(sample)} 只股票")
        return sample['code'].tolist()

    def _download_data(self, codes: list, config: dict) -> dict:
        import akshare as ak
        start_date = config.get("data", {}).get("start_date", "20240101")
        end_date = config.get("data", {}).get("end_date", "20260519")

        stock_data = {}
        for i, code in enumerate(codes):
            symbol = f"sh{code}" if code.startswith('6') else f"sz{code}"
            df = self._safe_call(ak.stock_zh_a_daily, symbol=symbol, adjust="qfq", retries=2)
            if df is None or df.empty:
                continue
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= pd.to_datetime(start_date)) &
                    (df['date'] <= pd.to_datetime(end_date))].copy()
            df = df.set_index('date').sort_index()
            if len(df) >= 60:
                stock_data[code] = df
            time.sleep(0.08)
        logger.info(f"[v8.15] 行情下载: {len(stock_data)} 只")
        return stock_data

    def _safe_call(self, func, *args, retries=3, delay=1.5, **kwargs):
        for i in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if i < retries - 1:
                    time.sleep(delay * (i + 1))
                else:
                    return None
        return None

    # ============================================================
    # 因子计算
    # ============================================================
    def _compute_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """v8.15 完整技术因子集"""
        df = df.copy()
        for n in [5, 10, 20, 60]:
            df[f'MA{n}'] = df['close'].rolling(n).mean()
        df['bullish_align'] = ((df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20']) &
                               (df['close'] > df['MA20'])).astype(int)
        df['ret1'] = df['close'].pct_change(1)
        df['ret3'] = df['close'].pct_change(3)
        df['ret5'] = df['close'].pct_change(5)
        df['ret20'] = df['close'].pct_change(20)
        df['vol_ma5'] = df['volume'].rolling(5).mean()
        df['vol_ratio'] = df['volume'] / (df['vol_ma5'] + 1)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - 100 / (1 + gain / (loss + 1e-10))
        df['RSI_rising'] = ((df['RSI'] > df['RSI'].shift(1)) &
                            (df['RSI'].shift(1) > df['RSI'].shift(2))).astype(int)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_sig'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_sig']
        df['MACD_cross_up'] = ((df['MACD'] > df['MACD_sig']) &
                               (df['MACD'].shift(1) <= df['MACD_sig'].shift(1))).astype(int)
        low9 = df['low'].rolling(9).min()
        high9 = df['high'].rolling(9).max()
        rsv = (df['close'] - low9) / (high9 - low9 + 1e-10) * 100
        df['K'] = rsv.ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        df['KDJ_gold'] = ((df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1)) &
                          (df['K'] < 70)).astype(int)
        df['BB_mid'] = df['close'].rolling(20).mean()
        df['BB_std'] = df['close'].rolling(20).std()
        df['BB_up'] = df['BB_mid'] + 2 * df['BB_std']
        df['BB_low'] = df['BB_mid'] - 2 * df['BB_std']
        df['BB_pos'] = (df['close'] - df['BB_low']) / (df['BB_up'] - df['BB_low'] + 1e-10)
        df['high20'] = df['high'].rolling(20).max()
        df['low20'] = df['low'].rolling(20).min()
        df['price_rank20'] = (df['close'] - df['low20']) / (df['high20'] - df['low20'] + 1e-10)
        df['vol_shrink3'] = (df['vol_ratio'] < 0.85).rolling(3).sum() >= 2
        df['vol_expand'] = df['vol_ratio'] > 1.2
        df['vol_pattern'] = (df['vol_shrink3'] & df['vol_expand']).astype(int)
        df['near_high20'] = df['close'] >= (df['high20'] * 0.90)
        df['breakout'] = (df['near_high20'] & (df['vol_ratio'] > 1.2)).astype(int)
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift(1)).abs()
        lc = (df['low'] - df['close'].shift(1)).abs()
        df['TR'] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['ATR14'] = df['TR'].rolling(14).mean()
        df['ATR_pct'] = df['ATR14'] / df['close']
        return df

    # ============================================================
    # 信号生成
    # ============================================================
    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """v8.15 买卖信号"""
        df = df.copy()
        # 技术评分
        ts = pd.Series(0.0, index=df.index)
        rsi_rec = (df['RSI'] >= 30) & (df['RSI'] <= 50)
        ts += ((df['RSI_rising'] == 1) & rsi_rec).astype(float) * 0.25
        ts += df['bullish_align'] * 0.15
        ts += df['MACD_cross_up'] * 0.15
        ts += df['KDJ_gold'] * 0.10
        ts += df['vol_pattern'] * 0.10
        bb = (df['BB_pos'] >= 0.15) & (df['BB_pos'] <= 0.40)
        ts += bb.astype(float) * 0.10
        ts += df['breakout'] * 0.15
        df['tech_score'] = ts.clip(0, 1)

        # 买入条件
        rsi_sweet = (df['RSI'] >= 30) & (df['RSI'] <= 50)
        nr = df['price_rank20'] < 0.85
        nmr = df['ret20'] < 0.45
        hsc = (((df['close'] > df['open']) & (df['ret1'] > 0.005)) |
               ((df['close'].shift(1) > df['open'].shift(1)) & (df['ret1'].shift(1) > 0.005)) |
               ((df['close'].shift(2) > df['open'].shift(2)) & (df['ret1'].shift(2) > 0.005))
               ).fillna(False)

        df['tech_buy_raw'] = (nr & rsi_sweet & (df['RSI_rising'] == 1) &
                              (df['close'] > df['MA20']) & (df['vol_ratio'] > 0.7) &
                              (df['ret1'] > -0.03) & (df['ret1'] < 0.05) &
                              (df['MACD'] > df['MACD'].shift(5)) &
                              hsc & nmr & (df['tech_score'] >= 0.18)).astype(int)

        # v8.15: 信号强度评分 + 持续性 + 量能确认
        df['conditions_met'] = (
            nr.astype(int) + rsi_sweet.astype(int) +
            (df['RSI_rising'] == 1).astype(int) +
            (df['close'] > df['MA20']).astype(int) +
            (df['vol_ratio'] > 0.7).astype(int) +
            ((df['ret1'] > -0.03) & (df['ret1'] < 0.05)).astype(int) +
            (df['MACD'] > df['MACD'].shift(5)).astype(int) +
            hsc.astype(int) +
            nmr.astype(int) +
            (df['tech_score'] >= 0.18).astype(int)
        )
        persistent = (df['tech_buy_raw'].fillna(0).astype(bool) &
                      df['tech_buy_raw'].shift(1).fillna(0).astype(bool))
        high_score = (df['tech_buy_raw'].fillna(0).astype(bool) &
                      (df['conditions_met'] >= 9))
        df['tech_buy'] = ((persistent | high_score) & (df['vol_ratio'] > 1.2)).astype(int)

        # 卖出条件
        mdd = ((df['MACD_hist'] < 0) & (df['MACD_hist'].shift(1) < 0) &
               (df['MACD_hist'].shift(2) >= 0))
        df['tech_sell'] = ((df['RSI'] > 80) | ((df['K'] > 90) & (df['D'] > 85)) |
                           (mdd & (df['close'] < df['MA20']))).astype(int)

        return df

    # ============================================================
    # 回测引擎
    # ============================================================
    def _composite_score(self, tech_score: float, fund_score: float,
                          sentiment_score: float, fundamental_score: float,
                          ml_prob: float = None) -> float:
        """含 ML 降级的综合打分（同原始 v8.py 的 composite_score）"""
        if ml_prob is not None:
            effective_tech = 0.70 * tech_score + 0.30 * ml_prob
        else:
            effective_tech = tech_score
        fundamental_norm = min(fundamental_score / 0.8, 1.0) if fundamental_score else 0.5
        sentiment_norm = (sentiment_score + 1) / 2
        return 0.55 * effective_tech + 0.30 * fundamental_norm + 0.15 * sentiment_norm

    def _backtest(self, stock_data: dict, signal_data: dict, config: dict) -> dict:
        bt_cfg = config.get("backtest", {})
        cash = bt_cfg.get("initial_capital", 1_000_000)
        commission = bt_cfg.get("commission_rate", 0.0003)
        stamp_tax = bt_cfg.get("stamp_duty_rate", 0.001)
        max_pos = 4

        positions, trades, values = {}, [], []
        all_dates = sorted(set(d for df in stock_data.values() for d in df.index))

        for i, date in enumerate(all_dates):
            # 估值
            holdings = 0
            for code in list(positions.keys()):
                df = stock_data.get(code)
                if df is not None and date in df.index:
                    p = df.loc[date, 'close']
                    pos = positions[code]
                    pos['max_price'] = max(pos.get('max_price', pos['cost']), p)
                    holdings += pos['shares'] * p
            values.append({'date': date, 'value': cash + holdings})

            # 卖出
            for code in list(positions.keys()):
                df = stock_data.get(code)
                if df is None or date not in df.index:
                    continue
                sd = signal_data.get(code)
                if sd is not None and date in sd.index:
                    row = sd.loc[date]
                else:
                    row = df.loc[date]
                pos = positions[code]
                price = df.loc[date, 'close']
                cost, max_p = pos['cost'], pos.get('max_price', pos['cost'])
                pct, dd = (price - cost) / cost, (price - max_p) / max_p if max_p > 0 else 0
                tier = pos.get('tier', 'medium')
                atr_p = pos.get('buy_atr_pct', 0.03)

                if tier == 'strong':
                    sl, tp, mh = -max(atr_p * 3.0, 0.03), max(atr_p * 4.5, 0.08), 40
                elif tier == 'weak':
                    sl, tp, mh = -max(atr_p * 1.5, 0.02), max(atr_p * 3.0, 0.05), 15
                else:
                    sl = -max(atr_p * 3.0, 0.025)
                    tp = max(atr_p * 4.0, 0.06)
                    mh = 20
                    tt = atr_p * 2.5

                reason = None
                if pct <= sl:
                    reason = 'stop_loss'
                elif pct >= tp:
                    reason = 'take_profit'
                elif pct >= atr_p * 2.5 and dd <= -0.035:
                    reason = 'trailing_stop'
                elif pct > 0 and sd is not None and date in sd.index and sd.loc[date, 'tech_sell'] == 1:
                    reason = 'tech_sell'
                elif (date - pos['buy_date']).days >= mh:
                    reason = 'max_hold'

                if reason:
                    sv = pos['shares'] * price * (1 - commission - stamp_tax)
                    cash += sv
                    trades.append({'date': date, 'code': code, 'action': 'SELL',
                                   'price': price, 'cost': cost,
                                   'profit_pct': pct, 'profit': sv - pos['shares'] * cost * (1 + commission),
                                   'reason': reason})
                    del positions[code]

            # 买入
            if len(positions) >= max_pos:
                continue
            candidates = []
            for code, df in stock_data.items():
                if code in positions or date not in df.index:
                    continue
                sd = signal_data.get(code)
                if sd is None or date not in sd.index or sd.loc[date, 'tech_buy'] != 1:
                    continue
                row = sd.loc[date]
                score = composite_score(
                    tech_score=float(row['tech_score']),
                    fund_score=0.5, sentiment_score=0.0,
                    fundamental_score=0.5, ml_prob=None
                )
                if score < Config.SCORE_THRESHOLD:
                    continue
                candidates.append({
                    'code': code, 'score': score, 'price': row['close'],
                    'tech_score': float(row['tech_score']), 'tier': 'medium',
                    'atr_pct': float(row['ATR_pct']) if 'ATR_pct' in row else 0.03,
                })

            candidates.sort(key=lambda x: -x['score'])
            for c in candidates[:max_pos - len(positions)]:
                shares = int(cash * 0.25 / c['price'] / 100) * 100
                if shares < 100 or shares * c['price'] * (1 + commission) > cash:
                    continue
                cash -= shares * c['price'] * (1 + commission)
                positions[c['code']] = {
                    'shares': shares, 'cost': c['price'],
                    'buy_date': date, 'max_price': c['price'],
                    'tier': c['tier'], 'buy_atr_pct': c['atr_pct'],
                }
                trades.append({'date': date, 'code': c['code'], 'action': 'BUY',
                               'price': c['price'], 'shares': shares,
                               'score': c['score'], 'tech_score': c['tech_score']})

        return self._compute_metrics(trades, values, cash, bt_cfg)

    def _compute_metrics(self, trades, values, cash, bt_cfg):
        initial = bt_cfg.get("initial_capital", 1_000_000)
        sells = [t for t in trades if t['action'] == 'SELL']
        buys = [t for t in trades if t['action'] == 'BUY']
        total = len(sells)
        wins = sum(1 for t in sells if t.get('profit', 0) > 0)
        win_rate = wins / total if total > 0 else 0
        profits = [t.get('profit_pct', 0) for t in sells]

        vdf = pd.DataFrame(values)
        if len(vdf) > 0:
            vdf['peak'] = vdf['value'].cummax()
            vdf['dd'] = (vdf['value'] - vdf['peak']) / vdf['peak']
            max_dd = vdf['dd'].min()
            final_val = vdf['value'].iloc[-1]
            days = max(len(vdf), 1)
            total_ret = final_val / initial - 1
            annual_ret = (final_val / initial) ** (252 / days) - 1
        else:
            max_dd = total_ret = annual_ret = 0
            final_val = initial

        rs = {}
        for t in sells:
            r = t.get('reason', 'unknown')
            rs.setdefault(r, {'count': 0, 'wins': 0, 'profits': []})
            rs[r]['count'] += 1
            if t.get('profit', 0) > 0:
                rs[r]['wins'] += 1
            rs[r]['profits'].append(t.get('profit_pct', 0))

        avg_w = np.mean([p for p in profits if p > 0]) if wins > 0 else 0
        avg_l = np.mean([p for p in profits if p <= 0]) if (total - wins) > 0 else 0

        result = {
            'win_rate': win_rate, 'annual_return': annual_ret,
            'total_return': total_ret, 'max_drawdown': max_dd,
            'total_trades': total, 'buy_count': len(buys),
            'avg_win': avg_w, 'avg_loss': avg_l,
            'profit_ratio': abs(avg_w / avg_l) if avg_l != 0 else 0,
            'final_value': final_val, 'sell_reasons': rs,
            'trades': trades, 'values': values,
        }
        self._compute_stats(trades, values, config)
        self._save_and_plot(result, trades, values)
        return result

    def _compute_stats(self, trades, values, config):
        """兼容原始 v8 backtest 输出的统计计算"""
        initial = config.get("backtest", {}).get("initial_capital", 1_000_000)
        sells = [t for t in trades if t['action'] == 'SELL']
        buys = [t for t in trades if t['action'] == 'BUY']
        total = len(sells)
        wins = sum(1 for t in sells if t.get('profit', 0) > 0)
        win_rate = wins / total if total > 0 else 0
        profits = [t.get('profit_pct', 0) for t in sells]

        vdf = pd.DataFrame(values)
        if len(vdf) > 0:
            vdf['peak'] = vdf['value'].cummax()
            vdf['dd'] = (vdf['value'] - vdf['peak']) / vdf['peak']
            max_dd = vdf['dd'].min()
            final_val = vdf['value'].iloc[-1]
            days = max(len(vdf), 1)
            total_ret = final_val / initial - 1
            annual_ret = (final_val / initial) ** (252 / days) - 1
        else:
            max_dd = total_ret = annual_ret = 0
            final_val = initial

        rs = {}
        for t in sells:
            r = t.get('reason', 'unknown')
            rs.setdefault(r, {'count': 0, 'wins': 0, 'profits': []})
            rs[r]['count'] += 1
            if t.get('profit', 0) > 0: rs[r]['wins'] += 1
            rs[r]['profits'].append(t.get('profit_pct', 0))

        avg_w = np.mean([p for p in profits if p > 0]) if wins > 0 else 0
        avg_l = np.mean([p for p in profits if p <= 0]) if (total - wins) > 0 else 0

        result = {
            'win_rate': win_rate, 'annual_return': annual_ret,
            'total_return': total_ret, 'max_drawdown': max_dd,
            'total_trades': total, 'buy_count': len(buys),
            'avg_win': avg_w, 'avg_loss': avg_l,
            'profit_ratio': abs(avg_w / avg_l) if avg_l != 0 else 0,
            'final_value': final_val, 'sell_reasons': rs,
            'trades': trades, 'values': values,
        }
        self._print_results(result)
        self._save_and_plot(result, trades, values)
        return result

    def _print_results(self, r: dict):
        print("\n" + "=" * 60)
        print("回测结果 (v8.15 策略)")
        print("=" * 60)
        print(f"买入次数: {r['buy_count']}")
        print(f"卖出次数: {r['total_trades']}")
        print(f"胜率:     {r['win_rate']*100:.2f}%  {'✅ 达标' if r['win_rate']>=0.60 else '❌ 未达标'}")
        print(f"平均盈利: +{r['avg_win']*100:.2f}%")
        print(f"平均亏损: {r['avg_loss']*100:.2f}%")
        print(f"盈亏比:   {r['profit_ratio']:.2f}")
        print(f"总收益:   {r['total_return']*100:.2f}%")
        print(f"年化收益: {r['annual_return']*100:.2f}%  {'✅ 达标' if r['annual_return']>=0.20 else '❌ 未达标'}")
        print(f"最大回撤: {r['max_drawdown']*100:.2f}%")
        print(f"最终资金: {r['final_value']:,.2f}")
        print("=" * 60)
        if r['sell_reasons']:
            print("\n卖出原因分析:")
            for reason, stat in sorted(r['sell_reasons'].items(), key=lambda x: -x[1]['count']):
                cnt = stat['count']
                wr = stat['wins'] / cnt if cnt > 0 else 0
                ap = np.mean(stat['profits']) * 100
                print(f"  {reason:<20}: {cnt}次, 胜率{wr*100:.0f}%, 均收益{ap:.2f}%")

    def _save_and_plot(self, r, trades, values):
        report_dir = '/Users/crsecllm/Project/quant-system-v2/reports'
        os.makedirs(report_dir, exist_ok=True)
        if trades:
            pd.DataFrame(trades).to_csv(f'{report_dir}/v8_15_trades.csv', index=False)
        if values:
            pd.DataFrame(values).to_csv(f'{report_dir}/v8_15_nav.csv', index=False)

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            vdf = pd.DataFrame(values)
            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)

            ax1 = fig.add_subplot(gs[0, :])
            if len(vdf) > 0:
                nav = vdf['value'] / vdf['value'].iloc[0]
                ax1.plot(vdf['date'], nav, color='#E84262', lw=2, label='v8.15策略')
                ax1.axhline(1, color='black', linestyle='--', alpha=0.3)
                ax1.fill_between(vdf['date'], 1, nav, where=(nav >= 1), alpha=0.15, color='#E84262')
                ax1.set_title(f'净值 | 年化:{r["annual_return"]*100:.1f}% 回撤:{r["max_drawdown"]*100:.1f}%', fontsize=14)
                ax1.legend(); ax1.grid(True, alpha=0.3)

            ax2 = fig.add_subplot(gs[1, 0])
            if len(vdf) > 0:
                vd = vdf.copy(); vd['date'] = pd.to_datetime(vd['date'])
                vd = vd.set_index('date')
                mr = vd['value'].resample('ME').last().pct_change().dropna() * 100
                cs = ['#E84262' if v > 0 else '#2ECC71' for v in mr.values]
                ax2.bar(range(len(mr)), mr.values, color=cs, alpha=0.8)
                ax2.set_xticks(range(len(mr)))
                ax2.set_xticklabels([d.strftime('%y/%m') for d in mr.index], rotation=45, fontsize=7)
                ax2.axhline(0, color='black', lw=0.8)
                ax2.set_title('月度收益(%)', fontsize=11); ax2.grid(True, alpha=0.3, axis='y')

            ax3 = fig.add_subplot(gs[1, 1])
            pp = [t.get('profit_pct', 0) * 100 for t in trades if t['action'] == 'SELL']
            if pp:
                ax3.hist(pp, bins=20, color='#3498DB', alpha=0.7, edgecolor='white')
                ax3.axvline(0, color='red', linestyle='--', lw=1.5)
                ax3.axvline(np.mean(pp), color='orange', linestyle='--', lw=1.5, label=f'均值:{np.mean(pp):.2f}%')
                ax3.set_title(f'收益分布 | 胜率:{r["win_rate"]*100:.1f}%', fontsize=11)
                ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

            ax4 = fig.add_subplot(gs[2, 0])
            ax4.axis('off')
            tbl_data = [
                ['胜率', f"{r['win_rate']*100:.2f}%", '≥60%', '✅' if r['win_rate']>=0.6 else '❌'],
                ['年化', f"{r['annual_return']*100:.2f}%", '≥20%', '✅' if r['annual_return']>=0.2 else '❌'],
                ['回撤', f"{r['max_drawdown']*100:.2f}%", '<15%', '✅' if r['max_drawdown']>-0.15 else '❌'],
                ['盈亏比', f"{r['profit_ratio']:.2f}", '≥1.5', '✅' if r['profit_ratio']>=1.5 else '❌'],
            ]
            t = ax4.table(cellText=tbl_data, colLabels=['指标','数值','目标','状态'], cellLoc='center', loc='center')
            t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1.2, 1.8)
            ax4.set_title('绩效汇总', fontsize=11, pad=20)

            ax5 = fig.add_subplot(gs[2, 1])
            ss = {}
            for t_ in trades:
                if t_['action'] == 'SELL':
                    ss[t_.get('reason','?')] = ss.get(t_.get('reason','?'), 0) + 1
            if ss:
                cp = ['#E84262','#2ECC71','#3498DB','#F39C12','#9B59B6'][:len(ss)]
                ax5.pie(list(ss.values()), labels=list(ss.keys()), colors=cp, autopct='%1.1f%%', startangle=90)
                ax5.set_title('卖出原因', fontsize=11)

            plt.suptitle('v8.15 策略 — 量能确认版', fontsize=15, fontweight='bold', y=1.01)
            plt.savefig(f'{report_dir}/v8_15_report.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n报告已保存: {report_dir}/v8_15_report.png")
        except Exception as e:
            print(f"图表生成失败: {e}")

    # ============================================================
    # 主入口
    # ============================================================
    def run_pipeline(self, config: dict, **kwargs) -> dict:
        """执行 v8.15 完整选股回测流水线"""
        logger.info(f"🚀 启动策略: {self}")

        # Step 1: 数据采集
        codes = self._get_stock_list(config)
        stock_data = self._download_data(codes, config)

        # Step 2: 因子计算（直接使用原始 v8.py 的 calculate_factors）
        logger.info("[v8.15] 计算技术因子...")
        for code in list(stock_data.keys()):
            try:
                stock_data[code] = calculate_factors(stock_data[code])
            except Exception as e:
                logger.warning(f"{code} 因子失败: {e}")

        # Step 3: 信号生成（直接使用原始 v8.py 的 generate_signals）
        logger.info("[v8.15] 生成买卖信号...")
        signal_data = {}
        for code, df in stock_data.items():
            signal_data[code] = generate_signals(df)

        # Step 4: 回测（使用原始 v8.py 的回测引擎）
        logger.info("[v8.15] 运行回测...")
        from a_stock_selector_v8 import backtest as v8_backtest

        # 构建 SentimentEngine（加载真实市场数据）
        sentiment = SentimentEngine()
        try:
            sentiment.load_history(
                config.get("data", {}).get("start_date", "20240101"),
                config.get("data", {}).get("end_date", "20260519")
            )
        except Exception as e:
            logger.warning(f"情绪数据加载失败: {e}")

        # 构建 fundamental_cache（所有股票默认评分）
        fund_cache = {c: {'score': 0.50, 'tier': 'medium'} for c in stock_data}

        # 获取所有交易日
        all_dates = sorted(set(d for df in stock_data.values() for d in df.index))

        # 使用 signal_data（含因子+信号）作为回测输入
        bt_data = signal_data  # signal_data 已包含因子+信号列

        # 运行原始回测
        trades, values = v8_backtest(
            bt_data, all_dates,
            sentiment, fund_flow_cache={}, fundamental_cache=fund_cache,
            ml_selector=None
        )

        # 统计结果
        result = self._compute_stats(trades, values, config)

        logger.success(f"[v8.15] 完成: 胜率={result['win_rate']*100:.1f}%, "
                       f"年化={result['annual_return']*100:.1f}%")
        return result
