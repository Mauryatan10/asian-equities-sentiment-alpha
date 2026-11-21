import json
from pathlib import Path
import pandas as pd
import numpy as np

def generate_report():
    # 1. Load Data
    with open('data/processed/tearsheet_metrics.json') as f:
        tearsheet = json.load(f)

    live_snap = pd.read_parquet('data/processed/live_snapshot.parquet')
    events = pd.read_parquet('data/processed/events_aligned.parquet')
    preds = pd.read_parquet('data/processed/predictions_oos.parquet')
    features = pd.read_parquet('data/processed/features_1h.parquet')

    # Merge features to get bar-level sentiment signals and S/R levels
    preds = preds.merge(
        features[['ticker', 'bar_close_utc', 'retail_sent_24h', 'news_sent_24h', 'divergence_retail_vs_news_24h', 'support_1', 'resistance_1', 'dist_to_support_1_atr', 'dist_to_resistance_1_atr']],
        on=['ticker', 'bar_close_utc'],
        how='left'
    )

    all_tickers = [
        "0700.HK", "9988.HK", "1810.HK", "0981.HK", "1211.HK",
        "005930.KS", "000660.KS", "035720.KS", "005380.KS", "373220.KS",
    ]

    # 2. Extract Per-Ticker Records strictly for out-of-sample test evaluation
    ticker_records = []
    for t in all_tickers:
        ls = live_snap[live_snap['ticker'] == t].iloc[0]
        ev = events[events['ticker'] == t]
        ts = tearsheet['per_ticker'].get(t, {})
        
        social_count = len(ev[ev['source'].isin(['telegram', 'naver_finance', 'lihkg'])])
        news_count = len(ev[ev['source'] == 'exa'])
        
        strat_full = ts.get('strategy_full_ls', {})
        strat_tech = ts.get('strategy_tech_ls', {})
        forecast = ts.get('forecast', {})
        
        full_ret_pct = float(strat_full.get('total_net_return_pct', 0.0))
        tech_ret_pct = float(strat_tech.get('total_net_return_pct', 0.0))
        alpha_delta_bps = (full_ret_pct - tech_ret_pct) * 100.0

        # Extract bar-by-bar position history from OOS predictions
        sub_preds = preds[preds['ticker'] == t]
        short_bars = int((sub_preds['full_pos_ls'] == -1).sum())
        flat_bars = int((sub_preds['full_pos_ls'] == 0).sum())
        long_bars = int((sub_preds['full_pos_ls'] == 1).sum())
        total_bars = len(sub_preds)
        
        direction = "Short" if short_bars > 0 else "Flat"

        avg_social_sent = float(sub_preds['retail_sent_24h'].mean()) if len(sub_preds) > 0 else float(ls['retail_sent_24h'])
        avg_news_sent = float(sub_preds['news_sent_24h'].mean()) if len(sub_preds) > 0 else float(ls['news_sent_24h'])
        avg_sent_diff = float(avg_social_sent - avg_news_sent)
        avg_actual_bps = float((sub_preds['actual_log_return_1h'] * 10000.0).mean()) if len(sub_preds) > 0 else 0.0

        rec = {
            'ticker': t,
            'exchange': ls['exchange'],
            'company_name': ls['company_name'],
            'social_mentions': int(social_count),
            'news_mentions': int(news_count),
            'total_mentions': int(social_count + news_count),
            'composite_sentiment': float(ls['composite_sentiment_score']),
            'retail_sentiment_24h': float(ls['retail_sent_24h']),
            'news_sentiment_24h': float(ls['news_sent_24h']),
            'divergence_retail_vs_news': float(ls['divergence_retail_vs_news_24h']),
            'avg_social_sent': avg_social_sent,
            'avg_news_sent': avg_news_sent,
            'avg_sent_diff': avg_sent_diff,
            'avg_actual_bps': avg_actual_bps,
            'direction': direction,
            'short_bars': short_bars,
            'flat_bars': flat_bars,
            'long_bars': long_bars,
            'total_bars': total_bars,
            'oos_net_return_pct': full_ret_pct,
            'oos_net_return_bps': full_ret_pct * 100.0,
            'tech_net_return_pct': tech_ret_pct,
            'tech_net_return_bps': tech_ret_pct * 100.0,
            'alpha_delta_bps': alpha_delta_bps,
            'oos_sharpe': float(strat_full.get('annualized_sharpe', 0.0)),
            'oos_max_dd_pct': float(strat_full.get('max_drawdown_pct', 0.0)),
            'profit_factor': float(strat_full.get('profit_factor', 0.0)),
        }
        ticker_records.append(rec)

    # 3. Extract Time Series for Chart 3 (Equity Curves)
    times = sorted(preds['bar_close_utc'].unique())
    time_series = []
    
    cum_hk_full = 1.0
    cum_hk_tech = 1.0
    cum_kr_full = 1.0
    cum_kr_tech = 1.0
    cum_blend_full = 1.0
    cum_blend_tech = 1.0
    
    for t in times:
        sub = preds[preds['bar_close_utc'] == t]
        hk = sub[sub['exchange'] == 'HKEX']
        kr = sub[sub['exchange'] == 'KRX']
        
        hk_f = hk['full_net_ret_ls'].mean() if len(hk) > 0 else 0.0
        hk_t = hk['tech_net_ret_ls'].mean() if len(hk) > 0 else 0.0
        kr_f = kr['full_net_ret_ls'].mean() if len(kr) > 0 else 0.0
        kr_t = kr['tech_net_ret_ls'].mean() if len(kr) > 0 else 0.0
        
        # In a 50/50 cross-exchange portfolio allocation, when an exchange is closed its return is 0
        bl_f = 0.5 * hk_f + 0.5 * kr_f
        bl_t = 0.5 * hk_t + 0.5 * kr_t
            
        cum_hk_full *= (1.0 + hk_f)
        cum_hk_tech *= (1.0 + hk_t)
        cum_kr_full *= (1.0 + kr_f)
        cum_kr_tech *= (1.0 + kr_t)
        cum_blend_full *= (1.0 + bl_f)
        cum_blend_tech *= (1.0 + bl_t)
        
        dt_str = pd.to_datetime(t).strftime('%m-%d %H:%M')
        time_series.append({
            'time': dt_str,
            'hk_full_pct': round((cum_hk_full - 1.0) * 100.0, 3),
            'hk_tech_pct': round((cum_hk_tech - 1.0) * 100.0, 3),
            'kr_full_pct': round((cum_kr_full - 1.0) * 100.0, 3),
            'kr_tech_pct': round((cum_kr_tech - 1.0) * 100.0, 3),
            'blend_full_pct': round((cum_blend_full - 1.0) * 100.0, 3),
            'blend_tech_pct': round((cum_blend_tech - 1.0) * 100.0, 3)
        })

    # 4. Extract Granular Bar-by-Bar Trade Blotter (180 Bars)
    blotter_records = []
    preds_sorted = preds.sort_values(by=['bar_close_utc', 'ticker'])
    for _, row in preds_sorted.iterrows():
        t = row['ticker']
        ls = live_snap[live_snap['ticker'] == t].iloc[0]
        full_pos = int(row['full_pos_ls'])
        stance_str = "Short" if full_pos == -1 else ("Long" if full_pos == 1 else "Flat")

        pred_ret_bps = float(row['full_pred_ret']) * 10000.0
        act_ret_bps = float(row['actual_log_return_1h']) * 10000.0
        net_pnl_bps = float(row['full_net_ret_ls']) * 10000.0
        slip_bps = float(row['full_slippage_ls']) * 10000.0

        ts = pd.to_datetime(row['bar_close_utc'])
        formatted_time = ts.strftime('%m-%d %H:%M UTC')
        local_date = str(row['trading_date_local'])

        curr_price = float(row['close'])
        next_price = float(row['actual_next_close'])
        s1 = float(row['support_1']) if pd.notna(row['support_1']) else 0.0
        r1 = float(row['resistance_1']) if pd.notna(row['resistance_1']) else 0.0
        dist_s1 = float(row['dist_to_support_1_atr']) if pd.notna(row['dist_to_support_1_atr']) else 0.0
        dist_r1 = float(row['dist_to_resistance_1_atr']) if pd.notna(row['dist_to_resistance_1_atr']) else 0.0

        social_s = float(row['retail_sent_24h']) if pd.notna(row['retail_sent_24h']) else 0.0
        news_s = float(row['news_sent_24h']) if pd.notna(row['news_sent_24h']) else 0.0
        div_s = float(row['divergence_retail_vs_news_24h']) if pd.notna(row['divergence_retail_vs_news_24h']) else 0.0

        blotter_records.append({
            'bar_close_utc': formatted_time,
            'trading_date_local': local_date,
            'ticker': t,
            'company_name': ls['company_name'],
            'exchange': row['exchange'],
            'social_sent': round(social_s, 3),
            'news_sent': round(news_s, 3),
            'divergence': round(div_s, 3),
            'close_price': curr_price,
            'support_1': round(s1, 2),
            'resistance_1': round(r1, 2),
            'dist_to_s1': round(dist_s1, 2),
            'dist_to_r1': round(dist_r1, 2),
            'actual_next_close': next_price,
            'pred_ret_bps': round(pred_ret_bps, 1),
            'stance': stance_str,
            'pos_val': full_pos,
            'act_ret_bps': round(act_ret_bps, 1),
            'slip_bps': round(slip_bps, 1),
            'net_pnl_bps': round(net_pnl_bps, 1),
        })

    # Summary metrics
    overall_ls = tearsheet['overall']['strategy_full_ls']
    overall_tech = tearsheet['overall']['strategy_tech_ls']
    hk_ls = tearsheet['hkex_portfolio']['strategy_full_ls']
    kr_ls = tearsheet['krx_portfolio']['strategy_full_ls']
    
    kpis = {
        'blend_return': overall_ls['total_net_return_pct'],
        'blend_return_bps': overall_ls['total_net_return_pct'] * 100.0,
        'blend_sharpe': tearsheet['overall']['exchange_50_50_summary']['annualized_sharpe'],
        'blend_profit_factor': overall_ls['profit_factor'],
        'blend_max_dd': overall_ls['max_drawdown_pct'],
        'tech_return': overall_tech['total_net_return_pct'],
        'tech_return_bps': overall_tech['total_net_return_pct'] * 100.0,
        'tech_sharpe': overall_tech['annualized_sharpe'],
        'alpha_spread_bps': (overall_ls['total_net_return_pct'] - overall_tech['total_net_return_pct']) * 100.0,
        'alpha_spread_pct': (overall_ls['total_net_return_pct'] - overall_tech['total_net_return_pct']),
        'hk_return': hk_ls['total_net_return_pct'],
        'hk_sharpe': hk_ls['annualized_sharpe'],
        'hk_max_dd': hk_ls['max_drawdown_pct'],
        'hk_profit_factor': hk_ls['profit_factor'],
        'kr_return': kr_ls['total_net_return_pct'],
        'kr_sharpe': kr_ls['annualized_sharpe'],
        'kr_max_dd': kr_ls['max_drawdown_pct'],
        'kr_profit_factor': kr_ls['profit_factor'],
        'ic_full': tearsheet['overall']['forecast_full']['spearman_ic'],
        'ic_tech': tearsheet['overall']['forecast_tech']['spearman_ic'],
        'active_bars': overall_ls['active_trade_bars']
    }

    # Load template
    template_path = Path('scripts/cwt_visual_report_template.html')
    template = template_path.read_text(encoding='utf-8')

    # Compute Sentiment Tier Distributions for Chart 2 across ALL dates (895 bars)
    df_all = features.dropna(subset=['target_log_return_1h', 'retail_sent_24h', 'news_sent_24h']).copy()
    df_all['actual_bps'] = df_all['target_log_return_1h'] * 10000.0
    df_all['sent_diff'] = df_all['retail_sent_24h'] - df_all['news_sent_24h']

    def calc_tiers(col_name, bin_edges, bin_labels):
        df_all['temp_bin'] = pd.cut(df_all[col_name], bins=bin_edges, labels=bin_labels)
        grouped = df_all.groupby(['temp_bin', 'exchange'], observed=False).agg(
            avg_bps=('actual_bps', 'mean'),
            count=('actual_bps', 'count')
        ).reset_index()
        
        result = []
        for lbl in bin_labels:
            hk_rows = grouped[(grouped['temp_bin'] == lbl) & (grouped['exchange'] == 'HKEX')]
            kr_rows = grouped[(grouped['temp_bin'] == lbl) & (grouped['exchange'] == 'KRX')]
            
            hk_bps = round(float(hk_rows['avg_bps'].iloc[0]), 2) if len(hk_rows) > 0 and pd.notnull(hk_rows['avg_bps'].iloc[0]) else 0.0
            hk_cnt = int(hk_rows['count'].iloc[0]) if len(hk_rows) > 0 else 0
            kr_bps = round(float(kr_rows['avg_bps'].iloc[0]), 2) if len(kr_rows) > 0 and pd.notnull(kr_rows['avg_bps'].iloc[0]) else 0.0
            kr_cnt = int(kr_rows['count'].iloc[0]) if len(kr_rows) > 0 else 0

            result.append({
                'tier': lbl,
                'hkex_bps': hk_bps,
                'hkex_count': hk_cnt,
                'krx_bps': kr_bps,
                'krx_count': kr_cnt
            })
        return result

    sentiment_tiers = {
        'social': calc_tiers('retail_sent_24h', [-1.0, -0.20, -0.10, 0.00, 1.0], ['<-0.20 (Bearish)', '[-0.20, -0.10)', '[-0.10, 0.00)', '>=0.00 (Bullish)']),
        'news': calc_tiers('news_sent_24h', [-1.0, -0.20, -0.10, 0.00, 1.0], ['<-0.20 (Bearish)', '[-0.20, -0.10)', '[-0.10, 0.00)', '>=0.00 (Bullish)']),
        'diff': calc_tiers('sent_diff', [-1.5, -0.12, 0.00, 0.12, 1.5], ['<-0.12 (News Dominant)', '[-0.12, 0.00)', '[0.00, +0.12)', '>=+0.12 (Social Dominant)'])
    }

    # Replace placeholders
    replacements = {
        '__BLEND_RET__': f"{kpis['blend_return']:.2f}",
        '__BLEND_RET_BPS__': f"{kpis['blend_return_bps']:.1f}",
        '__BLEND_SHARPE__': f"{kpis['blend_sharpe']:.2f}",
        '__BLEND_MAX_DD__': f"{kpis['blend_max_dd']:.2f}",
        '__BLEND_PF__': f"{kpis['blend_profit_factor']:.2f}",
        '__ACTIVE_BARS__': str(kpis['active_bars']),
        '__HK_RET__': f"{kpis['hk_return']:.2f}",
        '__HK_RET_BPS__': f"{kpis['hk_return']*100:.1f}",
        '__HK_SHARPE__': f"{kpis['hk_sharpe']:.2f}",
        '__HK_PF__': f"{kpis['hk_profit_factor']:.2f}",
        '__HK_MAX_DD__': f"{kpis['hk_max_dd']:.2f}",
        '__KR_RET__': f"{kpis['kr_return']:.2f}",
        '__KR_RET_BPS__': f"{kpis['kr_return']*100:.1f}",
        '__KR_SHARPE__': f"{kpis['kr_sharpe']:.2f}",
        '__KR_PF__': f"{kpis['kr_profit_factor']:.2f}",
        '__KR_MAX_DD__': f"{kpis['kr_max_dd']:.2f}",
        '__ALPHA_SPREAD_BPS__': f"{kpis['alpha_spread_bps']:.1f}",
        '__ALPHA_SPREAD_PCT__': f"{kpis['alpha_spread_pct']:.2f}",
        '__SHARPE_DELTA__': f"{kpis['blend_sharpe'] - kpis['tech_sharpe']:.2f}",
        '__TECH_SHARPE__': f"{kpis['tech_sharpe']:.2f}",
        '__IC_FULL__': f"{kpis['ic_full']:.4f}",
        '__IC_TECH__': f"{kpis['ic_tech']:.4f}",
        '__TICKERS_JSON__': json.dumps(ticker_records),
        '__TIMESERIES_JSON__': json.dumps(time_series),
        '__TRADES_JSON__': json.dumps(blotter_records),
        '__KPIS_JSON__': json.dumps(kpis),
        '__TIERS_JSON__': json.dumps(sentiment_tiers)
    }

    html_out = template
    for key, val in replacements.items():
        html_out = html_out.replace(key, val)

    # 4. Save to primary report directory
    out_path_reports = Path('reports/cwt_visual_report.html')
    out_path_reports.parent.mkdir(parents=True, exist_ok=True)
    out_path_reports.write_text(html_out, encoding='utf-8')
    
    print(f"Clean report generated successfully!")
    print(f"Saved to: {out_path_reports} ({len(html_out)} bytes)")

if __name__ == '__main__':
    generate_report()
