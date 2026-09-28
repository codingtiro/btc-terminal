"""Compare historical Coinbase proxy against actual Chainlink TWAP and official outcomes.
Only observed, timestamp-aligned records are scored; no invented fills or oracle ticks.
"""
import numpy as np
import pandas as pd
from model import make_examples, estimate

def normalize_markets(frame):
    required={'start_time','end_time','winner','strike'}
    missing=required-set(frame.columns)
    if missing: raise ValueError(f'Market CSV missing: {sorted(missing)}. Required columns: start_time,end_time,winner,strike')
    m=frame.copy()
    m['start_time']=pd.to_datetime(m.start_time,utc=True,errors='coerce')
    m['end_time']=pd.to_datetime(m.end_time,utc=True,errors='coerce')
    m['strike']=pd.to_numeric(m.strike,errors='coerce')
    m['up']=m.winner.astype(str).str.strip().str.lower().map({'up':1,'down':0})
    m=m.dropna(subset=['start_time','end_time','strike','up']).sort_values('start_time')
    m=m[(m.end_time-m.start_time)==pd.Timedelta(minutes=15)]
    if 'twapLookbackSeconds' in m: m=m[pd.to_numeric(m.twapLookbackSeconds,errors='coerce')==60]
    if 'resolution_source' in m: m=m[m.resolution_source.astype(str).str.contains('twap',case=False,na=False)]
    return m.drop_duplicates('start_time')

def normalize_twap(frame):
    if not {'timestamp','price'}<=set(frame): raise ValueError('TWAP file requires timestamp,price; use actual 60-second Chainlink TWAP reports, not spot-derived averages')
    x=frame[['timestamp','price']].copy()
    x['timestamp']=pd.to_datetime(x.timestamp,utc=True,errors='coerce')
    x['price']=pd.to_numeric(x.price,errors='coerce')
    return x.dropna().sort_values('timestamp').drop_duplicates('timestamp')

def boundary_prices(markets,twap,tolerance_seconds=2):
    t=normalize_twap(twap)
    m=normalize_markets(markets)
    if m.empty or t.empty:return pd.DataFrame()
    out=m.copy()
    for boundary,label in [('start_time','twap_open'),('end_time','twap_close')]:
        matches=pd.merge_asof(m[[boundary]].sort_values(boundary),t,left_on=boundary,right_on='timestamp',direction='backward',tolerance=pd.Timedelta(seconds=tolerance_seconds))
        matches=matches.set_index(boundary)['price']
        out[label]=out[boundary].map(matches)
    out=out.dropna(subset=['twap_open','twap_close']).copy()
    out['twap_up']=(out.twap_close>=out.twap_open).astype(int)
    out['twap_matches_official']=out.twap_up==out.up
    return out

def compare_outcomes(markets,coinbase,twap):
    aligned=boundary_prices(markets,twap)
    if aligned.empty:return aligned
    c=coinbase.close.asfreq('min')
    aligned['coinbase_open_proxy']=(aligned.start_time-pd.Timedelta(minutes=1)).map(c)
    aligned['coinbase_close_proxy']=(aligned.end_time-pd.Timedelta(minutes=1)).map(c)
    aligned=aligned.dropna(subset=['coinbase_open_proxy','coinbase_close_proxy']).copy()
    aligned['coinbase_up']=(aligned.coinbase_close_proxy>aligned.coinbase_open_proxy).astype(int)
    aligned['coinbase_matches_official']=aligned.coinbase_up==aligned.up
    aligned['coinbase_matches_twap']=aligned.coinbase_up==aligned.twap_up
    return aligned

def compare_forecasts(markets,coinbase,twap):
    """Chronological, no-lookahead proxy-model evaluation on official outcomes.
    TWAP-based model requires continuous TWAP history; this function does not pretend otherwise.
    """
    aligned=compare_outcomes(markets,coinbase,twap)
    if len(aligned)<100:return aligned,pd.DataFrame()
    examples=make_examples(coinbase)
    windows=aligned.start_time.sort_values().unique()
    cutoff=windows[int(len(windows)*.7)]
    train=examples[examples.window<cutoff]
    test=examples[(examples.window>=cutoff)&examples.window.isin(windows)]
    test=test.merge(aligned[['start_time','up']],left_on='window',right_on='start_time',suffixes=('_proxy','_official'))
    rows=[]
    for row in test.itertuples():
        result=estimate(train,row.remaining,row.distance,row.momentum,row.vol)
        if result:
            rows.append({'window':row.window,'remaining':row.remaining,'probability':result['probability'],'official_up':row.up_official,'baseline':result['base'],'effective_n':result['effective_n']})
    return aligned,pd.DataFrame(rows)

def summary(aligned,forecasts):
    if aligned.empty:return pd.DataFrame()
    rows=[{'metric':'Coinbase direction vs official outcome','value':aligned.coinbase_matches_official.mean(),'n':len(aligned)},
          {'metric':'Chainlink TWAP direction vs official outcome','value':aligned.twap_matches_official.mean(),'n':len(aligned)},
          {'metric':'Coinbase direction vs Chainlink TWAP','value':aligned.coinbase_matches_twap.mean(),'n':len(aligned)}]
    if not forecasts.empty:
        for horizon,g in forecasts.groupby('remaining'):
            p=g.probability.to_numpy(); y=g.official_up.to_numpy(); b=g.baseline.to_numpy()
            rows.extend([{'metric':f'Proxy model Brier vs official ({horizon}m left)','value':np.mean((p-y)**2),'n':len(g)},
                         {'metric':f'Baseline Brier vs official ({horizon}m left)','value':np.mean((b-y)**2),'n':len(g)},
                         {'metric':f'Proxy model accuracy vs official ({horizon}m left)','value':np.mean((p>=.5)==y),'n':len(g)}])
    return pd.DataFrame(rows)
