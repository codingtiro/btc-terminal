"""Historical BTC 15-minute research model. No trade recommendations."""
import math
import numpy as np
import pandas as pd
import requests

API='https://api.exchange.coinbase.com/products/BTC-USD/candles'

def fetch_history(days=14):
    """Fetch completed one-minute Coinbase candles in bounded 300-candle pages."""
    end=pd.Timestamp.now(tz='UTC').floor('min')-pd.Timedelta(minutes=1)
    start=end-pd.Timedelta(days=days)
    chunks=[]
    with requests.Session() as session:
        cursor=start
        while cursor<end:
            nxt=min(cursor+pd.Timedelta(minutes=299),end)
            response=session.get(API,params={'granularity':60,'start':cursor.isoformat(),'end':nxt.isoformat()},timeout=15)
            response.raise_for_status()
            chunks.extend(response.json())
            cursor=nxt+pd.Timedelta(minutes=1)
    if not chunks: raise ValueError('No historical candles returned')
    df=pd.DataFrame(chunks,columns=['unix','low','high','open','close','volume'])
    df['time']=pd.to_datetime(df.unix,unit='s',utc=True)
    df=df.drop_duplicates('time').sort_values('time').set_index('time')
    df=df[(df.index>=start)&(df.index<=end)]
    return df[['open','high','low','close','volume']].astype(float)

def make_examples(candles):
    """Create past snapshots using only information available at each snapshot.
    Strike proxy: last completed close before each quarter-hour window.
    Label: last completed close of that 15-minute window, NOT Kalshi settlement.
    """
    close=candles.close.asfreq('min')
    ret=np.log(close/close.shift(1))
    rv=ret.rolling(60,min_periods=45).std()*np.sqrt(60)
    momentum=np.log(close/close.shift(5))
    records=[]
    for t in close.index[close.index.minute%15==0]:
        prev=t-pd.Timedelta(minutes=1)
        finish=t+pd.Timedelta(minutes=14)
        if prev not in close.index or finish not in close.index: continue
        strike=close.get(prev)
        final=close.get(finish)
        if not np.isfinite(strike) or not np.isfinite(final): continue
        # Each minute snapshot is the close of the most recently completed minute.
        for elapsed in (0,3,6,9,12):
            at=prev+pd.Timedelta(minutes=elapsed)
            current=close.get(at)
            vol=rv.get(at)
            mom=momentum.get(at)
            if not all(np.isfinite(v) for v in (current,vol,mom)) or vol<=0: continue
            records.append({'window':t,'at':at,'remaining':15-elapsed,
                'distance':math.log(current/strike)/vol,
                'momentum':mom/vol,'vol':vol,
                'up':int(final>strike),'strike':strike,'final':final})
    return pd.DataFrame(records)

def estimate(train,remaining,distance,momentum,vol,minimum=40):
    """Kernel-weighted analog outcomes with beta prior and effective sample size."""
    if len(train)<minimum: return None
    r=train[train.remaining==remaining]
    if len(r)<minimum: return None
    # Relative to realized 60-minute volatility, all inputs are dimensionless.
    d=((r.distance.to_numpy()-distance)/0.24)**2
    m=((r.momentum.to_numpy()-momentum)/0.75)**2
    v=(np.log(r.vol.to_numpy()/vol)/0.65)**2
    w=np.exp(-0.5*(d+m+v))
    weight=w.sum()
    if weight<=1e-10:return None
    n_eff=weight**2/np.sum(w*w)
    if n_eff<12:return None
    raw=float(np.dot(w,r.up.to_numpy())/weight)
    # Conservative shrinkage toward the unconditional rate in the same remaining-time bucket.
    base=float(r.up.mean())
    prob=(n_eff*raw+30*base)/(n_eff+30)
    return {'probability':prob,'raw':raw,'base':base,'effective_n':n_eff,'rows':len(r)}

def snapshot(candles,strike,remaining):
    close=candles.close.asfreq('min')
    ret=np.log(close/close.shift(1))
    vol=float((ret.rolling(60,min_periods=45).std()*np.sqrt(60)).iloc[-1])
    current=float(close.iloc[-1]); mom=float(np.log(current/close.iloc[-6]))
    if not all(np.isfinite(v) for v in (vol,current,mom)) or vol<=0: raise ValueError('Insufficient contiguous recent data')
    return {'current':current,'distance':math.log(current/strike)/vol,'momentum':mom/vol,'vol':vol,'remaining':remaining}

def validate(examples):
    """Chronological 70/30 split, one prediction per held-out window at each horizon."""
    if examples.empty:return pd.DataFrame()
    windows=sorted(examples.window.unique())
    cut=int(len(windows)*.7)
    if cut<50 or len(windows)-cut<20:return pd.DataFrame()
    train=examples[examples.window.isin(windows[:cut])]
    test=examples[examples.window.isin(windows[cut:])]
    results=[]
    for horizon in (15,12,9,6,3):
        subset=test[test.remaining==horizon]
        predicted=[];actual=[];baseline=[]
        for row in subset.itertuples():
            result=estimate(train,horizon,row.distance,row.momentum,row.vol)
            if result:
                predicted.append(result['probability']);actual.append(row.up);baseline.append(result['base'])
        if actual:
            a=np.asarray(actual);p=np.asarray(predicted);b=np.asarray(baseline)
            results.append({'minutes_left':horizon,'held_out_windows':len(a),
                'model_brier':float(np.mean((p-a)**2)),
                'baseline_brier':float(np.mean((b-a)**2)),
                'model_accuracy':float(np.mean((p>=.5)==a)),
                'baseline_accuracy':float(np.mean((b>=.5)==a))})
    return pd.DataFrame(results)
