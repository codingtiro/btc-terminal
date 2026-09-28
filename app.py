"""BTC historical-probability terminal. Research only; no order execution."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from model import fetch_history,make_examples,estimate,snapshot,validate
from chainlink import latest_quote

st.set_page_config(page_title='BTC Historical Prediction Terminal',page_icon='₿',layout='wide')
st.title('₿ BTC Polymarket Prediction Terminal')
st.caption('Chainlink live reference when available • Coinbase historical proxy • Research only')
if st.button('Fetch live Chainlink BTC/USD quote'):
    try: st.session_state.chainlink_quote=latest_quote()
    except Exception as exc: st.error(f'Chainlink feed unavailable: {exc}. No settlement-aligned quote is shown.')
if 'chainlink_quote' in st.session_state:
    q=st.session_state.chainlink_quote
    st.metric('Chainlink BTC/USD (last fetched)',f"${q['price']:,.2f}")
    st.caption(f"Source: {q['source']} • Feed timestamp: {q['timestamp']} • Refresh manually; do not use a stale quote.")
st.caption('Historical analog probabilities • Chronological holdout testing • Research only')
with st.sidebar:
    days=st.select_slider('Coinbase history to download (days)',options=[3,7,14,30],value=14)
    st.caption('Initial download can take time. Cached for 10 minutes.')
    strike=st.number_input('Exact contract strike/reference ($)',min_value=1.,value=100000.,step=10.)
    remaining=st.selectbox('Minutes remaining (completed-minute snapshots)',[15,12,9,6,3])
    st.warning('Enter the exact Polymarket Price to Beat and remaining minutes. Historical training uses Coinbase as a PROXY; live Chainlink quotes alone do not make the forecast settlement-validated.')
@st.cache_data(ttl=600,show_spinner='Downloading historical candles...')
def history(days):return fetch_history(days)
@st.cache_data(ttl=600,show_spinner='Building historical analogs and evaluating holdout...')
def prepared(days):
    candles=history(days)
    examples=make_examples(candles)
    return candles,examples,validate(examples)
try:
    candles,examples,metrics=prepared(days)
    s=snapshot(candles,strike,remaining)
    # Never train live estimates on data newer than the most recent fully completed historical window.
    complete_before=candles.index[-1].floor('15min')
    train=examples[examples.window<complete_before]
    prediction=estimate(train,remaining,s['distance'],s['momentum'],s['vol'])
    st.metric('Coinbase BTC/USD (historical proxy, last completed 1m candle)',f"${s['current']:,.2f}")
    st.caption(f"Last candle: {candles.index[-1]} • Complete historical windows: {train.window.nunique():,} • Proxy: Coinbase candle closes")
    fig=go.Figure(data=[go.Candlestick(x=candles.index[-180:],open=candles.open.iloc[-180:],high=candles.high.iloc[-180:],low=candles.low.iloc[-180:],close=candles.close.iloc[-180:])])
    fig.add_hline(y=strike,line_dash='dot',annotation_text='User-entered strike')
    fig.update_layout(template='plotly_dark',height=390,xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)
    st.subheader('Historical analog forecast')
    if prediction:
        a,b,c=st.columns(3)
        a.metric('Historical-model UP probability',f"{prediction['probability']:.1%}")
        b.metric('Historical-model DOWN probability',f"{1-prediction['probability']:.1%}")
        c.metric('Effective historical sample',f"{prediction['effective_n']:.0f}")
        st.caption(f"Raw weighted UP rate: {prediction['raw']:.1%} • Same-horizon historical base rate: {prediction['base']:.1%} • Candidate historical snapshots: {prediction['rows']:,}. Estimates shrink toward base rates when analog evidence is weak.")
        if prediction['effective_n']<30:st.warning('Low effective sample: probability is unstable. Treat as insufficient evidence.')
    else:st.warning('Insufficient comparable historical examples for a reliable estimate. Try a longer history or a different horizon.')
    st.subheader('Out-of-sample historical evaluation')
    if not metrics.empty:
        display=metrics.copy()
        display['model_accuracy']=display.model_accuracy.map(lambda v:f'{v:.1%}')
        display['baseline_accuracy']=display.baseline_accuracy.map(lambda v:f'{v:.1%}')
        st.dataframe(display,hide_index=True,use_container_width=True)
        st.caption('First 70% of complete windows train the model; last 30% test it chronologically. Lower Brier score is better. Accuracy is directional classification, not expected profitability. A model worse than its baseline should not be trusted.')
        st.download_button('Download validation CSV',metrics.to_csv(index=False),'validation.csv','text/csv')
    else:st.warning('Not enough completed windows for holdout validation. Increase historical range.')
    st.subheader('Historical evidence')
    st.write('The model compares the current price-to-strike distance, 5-minute momentum and realized volatility with past snapshots at the same remaining time. It uses kernel-weighted outcomes and conservative shrinkage, rather than treating the closest few examples as certainty.')
    st.download_button('Download historical snapshots CSV',examples.to_csv(index=False),'historical_snapshots.csv','text/csv')
    st.info('IMPORTANT: Coinbase-trained probabilities are NOT verified Polymarket settlement probabilities. Chainlink live reference is shown separately, not substituted into a Coinbase-trained model. Before relying on signals, collect and backtest historical Chainlink feed and official resolved outcomes. No automated trades.')
except Exception as exc:
    st.error(f'Could not produce a validated forecast: {exc}')

st.divider()
st.header('Chainlink TWAP vs Coinbase: official-outcome comparison')
st.caption('Upload actual Chainlink BTC/USD 60s TWAP ticks and official resolved 15-minute market records. Nothing is scored without both. TWAP ticks must be authentic, not Coinbase-derived.')
market_upload=st.file_uploader('Official market CSV (start_time,end_time,winner,strike; optional twapLookbackSeconds,resolution_source)',type=['csv'],key='market_comparison')
twap_upload=st.file_uploader('Chainlink 60-second TWAP CSV (timestamp,price)',type=['csv'],key='twap_comparison')
if market_upload is not None and twap_upload is not None:
    try:
        from compare import compare_forecasts,summary
        with st.spinner('Aligning actual settlement reports and evaluating chronological forecasts...'):
            m=pd.read_csv(market_upload); t=pd.read_csv(twap_upload)
            candles=history(days)
            aligned,forecast=compare_forecasts(m,candles,t)
            report=summary(aligned,forecast)
        if aligned.empty:st.error('No matching market windows with authentic TWAP ticks within 2 seconds of both boundaries and complete Coinbase proxy candles. No comparison can be claimed.')
        else:
            st.dataframe(report,hide_index=True,use_container_width=True)
            st.caption('TWAP vs official outcome is a settlement-source consistency check, NOT a forecast accuracy score. The model is Coinbase-trained and evaluated on held-out official outcomes. No TWAP-trained model is claimed here.')
            st.download_button('Download comparison summary',report.to_csv(index=False),'comparison_summary.csv')
            st.download_button('Download aligned official outcomes',aligned.to_csv(index=False),'aligned_outcomes.csv')
            if not forecast.empty:st.download_button('Download held-out predictions',forecast.to_csv(index=False),'heldout_predictions.csv')
            else:st.warning('Fewer than 100 aligned market windows: no credible held-out model evaluation. Import more data.')
    except Exception as exc:st.error(f'Comparison not available: {exc}')
else:st.info('Comparison awaiting actual Chainlink TWAP history and official Polymarket outcomes. Do not interpret the Coinbase proxy as settlement-validated.')
