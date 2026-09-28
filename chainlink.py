"""Best-effort Polymarket RTDS Chainlink quote; no historical backfill."""
import json,time
import websocket
URL='wss://ws-live-data.polymarket.com'
SUB={'action':'subscribe','subscriptions':[{'topic':'crypto_prices_chainlink','type':'*','filters':json.dumps({'symbol':'btc/usd'})}]}
def latest_quote(timeout=8):
    ws=websocket.create_connection(URL,timeout=timeout)
    try:
        ws.send(json.dumps(SUB))
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            try: obj=json.loads(ws.recv())
            except (ValueError,websocket.WebSocketTimeoutException):continue
            for msg in obj if isinstance(obj,list) else [obj]:
                if not isinstance(msg,dict):continue
                if msg.get('topic')!='crypto_prices_chainlink':continue
                payload=msg.get('payload',{})
                if isinstance(payload,dict) and payload.get('symbol','').lower()=='btc/usd':
                    value=payload.get('value') or payload.get('price')
                    if value is not None:return {'price':float(value),'timestamp':payload.get('timestamp'),'source':'Polymarket RTDS Chainlink btc/usd'}
        raise TimeoutError('No BTC/USD Chainlink quote received before timeout')
    finally:ws.close()
