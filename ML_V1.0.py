# -*- coding: utf-8 -*-
"""
多数据源多特征版：Binance Public Data + LSTM/GRU
- 主数据：klines（按 --granularity 下载）
- 辅助：markPriceKlines / indexPriceKlines / premiumIndexKlines（按 --granularity）
- 资金费率 fundingRate：强制按 monthly 下载（不受 --granularity 影响）
- 统一解析 K线类 12 列；资金费率单独解析并对齐到目标 K 线频率
- 构造特征：价差/溢价、趋势/波动/量能、资金费率变化等
- 仅用训练集拟合标准化器；评估含方向准确率/相关/R²、与“恒为0”基线对比
"""

import os, io, math, zipfile, argparse, warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import requests

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ===== 常量与路径 =====
BASE_URL = "https://data.binance.vision"
MARKET_ROOT = {"spot":"spot","futures-um":"futures/um","futures-cm":"futures/cm"}
KLINE_COLS = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","num_trades",
    "taker_buy_base","taker_buy_quote","ignore"
]

def _path(market:str, granularity:str, subdir:str, symbol:str, interval:str=None):
    root = MARKET_ROOT[market]
    if interval is None:
        return f"data/{root}/{granularity}/{subdir}/{symbol}"
    return f"data/{root}/{granularity}/{subdir}/{symbol}/{interval}"

def _fname(symbol:str, interval:str, datestr:str):
    # K线家族文件名
    return f"{symbol}-{interval}-{datestr}.zip"

def _fr_fname_monthly(symbol: str, ym: str):
    # fundingRate 的正确命名：SYMBOL-fundingRate-YYYY-MM.zip
    return f"{symbol}-fundingRate-{ym}.zip"

def _url(path:str, fname:str):
    return f"{BASE_URL}/{path}/{fname}"

def _dates(granularity:str, start:str, end:str):
    if granularity=="daily":
        return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]
    elif granularity=="monthly":
        return [p.strftime("%Y-%m") for p in pd.period_range(start[:7], end[:7], freq="M")]
    else:
        raise ValueError("granularity must be daily or monthly")

# ===== 通用：读取 zip 内 csv（K线 12 列）=====
def read_kline_like_zip(zbytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs: raise ValueError("zip 内未见 csv")
        name = csvs[0]
        # 尝试 header=None
        with zf.open(name) as f:
            df = pd.read_csv(f, header=None)
        if df.shape[1] != 12:
            with zf.open(name) as f:
                df = pd.read_csv(f, header=0)
        if df.shape[1] != 12:
            raise ValueError(f"列数{df.shape[1]}!=12")
        df.columns = KLINE_COLS
        df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
        df = df[df["open_time"].notna()].reset_index(drop=True)
        return df

# ===== 时间戳（自动毫秒/微秒）=====
def ts_to_dt(ts:int)->pd.Timestamp:
    ts = int(ts); unit = "us" if ts >= 10**15 else "ms"
    return pd.to_datetime(ts, unit=unit, utc=True)

# ===== 下载器：K线家族（按命令 granularity）=====
def download_kline_family(market:str, symbol:str, interval:str,
                          start:str, end:str, granularity:str, subdir:str,
                          cache_dir:str)->pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    sess = requests.Session(); frames=[]
    dlist = _dates(granularity, start, end)
    base = _path(market, granularity, subdir, symbol, interval)
    for ds in dlist:
        url = _url(base, _fname(symbol, interval, ds))
        cache = os.path.join(cache_dir, f"{subdir}-{symbol}-{interval}-{ds}.zip")
        try:
            if not os.path.exists(cache):
                r = sess.get(url, timeout=60)
                if r.status_code==404:
                    warnings.warn(f"[缺失]{url}"); continue
                r.raise_for_status()
                with open(cache,"wb") as f:
                    f.write(r.content)
            with open(cache,"rb") as f:
                frames.append(read_kline_like_zip(f.read()))
        except Exception as e:
            warnings.warn(f"[跳过]{url} -> {e}")
    if not frames: return pd.DataFrame(columns=KLINE_COLS)

    df = pd.concat(frames, ignore_index=True)
    # 数值化
    for c in ["open_time","open","high","low","close","volume",
              "quote_asset_volume","num_trades","taker_buy_base","taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open_time"]).copy()
    df["open_time"] = df["open_time"].astype("int64").apply(ts_to_dt)
    df = df.sort_values("open_time").drop_duplicates("open_time").set_index("open_time")
    return df

# ===== 资金费率：强制 monthly =====
def read_funding_zip(zbytes: bytes)->pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs: raise ValueError("zip 内未见 csv")
        name = csvs[0]
        with zf.open(name) as f:
            df0 = pd.read_csv(f, header=0)
        if df0.shape[1] < 2:
            with zf.open(name) as f:
                df0 = pd.read_csv(f, header=None)

        df = df0.copy()
        # 猜列名
        cols = [str(c).lower() for c in df.columns]
        if len(df.columns) >= 3 and ("fundingrate" in "".join(cols) or "funding_rate" in "".join(cols)):
            mapper = {}
            for c in df.columns:
                lc = str(c).lower()
                if "fundingrate" in lc: mapper[c] = "funding_rate"
                elif "funding_time" in lc or "fundingtime" in lc or lc in ("time","timestamp"):
                    mapper[c] = "funding_time"
                elif "symbol" in lc: mapper[c] = "symbol"
            df = df.rename(columns=mapper)
        else:
            if df.shape[1] < 3:
                raise ValueError("fundingRate 列无法识别")
            df.columns = ["symbol","funding_time","funding_rate"] + list(df.columns[3:])

        df["funding_time"] = pd.to_numeric(df["funding_time"], errors="coerce")
        df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
        df = df.dropna(subset=["funding_time","funding_rate"])
        df["time"] = df["funding_time"].astype("int64").apply(ts_to_dt)
        df = df.set_index("time")[["funding_rate"]].sort_index()
        return df

def download_funding_rate(market:str, symbol:str,
                          start:str, end:str,
                          _granularity_ignored: str,  # 保持签名不变，但忽略
                          cache_dir:str)->pd.DataFrame:
    """
    fundingRate 仅从 monthly 读取；其它数据仍按命令 granularity。
    """
    os.makedirs(cache_dir, exist_ok=True)
    sess = requests.Session(); frames=[]
    months = [p.strftime("%Y-%m") for p in pd.period_range(start[:7], end[:7], freq="M")]
    path = _path(market, "monthly", "fundingRate", symbol, interval=None)

    for ym in months:
        url = _url(path, _fr_fname_monthly(symbol, ym))
        cache = os.path.join(cache_dir, f"fundingRate-{symbol}-{ym}.zip")
        try:
            if not os.path.exists(cache):
                r = sess.get(url, timeout=60)
                if r.status_code == 404:
                    warnings.warn(f"[缺失]{url}"); continue
                r.raise_for_status()
                with open(cache, "wb") as f:
                    f.write(r.content)
            with open(cache, "rb") as f:
                frames.append(read_funding_zip(f.read()))
        except Exception as e:
            warnings.warn(f"[跳过]{url} -> {e}")

    if not frames:
        return pd.DataFrame(columns=["funding_rate"])

    df = pd.concat(frames).sort_index().drop_duplicates()
    return df

# ===== 构造多源数据并对齐到主 K 线 =====
def build_multisource_frame(market, symbol, interval, start, end,
                            granularity, cache_dir,
                            use_mark, use_index, use_premium, use_funding) -> pd.DataFrame:
    # 主 K 线
    k = download_kline_family(market, symbol, interval, start, end, granularity, "klines", cache_dir)
    if k.empty: raise RuntimeError("主 K线为空")
    df = k.rename(columns={"close":"last_close","volume":"last_vol"})
    df["logret"] = np.log(df["last_close"]).diff()

    # 可选 K 线数据源（按命令 granularity）
    if use_mark:
        mk = download_kline_family(market, symbol, interval, start, end, granularity, "markPriceKlines", cache_dir)
        if not mk.empty: df = df.join(mk["close"].rename("mark_close"))
    if use_index:
        ix = download_kline_family(market, symbol, interval, start, end, granularity, "indexPriceKlines", cache_dir)
        if not ix.empty: df = df.join(ix["close"].rename("index_close"))
    if use_premium:
        pi = download_kline_family(market, symbol, interval, start, end, granularity, "premiumIndexKlines", cache_dir)
        if not pi.empty: df = df.join(pi["close"].rename("premium_index"))
    if use_funding:
        fr = download_funding_rate(market, symbol, start, end, granularity, cache_dir)  # 强制 monthly
        if not fr.empty:
            # 对齐到 K 线索引（使用最近一次资金费率）
            fr_rs = fr.reindex(df.index, method="ffill")
            df = df.join(fr_rs.rename(columns={"funding_rate":"funding_rate"}))

    # 价差/溢价特征
    if "mark_close" in df:   df["spread_mark_last"]  = (df["mark_close"] / df["last_close"] - 1.0)
    if "index_close" in df:  df["basis_last_index"]  = (df["last_close"] / df["index_close"] - 1.0)
    if "premium_index" in df: df["premium_index_chg"] = df["premium_index"].pct_change()

    # 技术指标（简版）
    def _roll(name, s, w, fn):
        try: df[name] = fn(s.rolling(w))
        except: df[name] = np.nan

    # 趋势
    for w in (5,12,24,48):
        _roll(f"ma_{w}", df["last_close"], w, lambda r: r.mean())
    df["ema_12"] = df["last_close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["last_close"].ewm(span=26, adjust=False).mean()
    df["macd"]   = df["ema_12"] - df["ema_26"]
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # 波动/区间（近似 TR 与 rolling 波动）
    tr = (df["last_close"].combine(df["last_close"].shift(), max) - 
          df["last_close"].combine(df["last_close"].shift(), min)).abs()
    _roll("atr_14", tr, 14, lambda r: r.mean())
    ret = df["last_close"].pct_change()
    for w in (12, 24, 48):
        # 滚动标准差 * sqrt(window)，避免全 NaN
        df[f"rv_{w}"] = ret.rolling(window=w, min_periods=w).std() * np.sqrt(w)

    # 量能/主动性
    if "last_vol" in df:
        for w in (12,24,48):
            _roll(f"vol_ma_{w}", df["last_vol"], w, lambda r: r.mean())
        df["vol_chg"] = df["last_vol"].pct_change()
    if "taker_buy_base" in df:
        df["taker_buy_ratio"] = df["taker_buy_base"] / (df["last_vol"] + 1e-12)

    # 资金费率派生
    if "funding_rate" in df:
        df["funding_abs"] = df["funding_rate"].astype(float)
        df["funding_chg"] = df["funding_abs"].diff()

    return df

# ===== 监督样本构造 =====
def make_supervised(df: pd.DataFrame, seq_len=128, horizon=1, feature_cols:List[str]=None):
    if feature_cols is None:
        base_exclude = set(["close_time","ignore"])
        feature_cols = [c for c in df.columns if c not in base_exclude]
        if "y" in feature_cols: feature_cols.remove("y")

    if "last_close" not in df: raise RuntimeError("缺少 last_close")
    y = (df["last_close"].shift(-horizon) / df["last_close"] - 1.0).rename("y")

    X = df.copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    data = X.join(y).dropna().copy()

    X_np = data[feature_cols].values.astype(np.float32)
    y_np = data["y"].values.astype(np.float32).reshape(-1,1)

    X_seq, y_seq = [], []
    N, F = X_np.shape
    for i in range(N - seq_len - horizon + 1):
        X_seq.append(X_np[i:i+seq_len, :])
        y_seq.append(y_np[i+seq_len-1, :])
    X_seq = np.asarray(X_seq, dtype=np.float32)  # [S,T,F]
    y_seq = np.asarray(y_seq, dtype=np.float32)  # [S,1]
    return X_seq, y_seq, feature_cols

# ===== 数据集/模型/训练 =====
class SeqDataset(Dataset):
    def __init__(self, X, y): self.X=torch.from_numpy(X); self.y=torch.from_numpy(y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self,i): return self.X[i], self.y[i]

class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.2, model_type="gru"):
        super().__init__()
        if model_type=="gru":
            self.rnn = nn.GRU(input_size, hidden, num_layers=layers,
                              dropout=dropout if layers>1 else 0.0, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size, hidden, num_layers=layers,
                               dropout=dropout if layers>1 else 0.0, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden,hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden,1))
    def forward(self,x):
        out,_=self.rnn(x); return self.head(out[:,-1,:])

def time_split(X,y,tr=0.7,va=0.15):
    N=len(X); n_tr=int(N*tr); n_va=int(N*va)
    return (X[:n_tr],y[:n_tr]), (X[n_tr:n_tr+n_va],y[n_tr:n_tr+n_va]), (X[n_tr+n_va:],y[n_tr+n_va:])

def fit_transform_seq_by_feature(Xtr,Xva,Xte):
    S_tr,T,F=Xtr.shape; S_va=Xva.shape[0]; S_te=Xte.shape[0]
    scaler = StandardScaler().fit(Xtr.reshape(-1,F))
    def tf(X,S): return scaler.transform(X.reshape(-1,F)).reshape(S,T,F)
    return tf(Xtr,S_tr), tf(Xva,S_va), tf(Xte,S_te), scaler

def train_model(X,y,model_type="gru",epochs=40,batch=512,lr=1e-3,hidden=128,layers=2,dropout=0.2,device=None):
    (Xtr_raw,ytr),(Xva_raw,yva),(Xte_raw,yte) = time_split(X,y)
    Xtr,Xva,Xte,scaler = fit_transform_seq_by_feature(Xtr_raw,Xva_raw,Xte_raw)
    tr_dl=DataLoader(SeqDataset(Xtr,ytr),batch_size=batch,shuffle=False,drop_last=True)
    va_dl=DataLoader(SeqDataset(Xva,yva),batch_size=batch,shuffle=False)
    te_dl=DataLoader(SeqDataset(Xte,yte),batch_size=batch,shuffle=False)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNRegressor(input_size=X.shape[2], hidden=hidden, layers=layers, dropout=dropout, model_type=model_type).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr); loss_fn=nn.MSELoss()

    best_va=math.inf; patience=10; bad=0; best_state=None
    for ep in range(1,epochs+1):
        model.train(); tr_loss=0.0
        for xb,yb in tr_dl:
            xb,yb=xb.to(device),yb.to(device)
            pred=model(xb); loss=loss_fn(pred,yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()*xb.size(0)
        tr_loss/=len(tr_dl.dataset)

        model.eval(); va_loss=0.0
        with torch.no_grad():
            for xb,yb in va_dl:
                xb,yb=xb.to(device),yb.to(device)
                va_loss += loss_fn(model(xb),yb).item()*xb.size(0)
        va_loss/=len(va_dl.dataset)
        print(f"[{ep:03d}/{epochs}] train_mse={tr_loss:.6f} val_mse={va_loss:.6f}")

        if va_loss + 1e-12 < best_va:
            best_va=va_loss; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=patience:
                print("Early stopping."); break

    if best_state is not None: model.load_state_dict(best_state)

    # 测试评估
    model.eval(); preds=[]; trues=[]
    with torch.no_grad():
        for xb,yb in te_dl:
            xb=xb.to(device)
            yhat=model(xb).cpu().numpy()
            preds.append(yhat); trues.append(yb.numpy())
    y_pred=np.vstack(preds).ravel(); y_true=np.vstack(trues).ravel()

    mae=float(np.mean(np.abs(y_pred-y_true)))
    rmse=float(np.sqrt(np.mean((y_pred-y_true)**2)))
    mape=float(np.mean(np.abs((y_pred-y_true)/(np.abs(y_true)+1e-8)))*100.0)

    def sgn(a): a=a.copy(); a[np.abs(a)<1e-12]=0.0; return np.sign(a)
    dir_acc=float(np.mean(sgn(y_pred)==sgn(y_true)))
    corr=float(np.corrcoef(y_pred,y_true)[0,1]) if len(y_pred)>1 else np.nan
    r2=float(r2_score(y_true,y_pred)) if len(y_pred)>1 else np.nan
    rmse0=float(np.sqrt(np.mean((0.0-y_true)**2))); mae0=float(np.mean(np.abs(0.0-y_true)))
    d_rmse=1.0 - rmse/(rmse0+1e-12); d_mae=1.0 - mae/(mae0+1e-12)

    print(f"[TEST] MAE={mae:.6f} RMSE={rmse:.6f} MAPE={mape:.6f}% | DIR_ACC={dir_acc:.3f} CORR={corr:.3f} R2={r2:.3f}")
    print(f"[BASELINE zero] MAE={mae0:.6f} RMSE={rmse0:.6f} | ΔRMSE={d_rmse:+.3%} ΔMAE={d_mae:+.3%}")
    return model, dict(MAE=mae,RMSE=rmse,MAPE=mape,DIR_ACC=dir_acc,CORR=corr,R2=r2,
                       BASELINE_MAE=mae0,BASELINE_RMSE=rmse0,DELTA_RMSE=d_rmse,DELTA_MAE=d_mae)

# ===== 主流程 =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="futures-um", choices=["spot","futures-um","futures-cm"])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", required=True)  # YYYY-MM-DD
    ap.add_argument("--end", required=True)
    ap.add_argument("--granularity", default="daily", choices=["daily","monthly"])
    ap.add_argument("--cache-dir", default="data_cache")

    # 辅助数据源
    ap.add_argument("--use-mark", action="store_true")
    ap.add_argument("--use-index", action="store_true")
    ap.add_argument("--use-premium", action="store_true")
    ap.add_argument("--use-funding", action="store_true")

    # 模型与训练
    ap.add_argument("--model", default="gru", choices=["gru","lstm"])
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)

    args = ap.parse_args()
    warnings.simplefilter("once")
    pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)

    print(f"== 下载 {args.market} {args.symbol} {args.interval} {args.start}~{args.end} ==")
    df = build_multisource_frame(
        args.market, args.symbol, args.interval, args.start, args.end,
        args.granularity, args.cache_dir,
        args.use_mark, args.use_index, args.use_premium, args.use_funding
    )

    # 选择特征列（可自行裁剪/增删）
    feature_cols = [c for c in df.columns if c not in ["close_time","ignore"]]
    print(f"[DEBUG] 原始行数: {len(df)}")
    nan_ratio = df.isna().mean().sort_values(ascending=False)
    print("[DEBUG] 缺失率最高的前 15 个特征:")
    print(nan_ratio.head(15))
    X, y, used_cols = make_supervised(df[feature_cols], seq_len=args.seq_len, horizon=args.horizon, feature_cols=feature_cols)
    print(f"Samples={len(X)}, Seq={args.seq_len}, Features={len(used_cols)}")
    print("Head features:", used_cols[:10])

    _model, _metrics = train_model(
        X, y, model_type=args.model, epochs=args.epochs, batch=args.batch, lr=args.lr,
        hidden=args.hidden, layers=args.layers, dropout=args.dropout
    )

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    warnings.simplefilter("once")
    main()
