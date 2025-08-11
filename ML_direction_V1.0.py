# -*- coding: utf-8 -*-
"""
ML_direction_V1.3.py

功能概览
- 数据：klines(主)、markPriceKlines、indexPriceKlines、premiumIndexKlines、fundingRate(月)、metrics(日, 含 OI/多空比)
- 多币种、多时间尺度（主 interval + --extra-intervals）
- 特征：价量/动量/波动率/基差/溢价/资金费率/主动性 + OI 与多空比；新增 交互/布尔/极端值zscore 特征
- 序列建模：GRU/LSTM，recurrent_dropout + L2 正则
- 特征选择：仅训练集；稳定性筛选（多窗口投票）+ TopK
- 评估：阈值表、覆盖–准确率、PR/AP；导出高置信度信号CSV
- （可选）Walk-Forward 时间滚动评估

示例命令（Windows 一行）：
python ML_direction_V1.3.py --market futures-um --symbols BTCUSDT,ETHUSDT --interval 1h --extra-intervals 15m,4h --granularity daily --use-mark --use-index --use-premium --use-funding --use-metrics --model gru --seq-len 128 --horizon 4 --epochs 40 --batch 256 --topk 80 --start 2022-01-01 --lr 0.002 --recurrent-dropout 0.1 --l2 1e-5 --stable-windows 3 --stable-votes 2
"""

import os, io, zipfile, warnings, argparse, time
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import requests

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, average_precision_score

import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2 as L2

# -------------------- 常量 --------------------
BASE_URL = "https://data.binance.vision"
MARKET_ROOT = {"spot":"spot","futures-um":"futures/um","futures-cm":"futures/cm"}
KLINE_COLS = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","num_trades",
    "taker_buy_base","taker_buy_quote","ignore"
]
DEFAULT_BEGIN = "2020-01-01"
CACHE_DIR = "data_cache"
EPS = 1e-12

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("reports", exist_ok=True)
warnings.simplefilter("once")
pd.set_option("display.width", 180); pd.set_option("display.max_columns", 220)

# -------------------- 小工具 --------------------
def _path(market:str, granularity:str, subdir:str, symbol:str, interval:str=None):
    root = MARKET_ROOT[market]
    if interval is None:
        return f"data/{root}/{granularity}/{subdir}/{symbol}"
    return f"data/{root}/{granularity}/{subdir}/{symbol}/{interval}"

def _fname(symbol:str, interval:str, datestr:str): return f"{symbol}-{interval}-{datestr}.zip"
def _fr_fname_monthly(symbol: str, ym: str): return f"{symbol}-fundingRate-{ym}.zip"
def _url(path:str, fname:str): return f"{BASE_URL}/{path}/{fname}"

def _dates(granularity:str, start:str, end:str):
    if granularity=="daily":
        return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]
    elif granularity=="monthly":
        return [p.strftime("%Y-%m") for p in pd.period_range(start[:7], end[:7], freq="M")]
    else:
        raise ValueError("granularity must be daily/monthly")

def ts_to_dt(ts:int)->pd.Timestamp:
    t = int(ts); unit = "us" if t >= 10**15 else "ms"
    return pd.to_datetime(t, unit=unit, utc=True)

# -------------------- 读取 zip 内 CSV（K线12列） --------------------
def read_kline_like_zip(zbytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs: raise ValueError("zip 内未见 csv")
        name = csvs[0]
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

# -------------------- 通用下载（klines / mark / index / premium） --------------------
def _download_series(market:str, symbol:str, interval:str, start:str, end:str,
                     granularity:str, subdir:str) -> pd.DataFrame:
    sess = requests.Session(); frames=[]
    base = _path(market, granularity, subdir, symbol, interval)
    for ds in _dates(granularity, start, end):
        url = _url(base, _fname(symbol, interval, ds))
        cache = os.path.join(CACHE_DIR, f"{subdir}-{symbol}-{interval}-{ds}.zip")
        try:
            if not os.path.exists(cache):
                r = sess.get(url, timeout=30)
                if r.status_code==404:
                    warnings.warn(f"[缺失]{url}"); continue
                r.raise_for_status(); open(cache,"wb").write(r.content)
            frames.append(read_kline_like_zip(open(cache,"rb").read()))
        except Exception as e:
            warnings.warn(f"[跳过]{url} -> {e}")
    if not frames: return pd.DataFrame(columns=KLINE_COLS)
    df = pd.concat(frames, ignore_index=True)
    for c in ["open_time","open","high","low","close","volume",
              "quote_asset_volume","num_trades","taker_buy_base","taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open_time"]).copy()
    df["open_time"] = df["open_time"].astype("int64").apply(ts_to_dt)
    df = df.sort_values("open_time").drop_duplicates("open_time").set_index("open_time")
    return df

def download_klines(market:str, symbol:str, interval:str, start:str, end:str, granularity:str) -> pd.DataFrame:
    return _download_series(market, symbol, interval, start, end, granularity, "klines")

def download_mark_price(market:str, symbol:str, interval:str, start:str, end:str, granularity:str) -> pd.DataFrame:
    d = _download_series(market, symbol, interval, start, end, "daily", "markPriceKlines")
    m = _download_series(market, symbol, interval, start, end, "monthly", "markPriceKlines")
    if d.empty and m.empty: return pd.DataFrame()
    if d.empty: return m
    if m.empty: return d
    return pd.concat([d,m]).sort_index().loc[lambda x: ~x.index.duplicated(keep="first")]

def download_index_price(market:str, symbol:str, interval:str, start:str, end:str, granularity:str) -> pd.DataFrame:
    d = _download_series(market, symbol, interval, start, end, "daily", "indexPriceKlines")
    m = _download_series(market, symbol, interval, start, end, "monthly", "indexPriceKlines")
    if d.empty and m.empty: return pd.DataFrame()
    if d.empty: return m
    if m.empty: return d
    return pd.concat([d,m]).sort_index().loc[lambda x: ~x.index.duplicated(keep="first")]

def download_premium_price(market:str, symbol:str, interval:str, start:str, end:str, granularity:str) -> pd.DataFrame:
    return _download_series(market, symbol, interval, start, end, "daily", "premiumIndexKlines")

# -------------------- fundingRate（月度） --------------------
def read_funding_zip(zbytes: bytes)->pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs: raise ValueError("zip 内未见 csv")
        with zf.open(csvs[0]) as f: df = pd.read_csv(f)
    if df.empty: raise ValueError("fundingRate 文件为空")
    lower = {c: str(c).lower() for c in df.columns}; inv = {v:k for k,v in lower.items()}
    t_candidates = ["funding_time","fundingtime","time","timestamp","open_time","close_time"]
    r_candidates = ["funding_rate","fundingrate","rate"]
    tcol = next((inv[c] for c in t_candidates if c in inv), None)
    rcol = next((inv[c] for c in r_candidates if c in inv), None)
    if tcol is None or rcol is None: raise ValueError("fundingRate 列识别失败")
    df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
    df[rcol] = pd.to_numeric(df[rcol], errors="coerce")
    df = df.dropna(subset=[tcol,rcol])
    df["time"] = df[tcol].astype("int64").apply(ts_to_dt)
    return df.set_index("time")[[rcol]].rename(columns={rcol:"funding_rate"}).sort_index()

def download_funding(market:str, symbol:str, start:str, end:str) -> pd.DataFrame:
    sess = requests.Session(); frames=[]
    path = _path(market, "monthly", "fundingRate", symbol, interval=None)
    for ym in _dates("monthly", start, end):
        url = _url(path, _fr_fname_monthly(symbol, ym))
        cache = os.path.join(CACHE_DIR, f"fundingRate-{symbol}-{ym}.zip")
        try:
            if not os.path.exists(cache):
                r = sess.get(url, timeout=30)
                if r.status_code==404: warnings.warn(f"[缺失]{url}"); continue
                r.raise_for_status(); open(cache,"wb").write(r.content)
            frames.append(read_funding_zip(open(cache,"rb").read()))
        except Exception as e:
            warnings.warn(f"[跳过]{url} -> {e}")
    if not frames: return pd.DataFrame(columns=["funding_rate"])
    return pd.concat(frames).sort_index().drop_duplicates()

# -------------------- metrics（日度，含 OI/多空比） --------------------
def read_metrics_zip(zbytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs: raise ValueError("zip 内未见 csv")
        with zf.open(csvs[0]) as f: df = pd.read_csv(f)
    if df.empty: raise ValueError("metrics 文件为空")
    lower = {c: str(c).lower().strip() for c in df.columns}; inv = {v:k for k,v in lower.items()}
    t_candidates = ["create_time","timestamp","time","open_time","close_time"]
    tcol = next((inv[c] for c in t_candidates if c in inv), None)
    def pick(*names):
        for n in names:
            if n in inv: return inv[n]
        return None
    col_oi   = pick("sum_open_interest","open_interest","sumopeninterest")
    col_oiv  = pick("sum_open_interest_value","open_interest_value","sumopeninterestvalue")
    col_ctls = pick("count_toptrader_long_short_ratio")
    col_stls = pick("sum_toptrader_long_short_ratio")
    col_clsr = pick("count_long_short_ratio")
    col_stlvr= pick("sum_taker_long_short_vol_ratio")
    keep = {}
    if col_oi:   keep["open_interest"] = col_oi
    if col_oiv:  keep["open_interest_value"] = col_oiv
    if col_ctls: keep["count_toptrader_long_short_ratio"] = col_ctls
    if col_stls: keep["sum_toptrader_long_short_ratio"] = col_stls
    if col_clsr: keep["count_long_short_ratio"] = col_clsr
    if col_stlvr:keep["sum_taker_long_short_vol_ratio"] = col_stlvr
    if tcol is None or not keep: raise ValueError("metrics 列识别失败")
    try:
        df[tcol] = pd.to_numeric(df[tcol], errors="raise")
        t = df[tcol].astype("int64").apply(ts_to_dt)
    except Exception:
        t = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    out = df[list(keep.values())].copy()
    out.columns = list(keep.keys())
    out["time"] = t
    out = out.dropna(subset=["time"]).set_index("time").sort_index()
    for c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(how="all"); out.replace([np.inf,-np.inf], np.nan, inplace=True)
    return out

def download_metrics(market: str, symbol: str, start: str, end: str, granularity: str="daily") -> pd.DataFrame:
    sess = requests.Session(); frames=[]; base = _path(market, granularity, "metrics", symbol, interval=None)
    for ds in _dates(granularity, start, end):
        candidates = [f"{symbol}-{ds}.zip", f"{symbol}-metrics-{ds}.zip"]
        hit=False
        for fname in candidates:
            url = _url(base, fname); cache = os.path.join(CACHE_DIR, f"metrics-{symbol}-{ds}.zip")
            try:
                if not os.path.exists(cache):
                    r = sess.get(url, timeout=30)
                    if r.status_code==404: continue
                    r.raise_for_status(); open(cache,"wb").write(r.content)
                frames.append(read_metrics_zip(open(cache,"rb").read()))
                hit=True; break
            except Exception: continue
        if not hit: warnings.warn(f"[缺失]{_url(base, candidates[-1])}")
    if not frames: return pd.DataFrame(columns=["open_interest"])
    return pd.concat(frames).sort_index().drop_duplicates()

# -------------------- （可选）REST API 拉 OI --------------------
def _market_base_url(market:str)->str:
    if market=="futures-um": return "https://fapi.binance.com"
    if market=="futures-cm": return "https://dapi.binance.com"
    return ""

def download_open_interest_api(market: str, symbol: str, period: str,
                               start: str, end: str, sleep_ms: int=250) -> pd.DataFrame:
    base = _market_base_url(market)
    if not base: return pd.DataFrame(columns=["open_interest"])
    url = f"{base}/futures/data/openInterestHist"
    st = int(pd.Timestamp(start, tz="UTC").timestamp()*1000)
    et = int(pd.Timestamp(end, tz="UTC").timestamp()*1000)
    out=[]; cur=st; sess=requests.Session()
    while True:
        params={"symbol":symbol,"period":period,"limit":500,"startTime":cur,"endTime":et}
        try:
            r = sess.get(url, params=params, timeout=30)
            if r.status_code!=200:
                if r.status_code==400: break
                time.sleep(max(sleep_ms,1000)/1000.0); continue
            data = r.json()
            if not isinstance(data,list) or not data: break
            for row in data:
                ts=int(row.get("timestamp",0))
                oi=float(row.get("sumOpenInterest",np.nan))
                oiv=row.get("sumOpenInterestValue",None)
                oiv=float(oiv) if oiv is not None else np.nan
                if ts>0: out.append((ts,oi,oiv))
            nxt = int(data[-1]["timestamp"])+1
            if nxt>=et: break
            cur=nxt; time.sleep(sleep_ms/1000.0)
        except Exception:
            time.sleep(max(sleep_ms,1000)/1000.0); break
    if not out: return pd.DataFrame(columns=["open_interest"])
    df=pd.DataFrame(out,columns=["ts","open_interest","open_interest_value"])
    df["time"]=pd.to_datetime(df["ts"],unit="ms",utc=True)
    return df.drop(columns=["ts"]).set_index("time").sort_index()

# -------------------- 特征工程（基础） --------------------
def add_base_features(df: pd.DataFrame, suffix: str="") -> pd.DataFrame:
    out = df.copy()
    out[f"last_close{suffix}"] = out["close"].shift(1)
    out[f"last_vol{suffix}"]   = out["volume"].shift(1)
    out[f"logret{suffix}"]     = np.log(out["close"]).diff()
    for w in (5,12,24,48):
        out[f"ma_{w}{suffix}"] = out["close"].rolling(w, min_periods=w).mean()
    out[f"ema_12{suffix}"] = out["close"].ewm(span=12, adjust=False).mean()
    out[f"ema_26{suffix}"] = out["close"].ewm(span=26, adjust=False).mean()
    out[f"macd{suffix}"]   = out[f"ema_12{suffix}"] - out[f"ema_26{suffix}"]
    out[f"macd_sig{suffix}"]  = out[f"macd{suffix}"].ewm(span=9, adjust=False).mean()
    out[f"macd_hist{suffix}"] = out[f"macd{suffix}"] - out[f"macd_sig{suffix}"]
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out[f"atr_14{suffix}"] = tr.rolling(14, min_periods=14).mean()
    ret = out["close"].pct_change()
    for w in (12,24,48):
        out[f"rv_{w}{suffix}"] = ret.rolling(w, min_periods=w).std() * np.sqrt(w)
    for w in (12,24,48):
        out[f"vol_ma_{w}{suffix}"] = out["volume"].rolling(w, min_periods=w).mean()
    out[f"vol_chg{suffix}"] = out["volume"].pct_change()
    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    return out

# -------------------- 构建单币多时间尺度数据（含新增交互/布尔/极端值特征） --------------------
def build_symbol_multitimeframe(args, symbol, start, end) -> pd.DataFrame:
    base = download_klines(args.market, symbol, args.interval, start, end, args.granularity)
    if base.empty:
        warnings.warn(f"[跳过]{symbol} 主K线为空"); return pd.DataFrame()
    base = base[["open","high","low","close","volume","quote_asset_volume","num_trades","taker_buy_base","taker_buy_quote"]].copy()
    base = add_base_features(base, suffix="")

    # 标记/指数/溢价/资金费率
    if args.use_mark:
        mk = download_mark_price(args.market, symbol, args.interval, start, end, args.granularity)
        if not mk.empty: base["mark_close"] = mk["close"].reindex(base.index).interpolate(method="time", limit=6)
    if args.use_index:
        ix = download_index_price(args.market, symbol, args.interval, start, end, args.granularity)
        if not ix.empty: base["index_close"] = ix["close"].reindex(base.index).interpolate(method="time", limit=6)
    if args.use_premium:
        pm = download_premium_price(args.market, symbol, args.interval, start, end, args.granularity)
        if not pm.empty: base["premium_close"] = pm["close"].reindex(base.index).interpolate(method="time", limit=6)
    if args.use_funding:
        fr = download_funding(args.market, symbol, start, end)
        if not fr.empty: base["funding_rate"] = fr["funding_rate"].reindex(base.index, method="ffill")

    # metrics（含 OI、多空比）
    if args.use_metrics:
        mt = download_metrics(args.market, symbol, start, end, granularity="daily")
        if not mt.empty:
            base = pd.merge_asof(base.sort_index(), mt.sort_index(),
                                 left_index=True, right_index=True, direction="backward")
            if "open_interest" in base:
                base["oi_chg"] = base["open_interest"].pct_change()
                base["oi_z"] = (base["open_interest"] - base["open_interest"].rolling(96, min_periods=30).mean()) / \
                               (base["open_interest"].rolling(96, min_periods=30).std() + 1e-12)
            if "open_interest_value" in base:
                base["oi_val_chg"] = base["open_interest_value"].pct_change()
            for c in ["count_toptrader_long_short_ratio","sum_toptrader_long_short_ratio",
                      "count_long_short_ratio","sum_taker_long_short_vol_ratio"]:
                if c in base: base[f"{c}_chg"] = base[c].pct_change()

    # 衍生指标
    if "mark_close" in base:  base["spread_mark_last"] = (base["mark_close"]/(base["last_close"]+EPS)-1.0)
    if "index_close" in base: base["basis_last_index"] = (base["last_close"]/(base["index_close"]+EPS)-1.0)
    if "premium_close" in base:
        base["prem_last"] = (base["premium_close"]/(base["last_close"]+EPS)-1.0)
        base["prem_z"] = (base["prem_last"] - base["prem_last"].rolling(96, min_periods=30).mean()) / \
                         (base["prem_last"].rolling(96, min_periods=30).std() + 1e-12)
    if "taker_buy_base" in base: base["taker_buy_ratio"] = base["taker_buy_base"]/(base["last_vol"]+EPS)
    if "funding_rate" in base:
        base["funding_abs"] = base["funding_rate"].astype(float)
        base["funding_chg"] = base["funding_abs"].diff()

    # ===== 新增：极端值Z、交互项、布尔状态 =====
    def zscore(s, win=96, minp=30):
        mu = s.rolling(win, min_periods=minp).mean()
        sd = s.rolling(win, min_periods=minp).std()
        return (s - mu) / (sd + 1e-12)

    if "funding_abs" in base:
        base["funding_abs_z"] = zscore(base["funding_abs"], win=96, minp=30)
    if "basis_last_index" in base:
        base["basis_z"] = zscore(base["basis_last_index"], win=96, minp=30)
    if "prem_last" in base:
        base["prem_last_z"] = zscore(base["prem_last"], win=96, minp=30)
    if "rv_24" in base and "rv_48" in base:
        base["rv_24_over_48"] = base["rv_24"] / (base["rv_48"] + 1e-12)

    if "basis_last_index" in base and "oi_chg" in base:
        base["x_basis_oi"] = base["basis_last_index"] * base["oi_chg"]
    if "prem_last_z" in base and "oi_z" in base:
        base["x_premz_oiz"] = base["prem_last_z"] * base["oi_z"]
    if "funding_abs_z" in base and "oi_z" in base:
        base["x_fundz_oiz"] = base["funding_abs_z"] * base["oi_z"]
    if "taker_buy_ratio" in base and "rv_12" in base:
        base["x_takerbuy_rv12"] = base["taker_buy_ratio"] * base["rv_12"]
    if "macd_hist" in base and "oi_chg" in base:
        base["x_macd_hist_sign_oi"] = np.sign(base["macd_hist"]) * np.sign(base["oi_chg"].fillna(0))

    if "macd_hist" in base:
        for ex in [s.strip() for s in (args.extra_intervals or "").split(",") if s.strip()]:
            ex_col = f"macd_hist_{ex}"
            if ex_col in base:
                base[f"cons_macd_{ex}"] = (np.sign(base["macd_hist"]) == np.sign(base[ex_col])).astype(float)
    if "logret" in base:
        for ex in [s.strip() for s in (args.extra_intervals or "").split(",") if s.strip()]:
            ex_col = f"logret_{ex}"
            if ex_col in base:
                base[f"cons_ret_{ex}"] = (np.sign(base["logret"]) == np.sign(base[ex_col])).astype(float)

    if "close" in base and "ma_24" in base:
        base["state_above_ma24"] = (base["close"] > base["ma_24"]).astype(float)
    if "ma_12" in base and "ma_24" in base and "ma_48" in base:
        base["state_ma_bull"] = ((base["ma_12"] > base["ma_24"]) & (base["ma_24"] > base["ma_48"])).astype(float)
        base["state_ma_bear"] = ((base["ma_12"] < base["ma_24"]) & (base["ma_24"] < base["ma_48"])).astype(float)
    # ===== 新增结束 =====

    base.replace([np.inf,-np.inf], np.nan, inplace=True)
    base["symbol"] = symbol

    # 副周期
    extra = [s.strip() for s in args.extra_intervals.split(",")] if args.extra_intervals else []
    for ex in extra:
        if not ex: continue
        exdf = download_klines(args.market, symbol, ex, start, end, args.granularity)
        if exdf.empty:
            warnings.warn(f"[副周期缺失] {symbol} {ex}"); continue
        exdf = exdf[["open","high","low","close","volume"]].copy()
        exdf = add_base_features(exdf, suffix=f"_{ex}")
        ex_cols = [c for c in exdf.columns if c.endswith(f"_{ex}")]
        base = pd.merge_asof(base.sort_index(), exdf[ex_cols].sort_index(),
                             left_index=True, right_index=True, direction="backward")
    return base

def build_multisymbol(args, symbols, start, end) -> pd.DataFrame:
    frames=[]
    print("[STEP] 多时间周期数据构建开始")
    for sym in symbols:
        print(f"  - {sym}")
        f = build_symbol_multitimeframe(args, sym, start, end)
        if not f.empty: frames.append(f)
    if not frames: raise RuntimeError("没有可用币种数据")
    df = pd.concat(frames).sort_index()
    print("[STEP] 多时间周期数据构建完成 行数=", len(df))
    return df

# -------------------- 序列化（可返回时间/品种以导出信号） --------------------
def make_sequences_by_symbol(df_all: pd.DataFrame, horizon:int, seq_len:int,
                             feature_cols:List[str], symbols:List[str],
                             return_index: bool=False):
    X_list, y_list, t_list, s_list = [], [], [], []
    for sym in symbols:
        d = df_all[df_all["symbol"]==sym].copy()
        d[feature_cols] = d[feature_cols].replace([np.inf,-np.inf], np.nan)
        y = (d["last_close"].shift(-horizon) > d["last_close"]).astype(int).rename("label")
        data = d[feature_cols].join(y).dropna()
        if len(data) < seq_len + horizon + 1: continue
        X_np = data[feature_cols].values.astype(np.float32)
        y_np = data["label"].values.astype(np.int32)
        idx  = data.index.to_numpy()
        for i in range(len(X_np) - seq_len):
            X_list.append(X_np[i:i+seq_len,:])
            y_list.append(y_np[i+seq_len])
            if return_index:
                t_list.append(idx[i+seq_len])
                s_list.append(sym)
    if not X_list: raise RuntimeError("未生成样本，请调整 seq_len/horizon 或检查缺失。")
    X = np.asarray(X_list, np.float32)
    y = np.asarray(y_list, np.int32)
    if return_index:
        return X, y, np.array(t_list), np.array(s_list)
    return X, y

# -------------------- 特征选择（基础打分） --------------------
def feature_selection_rank(X: np.ndarray, y: np.ndarray, colnames: List[str], topk:int=80) -> List[str]:
    dfX = pd.DataFrame(X, columns=colnames)
    var = dfX.var().replace(0, 1e-12)
    keep_mask = var > np.quantile(var, 0.05)
    dfX = dfX.loc[:, keep_mask]; cols1 = dfX.columns.tolist()
    idx = np.random.choice(len(dfX), size=min(30000, len(dfX)), replace=False)
    mi = mutual_info_classif(dfX.values[idx], y[idx], discrete_features=False, random_state=42)
    mi = pd.Series(mi, index=cols1); mi = (mi - mi.min())/(mi.max()-mi.min()+1e-12)
    idx2 = np.random.choice(len(dfX), size=min(50000, len(dfX)), replace=False)
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(dfX.values[idx2], y[idx2])
    im = pd.Series(rf.feature_importances_, index=cols1); im = (im - im.min())/(im.max()-im.min()+1e-12)
    score = 0.5*mi + 0.5*im
    return score.sort_values(ascending=False).head(topk).index.tolist()

# -------------------- 稳定性特征选择（多窗口投票） --------------------
def stable_topk(Xtr_raw: np.ndarray, ytr: np.ndarray, feature_cols_all: List[str],
                topk:int=80, n_windows:int=3, min_votes:int=2) -> List[str]:
    N = len(Xtr_raw)
    wins = []
    for k in range(n_windows):
        s = int(k * N / n_windows)
        e = N if k==n_windows-1 else int((k+1) * N / n_windows)
        wins.append((s, e))
    from collections import Counter
    counter = Counter()
    for (s, e) in wins:
        Xw = Xtr_raw[s:e]; yw = ytr[s:e]
        F_all = Xw.shape[2]
        Xw_flat = np.nan_to_num(Xw.reshape(-1, F_all), nan=0.0, posinf=1e6, neginf=-1e6)
        yw_rep  = np.repeat(yw, Xw.shape[1])
        top_cols_w = feature_selection_rank(Xw_flat, yw_rep, feature_cols_all, topk=topk)
        counter.update(top_cols_w)
    voted = [c for c, cnt in counter.items() if cnt >= min_votes]
    if not voted:
        voted = [c for c,_ in counter.most_common(topk)]
    if len(voted) > topk:
        F_all = Xtr_raw.shape[2]
        Xtr_flat = np.nan_to_num(Xtr_raw.reshape(-1, F_all), nan=0.0, posinf=1e6, neginf=-1e6)
        ytr_rep  = np.repeat(ytr, Xtr_raw.shape[1])
        dfX = pd.DataFrame(Xtr_flat, columns=feature_cols_all)[voted]
        mi = mutual_info_classif(dfX, ytr_rep, discrete_features=False, random_state=42)
        mi = pd.Series(mi, index=dfX.columns); mi = (mi - mi.min())/(mi.max()-mi.min()+1e-12)
        rf = RandomForestClassifier(n_estimators=150, max_depth=10, n_jobs=-1, random_state=42)
        rf.fit(dfX.values, ytr_rep)
        im = pd.Series(rf.feature_importances_, index=dfX.columns); im = (im - im.min())/(im.max()-im.min()+1e-12)
        score = (0.5*mi + 0.5*im).sort_values(ascending=False)
        voted = score.head(topk).index.tolist()
    return voted

# -------------------- 模型 --------------------
def build_model(model_type:str, input_shape, hidden:int=128, layers:int=3,
                dropout:float=0.3, recurrent_dropout:float=0.1,
                l2_reg:float=1e-5, lr:float=1e-3):
    model = Sequential()
    model.add(Input(shape=input_shape))
    RNN = LSTM if model_type.lower()=="lstm" else GRU
    if layers <= 1:
        model.add(RNN(hidden, dropout=dropout, recurrent_dropout=recurrent_dropout))
    else:
        model.add(RNN(hidden, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))
        for _ in range(layers-2):
            model.add(RNN(hidden, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))
        model.add(RNN(hidden, dropout=dropout, recurrent_dropout=recurrent_dropout))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation="relu", kernel_regularizer=L2(l2_reg)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid", kernel_regularizer=L2(l2_reg)))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

# -------------------- 评估工具：阈值表、覆盖–准确率、PR、导出信号 --------------------
def threshold_table(y_true, y_prob, grid=None):
    if grid is None: grid = np.linspace(0.1, 0.9, 17)
    rows = []
    for thr in grid:
        pred = (y_prob >= thr).astype(int)
        tp = ((pred==1) & (y_true==1)).sum()
        fp = ((pred==1) & (y_true==0)).sum()
        tn = ((pred==0) & (y_true==0)).sum()
        fn = ((pred==0) & (y_true==1)).sum()
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2*prec*rec/(prec+rec+1e-12)
        cov  = (pred==1).mean()
        acc  = (pred==y_true).mean()
        rows.append([thr, prec, rec, f1, cov, acc])
    return pd.DataFrame(rows, columns=["threshold","precision","recall","f1","coverage","accuracy"])

def coverage_accuracy_curve(y_true, y_prob):
    ts = np.linspace(0.5, 0.95, 10)
    rows=[]
    for t in ts:
        mask = (y_prob>=t) | (y_prob<=1-t)
        if mask.any():
            pred = (y_prob[mask]>=t).astype(int)
            acc = (pred==y_true[mask]).mean()
            rows.append([t, mask.mean(), acc])
        else:
            rows.append([t, 0.0, np.nan])
    return pd.DataFrame(rows, columns=["conf_thresh","coverage","accuracy"])

def export_signals_csv(path, timestamps, symbols, y_prob, thr=0.6):
    mask = (y_prob>=thr) | (y_prob<=1-thr)
    if not mask.any():
        print(f"[EXPORT] 无高置信度样本（thr={thr}）"); return None
    df = pd.DataFrame({
        "time": pd.to_datetime(timestamps[mask]),
        "symbol": symbols[mask],
        "prob_up": y_prob[mask],
        "direction": np.where(y_prob[mask]>=thr, 1, 0)  # 1=看涨，0=看跌
    }).sort_values("time")
    df.to_csv(path, index=False)
    print(f"[EXPORT] 高置信度信号已导出: {path} | 条数={len(df)}")
    return df

def pr_metrics(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    return precision, recall, ap

# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="futures-um", choices=["spot","futures-um","futures-cm"])
    ap.add_argument("--symbols", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--extra-intervals", default="", help="副周期，逗号分隔，如 15m,4h")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--granularity", default="daily", choices=["daily","monthly"])

    ap.add_argument("--use-mark", action="store_true")
    ap.add_argument("--use-index", action="store_true")
    ap.add_argument("--use-premium", action="store_true")
    ap.add_argument("--use-funding", action="store_true")
    ap.add_argument("--use-metrics", action="store_true")
    ap.add_argument("--use-oi-api", action="store_true")
    ap.add_argument("--oi-period", type=str, default="1h")
    ap.add_argument("--oi-rate-limit-ms", type=int, default=250)

    ap.add_argument("--model", default="gru", choices=["gru","lstm"])
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--recurrent-dropout", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--topk", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--stable-windows", type=int, default=3)
    ap.add_argument("--stable-votes", type=int, default=2)
    args = ap.parse_args()

    start = args.start or DEFAULT_BEGIN
    end = args.end or datetime.utcnow().strftime("%Y-%m-%d")
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    print(f"== 任务 == 市场={args.market} | 币种={symbols} | 主周期={args.interval} | 副周期={args.extra_intervals} | 时间={start}~{end}")
    print(f"开启: mark={args.use_mark}, index={args.use_index}, premium={args.use_premium}, funding={args.use_funding}, metrics={args.use_metrics}, oi_api={args.use_oi_api}")
    print(f"FS TopK={args.topk} | 模型={args.model} | Seq={args.seq_len} | Horizon={args.horizon} | LR={args.lr}")

    # 1) 数据
    df = build_multisymbol(args, symbols, start, end)

    # 自动把新增/交互/布尔类列名并入候选（存在即加入）
    sym_ohe = pd.get_dummies(df["symbol"], prefix="sym")
    df = pd.concat([df, sym_ohe], axis=1)

    base_cols = [
        "open","high","low","close","last_close","last_vol","quote_asset_volume","num_trades",
        "taker_buy_base","taker_buy_quote","logret",
        "spread_mark_last","basis_last_index","prem_last","prem_z",
        "ma_5","ma_12","ma_24","ma_48",
        "ema_12","ema_26","macd","macd_sig","macd_hist",
        "atr_14","rv_12","rv_24","rv_48",
        "vol_ma_12","vol_ma_24","vol_ma_48","vol_chg","taker_buy_ratio",
        "funding_rate","funding_abs","funding_chg","funding_abs_z",
        "open_interest","open_interest_value","oi_chg","oi_val_chg","oi_z",
        "count_toptrader_long_short_ratio","sum_toptrader_long_short_ratio",
        "count_long_short_ratio","sum_taker_long_short_vol_ratio",
        "count_toptrader_long_short_ratio_chg","sum_toptrader_long_short_ratio_chg",
        "count_long_short_ratio_chg","sum_taker_long_short_vol_ratio_chg",
        "basis_z","prem_last_z","rv_24_over_48",
        "x_basis_oi","x_premz_oiz","x_fundz_oiz","x_takerbuy_rv12","x_macd_hist_sign_oi",
        "state_above_ma24","state_ma_bull","state_ma_bear",
    ]
    extra_cols=[]
    if args.extra_intervals:
        for ex in [s.strip() for s in args.extra_intervals.split(",")]:
            if ex: extra_cols += [c for c in df.columns if c.endswith(f"_{ex}") or c.startswith(f"cons_macd_{ex}") or c.startswith(f"cons_ret_{ex}")]
    auto_new_cols = [c for c in df.columns if c.startswith(("funding_abs_z","basis_z","prem_last_z",
                                                            "rv_24_over_48","x_basis_oi","x_premz_oiz",
                                                            "x_fundz_oiz","x_takerbuy_rv12","x_macd_hist_sign_oi",
                                                            "cons_macd_","cons_ret_","state_above_ma24",
                                                            "state_ma_bull","state_ma_bear"))]
    feature_cols_all = [c for c in base_cols if c in df.columns] + extra_cols + auto_new_cols + list(sym_ohe.columns)
    # 去重保持顺序
    feature_cols_all = list(dict.fromkeys(feature_cols_all))

    # 丢高缺失列（<=60%）
    miss = df[feature_cols_all].isna().mean()
    feature_cols_all = [c for c in feature_cols_all if miss.get(c,0) <= 0.60]
    print("[INFO] 进入 FS 的候选列数：", len(feature_cols_all))

    # 2) 构造序列（带索引/符号用于导出信号）
    X_all, y_all, t_all, s_all = make_sequences_by_symbol(df, args.horizon, args.seq_len, feature_cols_all, symbols, return_index=True)
    N=len(X_all); n_tr=int(N*0.7); n_va=int(N*0.15)
    Xtr_raw,ytr = X_all[:n_tr], y_all[:n_tr]
    Xva_raw,yva = X_all[n_tr:n_tr+n_va], y_all[n_tr:n_tr+n_va]
    Xte_raw,yte = X_all[n_tr+n_va:], y_all[n_tr+n_va:]
    t_tr, s_tr = t_all[:n_tr], s_all[:n_tr]
    t_va, s_va = t_all[n_tr:n_tr+n_va], s_all[n_tr:n_tr+n_va]
    t_te, s_te = t_all[n_tr+n_va:],   s_all[n_tr+n_va:]
    print(f"[STEP] 初始序列: train={len(Xtr_raw)} val={len(Xva_raw)} test={len(Xte_raw)} | F={Xtr_raw.shape[2]}")

    # 3) 稳定性特征选择（仅训练集）
    top_cols = stable_topk(Xtr_raw, ytr, feature_cols_all, topk=args.topk,
                           n_windows=args.stable_windows, min_votes=args.stable_votes)
    print(f"[FS] 稳定 Top{args.topk}：{len(top_cols)}")
    print("[FS] Top样例：", top_cols[:20])

    sel_idx = [feature_cols_all.index(c) for c in top_cols]
    Xtr_raw = Xtr_raw[:,:,sel_idx]; Xva_raw = Xva_raw[:,:,sel_idx]; Xte_raw = Xte_raw[:,:,sel_idx]
    F_final = len(sel_idx)

    # 4) 标准化（只在 train 拟合）
    def sanitize(X): return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    Xtr_raw, Xva_raw, Xte_raw = sanitize(Xtr_raw), sanitize(Xva_raw), sanitize(Xte_raw)
    scaler = StandardScaler().fit(Xtr_raw.reshape(-1, F_final))
    def tfm(X):
        S,T,F_ = X.shape
        return scaler.transform(X.reshape(-1, F_)).reshape(S,T,F_)
    Xtr, Xva, Xte = tfm(Xtr_raw), tfm(Xva_raw), tfm(Xte_raw)
    print(f"[STEP] 裁剪后: train={len(Xtr)} val={len(Xva)} test={len(Xte)} | F={F_final}")

    # 5) 模型
    model = build_model(args.model, input_shape=(args.seq_len, F_final),
                        hidden=args.hidden, layers=args.layers,
                        dropout=args.dropout, recurrent_dropout=args.recurrent_dropout,
                        l2_reg=args.l2, lr=args.lr)
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, verbose=1)
    classes = np.array([0,1]); w = compute_class_weight("balanced", classes=classes, y=ytr)
    class_weight = {0:w[0], 1:w[1]}
    print(f"[INFO] class_weight={class_weight}")

    print("[STEP] 训练开始")
    model.fit(Xtr, ytr, validation_data=(Xva, yva),
              epochs=args.epochs, batch_size=args.batch,
              callbacks=[es, rlrop], class_weight=class_weight, verbose=1)

    # 6) 评估与阈值调优
    p_val = model.predict(Xva, verbose=0).ravel()
    p_test = model.predict(Xte, verbose=0).ravel()
    auc_val_raw = roc_auc_score(yva, p_val) if len(np.unique(yva))>1 else np.nan
    flipped=False
    if not np.isnan(auc_val_raw) and auc_val_raw < 0.5:
        flipped=True; p_val=1-p_val; p_test=1-p_test
        print(f"[TIP] VAL AUC={auc_val_raw:.4f}<0.5 已翻转概率")

    ths = np.linspace(0.3,0.7,81)
    best_t_acc = max(ths, key=lambda t: accuracy_score(yva,(p_val>=t).astype(int)))
    best_t_f1  = max(ths, key=lambda t: f1_score(yva,(p_val>=t).astype(int)))
    print(f"[VAL] best_acc_thresh={best_t_acc:.3f}, best_f1_thresh={best_t_f1:.3f}, (AUC_raw={auc_val_raw:.4f}, flipped={flipped})")

    y_pred = (p_test>=best_t_acc).astype(int)
    acc = accuracy_score(yte, y_pred); f1  = f1_score(yte, y_pred)
    auc = roc_auc_score(yte, p_test) if len(np.unique(yte))>1 else np.nan
    cm  = confusion_matrix(yte, y_pred)
    maj = 1 if (ytr.mean()>=0.5) else 0; baseline = (yte==maj).mean()
    print(f"[TEST] Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  (baseline={baseline:.4f}, thr={best_t_acc:.3f})")
    print("Confusion Matrix:\n", cm)

    # 阈值表（保存到 reports/）
    print("\n[THRESHOLD TABLE] (VAL)")
    tbl_val = threshold_table(yva, p_val, grid=np.linspace(0.2, 0.8, 25))
    print(tbl_val.head(10).to_string(index=False))
    print("\n[THRESHOLD TABLE] (TEST)")
    tbl_test = threshold_table(yte, p_test, grid=np.linspace(0.2, 0.8, 25))
    print(tbl_test.head(10).to_string(index=False))
    tbl_val.to_csv("reports/threshold_table_val.csv", index=False)
    tbl_test.to_csv("reports/threshold_table_test.csv", index=False)
    print("[SAVE] 阈值表已保存到 reports/")

    # 覆盖率–准确率（测试集）
    cov_curve = coverage_accuracy_curve(yte, p_test)
    print("\n[COVERAGE–ACCURACY] (TEST)")
    print(cov_curve.to_string(index=False))
    cov_curve.to_csv("reports/coverage_accuracy_test.csv", index=False)

    # PR 曲线 / AP
    prec, rec, ap = pr_metrics(yte, p_test)
    print(f"\n[PR] Average Precision (TEST) = {ap:.4f}")
    pd.DataFrame({"precision":prec, "recall":rec}).to_csv("reports/pr_curve_test.csv", index=False)

    # 导出高置信度信号
    export_signals_csv("reports/high_conf_signals.csv", t_te, s_te, p_test, thr=float(args.conf))

if __name__ == "__main__":
    main()
