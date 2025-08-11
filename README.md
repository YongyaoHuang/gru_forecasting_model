任务与输出
任务：方向二分类。判断主周期 seq_len 根 K 线之后 horizon 根（如 4 根）的收盘价相对当前是否上涨(1)/下跌(0)。

输出：每个样本给出 上涨概率 p∈[0,1]。

评估时：自动在验证集调阈值（分别找 Acc 与 F1 的最佳阈值），默认用 best_acc 阈值在测试集上汇报。

还会给出 AUC、混淆矩阵、以及高置信度（|p-0.5|≥(conf)）覆盖率与准确率。

数据范围与时间粒度
市场：futures-um（USDT 本位永续）

币种：可多币种，如 --symbols BTCUSDT,ETHUSDT

主周期：--interval 1h（示例）

副周期：--extra-intervals 15m,4h（可多种，merge_asof 退后对齐）

时间范围：--start 到 --end（未给 --end 则到今天）

下载粒度目录：--granularity daily（部分源也会月度回退）

具体用到的数据源
（都来自 data.binance.vision）

klines/：主数据（成交形成的K线，open/high/low/close/volume 等）

markPriceKlines/（可选 --use-mark）：标记价格 K 线

indexPriceKlines/（可选 --use-index）：指数价格 K 线

premiumIndexKlines/（可选 --use-premium）：溢价率相关 K 线

fundingRate/（可选 --use-funding，月度目录）：资金费率

metrics/（可选 --use-metrics，日度）：包括 Open Interest / Open Interest Value / 多空比 / taker 多空成交量比 等

（可选）REST API：/futures/data/openInterestHist 作为 OI 的补充（--use-oi-api）

输入特征（主要类别）
先用全量候选列构建序列 → 仅在训练集上做特征选择（方差过滤 + 互信息 + 随机森林）→ 取 TopK（默认 80）→ 用同一列投到 val/test

价格/量/动量

open, high, low, close, last_close, last_vol, quote_asset_volume, num_trades, taker_buy_base, taker_buy_quote

logret、ma_5/12/24/48、ema_12/26、macd/macd_sig/macd_hist、atr_14

波动率：rv_12/24/48

成交量：vol_ma_12/24/48, vol_chg、taker_buy_ratio

跨价格体系偏离/基差/溢价

与标记价、指数价的偏离：spread_mark_last, basis_last_index

溢价相关：prem_last, prem_z

资金费率

funding_rate, funding_abs, funding_chg

持仓与多空情绪（来自 metrics）

open_interest, open_interest_value, oi_chg, oi_val_chg, oi_z

count_toptrader_long_short_ratio / sum_toptrader_long_short_ratio

count_long_short_ratio / sum_taker_long_short_vol_ratio

及上述的 *_chg 变化率特征

多时间尺度衍生

对副周期（如 15m、4h）重复一套基础特征，并以后缀区分：*_15m, *_4h 等

币种 one-hot

sym_BTCUSDT, sym_ETHUSDT, ...

注：特征中缺失率 >60% 的列会被剔除；标准化在训练集拟合，val/test 只 transform。

模型与训练细节
模型：GRU（或切换 LSTM），结构：

RNN × layers（默认 3 层）→ Dropout → Dense(64, relu) → Dropout → Dense(1, sigmoid)

序列长度：--seq-len（如 128；即输入 128 根主周期 K 线的窗口）

预测步长：--horizon（如 4；1h 主周期即 4 小时后方向）

优化器/学习率：Adam，--lr（如 0.001~0.002）

正负样本不平衡：class_weight 自动平衡

回调：EarlyStopping(patience=8)、ReduceLROnPlateau(patience=4, factor=0.5)

评估：Acc / F1 / AUC / 混淆矩阵；验证集若 AUC<0.5 自动“翻转概率”再评估（防止方向整体相反）
