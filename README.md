方向二分类预测模型
📌 任务说明
本项目实现了加密货币永续合约方向预测（二分类）：

目标：在主周期 seq_len 根 K 线之后，判断未来 horizon 根（例如 4 根）K 线的收盘价相对当前是否 上涨 (1) / 下跌 (0)。

输出：每个样本的上涨概率 p ∈ [0,1]。

评估时：

在验证集自动调节两个最优阈值：

best_acc（最高准确率）

best_f1（最高 F1 分数）

默认在测试集使用 best_acc 阈值汇报结果。

输出指标：

AUC

混淆矩阵

高置信度样本（|p - 0.5| ≥ conf）的覆盖率与准确率

📊 数据范围与时间粒度
参数	示例	说明
市场	futures-um	USDT 本位永续
币种	--symbols BTCUSDT,ETHUSDT	可多币种
主周期	--interval 1h	训练主时间尺度
副周期	--extra-intervals 15m,4h	可多种，merge_asof 退后对齐
时间范围	--start 2023-01-01 --end 2024-01-01	不指定 --end 则到今天
下载粒度	--granularity daily	部分源会按月度回退

📂 数据源（来自 data.binance.vision）
主数据
klines/ — 主 K 线数据（open/high/low/close/volume 等）

可选数据（按参数开启）
数据源	参数示例	说明
markPriceKlines/	--use-mark	标记价格 K 线
indexPriceKlines/	--use-index	指数价格 K 线
premiumIndexKlines/	--use-premium	溢价率
fundingRate/	--use-funding	资金费率（月度目录）
metrics/	--use-metrics	OI、多空比、成交量比等
REST API /futures/data/openInterestHist	--use-oi-api	OI 补充数据

📐 输入特征
1. 价格 / 量 / 动量
swift
复制
编辑
open, high, low, close, last_close, last_vol,
quote_asset_volume, num_trades, taker_buy_base, taker_buy_quote
logret, ma_5/12/24/48, ema_12/26, macd, macd_sig, macd_hist, atr_14
rv_12/24/48, vol_ma_12/24/48, vol_chg, taker_buy_ratio
2. 跨价格体系偏离
复制
编辑
spread_mark_last, basis_last_index, prem_last, prem_z
3. 资金费率
复制
编辑
funding_rate, funding_abs, funding_chg
4. 持仓与多空情绪（metrics）
复制
编辑
open_interest, open_interest_value, oi_chg, oi_val_chg, oi_z
count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio
count_long_short_ratio, sum_taker_long_short_vol_ratio
及上述 *_chg 变化率
5. 多时间尺度衍生
对每个副周期（如 15m, 4h）重复一套基础特征，并添加后缀：

markdown
复制
编辑
*_15m, *_4h
6. 币种 One-Hot
复制
编辑
sym_BTCUSDT, sym_ETHUSDT, ...
特征筛选流程：

全量候选列构建序列

仅在训练集做特征选择（方差过滤 + 互信息 + 随机森林）

取 TopK（默认 80）

同列投到验证/测试集

缺失率 > 60% 的列剔除

标准化在训练集拟合，验证/测试仅 transform

🧠 模型与训练
模型结构
scss
复制
编辑
GRU × layers（默认 3 层）
→ Dropout
→ Dense(64, relu)
→ Dropout
→ Dense(1, sigmoid)
可切换为 LSTM

核心参数
参数	示例值	说明
--seq-len	128	输入序列长度（主周期 K 线数）
--horizon	4	预测步长（未来 K 线数）
--lr	0.001	学习率
优化器	Adam	—
样本不平衡处理	class_weight 自动平衡	—

回调机制
EarlyStopping(patience=8)

ReduceLROnPlateau(patience=4, factor=0.5)

📈 评估指标
Acc / F1 / AUC

混淆矩阵

高置信度样本（|p - 0.5| ≥ conf）的覆盖率与准确率

验证集若 AUC < 0.5，自动“翻转概率”再评估（防止方向整体相反）

🚀 示例运行命令
bash
复制
编辑
python train.py \
  --symbols BTCUSDT,ETHUSDT \
  --interval 1h \
  --extra-intervals 15m,4h \
  --start 2023-01-01 \
  --seq-len 128 \
  --horizon 4 \
  --use-funding \
  --use-metrics \
  --topk 80 \
  --lr 0.001 \
  --epochs 100
