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
