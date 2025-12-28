# -*- coding: utf-8 -*- 

import sys
import toad
sys.path.append("./")
sys.path.append("../")

from utils import data_utils

# 加载数据
all_x_y = data_utils.get_all_x_y()
# 定义分箱方法


"""
toad.transform.Combiner 是 toad 库中的一个重要的数据预处理工具，主要用于特征分箱（Binning）操作。
分箱是风控建模中的关键步骤，它能够：
1. 将连续变量转换为离散的区间，提高模型稳定性
2. 处理异常值和噪声数据
3. 使特征分布更加均匀，便于后续建模
4. 在计算 PSI 等指标时提供统一的分箱标准

Combiner 的主要参数说明：
- n_bins: 分箱数量，默认为 10
- method: 分箱方法，可选 'quantile'(等频)、'step'(等距)、'chi'(卡方分箱) 等
- empty_separate: 是否将空值单独分为一箱，默认为 False
"""
Combiner = toad.transform.Combiner()
Combiner.fit(all_x_y,
             y=data_utils.label,
             n_bins=6,
             method='quantile',
             empty_separate=True)
# 计算psi

"""
PSI (Population Stability Index) 用于衡量特征分布的稳定性，计算公式为：
PSI = Σ[(实际分布比例 - 期望分布比例) * ln(实际分布比例 / 期望分布比例)]
其中：
- 实际分布比例：样本在各分箱中的占比
- 期望分布比例：基准样本在各分箱中的占比
- ln：自然对数函数
PSI值越小，表示特征分布越稳定，通常阈值设置为0.1，超过此值认为特征分布发生显著变化
"""

"""
toad.metrics.PSI 是 toad 库中用于计算 PSI (Population Stability Index) 指标的函数
PSI 用于评估特征在不同时间窗口或样本集之间的分布稳定性
函数参数说明：
- all_x_y.iloc[:500, :]：基准样本数据（前500行），作为期望分布的参考
- all_x_y.iloc[500:, :]：比较样本数据（后半部分），用于与基准样本进行比较
- combiner：使用预先训练好的 Combiner 对象进行统一分箱处理
PSI 计算逻辑：
1. 首先使用 combiner 对两个数据集进行相同的分箱处理
2. 计算每个分箱在基准样本和比较样本中的分布比例
3. 根据 PSI 公式计算差异指数：Σ[(比较样本比例 - 基准样本比例) * ln(比较样本比例 / 基准样本比例)]
4. 返回每个特征的 PSI 值，用于评估特征的稳定性
"""
var_psi = toad.metrics.PSI(all_x_y.iloc[:500, :],
                           all_x_y.iloc[500:, :],
                           combiner=Combiner)
var_psi_df = var_psi.to_frame(name='psi')

selected_cols = var_psi[var_psi_df.psi < 0.1].index.tolist()
print("各特征的psi值计算结果: \n", var_psi_df)
print("设置psi阈值为0.1, 筛选得到%s个特征: \n" % len(selected_cols), selected_cols)
