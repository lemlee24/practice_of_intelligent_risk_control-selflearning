# -*- coding: utf-8 -*- 

import sys
import toad
sys.path.append("./")
sys.path.append("../")

from utils import data_utils

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
# 逐步回归法进行筛选并踢出引起多重共线性变量的方法
"""
toad.selection.stepwise 逐步回归法参数说明：
all_x_y: 输入的数据集，包含特征和目标变量
target: 目标变量的列名，这里使用data_utils.label指定目标列
estimator: 估计器类型，'lr'表示逻辑回归
direction: 逐步回归方向，'both'表示双向逐步回归（既可加入也可剔除变量）
criterion: 选择变量的评价标准，'aic'表示使用赤池信息准则
return_drop: 是否返回被剔除的变量，默认False只返回最终保留的变量
"""
final_data = toad.selection.stepwise(all_x_y,
                                     target=data_utils.label,
                                     estimator='lr',
                                     direction='both',
                                     criterion='aic',
                                     return_drop=False)
selected_cols = final_data.columns
print("通过stepwise筛选得到%s个特征: \n" % len(selected_cols), selected_cols)
