# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from scipy.stats import variation

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
# 计算各个特征的变异系数,也叫离散系数，是概率分布离散程度的一个归一化量度，其定义为标准差与均值之比。
x_var = variation(x, nan_policy='omit')
# 参数nan_policy指定缺失值处理方法，该参数'omit'时，忽略缺失值，设为'raise'时，抛出异常，设为'propagate'时，返回NaN.
result = dict(zip(x.columns ,x_var))
print("变异系数结果: \n", result)