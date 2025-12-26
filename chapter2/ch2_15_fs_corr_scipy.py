# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from scipy.stats import pearsonr


# 导入数值型样例数据
# p值能够反映两个特征的显著水平, p值越小，则两个特征越显著。
# 适合两个数值型特征之间的相关性

all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
x1, x2 = x.loc[:, 'age.in.years'], x.loc[:, 'credit.history',]
r, p_value = pearsonr(x1, x2)
print("scipy库计算 特征'age.in.years'和'credit.history'的pearson相关系数 \n", 
    "pearson相关系数: %s, \n" % r, "p_value: %s" % p_value)
