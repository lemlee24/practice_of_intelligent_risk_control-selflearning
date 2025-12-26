# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 导入数值型样例数据
# 方差膨胀系数是一种衡量共线性程度的常用指标，它的计算公式为：vif = 1/(1-R^2)，它表示回归系数估计量的方差与假设特征间不线性相关时方差的比值。
# VIF越大，该特征与其他特征的关系越复杂，多重共线性越严重。若VIF大于10，则认为该特征与其他特征存在多重共线性。
all_x_y = data_utils.get_all_x_y()
# 使用drop函数删除指定的标签列，axis=1表示按列删除，这里删除的是目标变量列，保留所有特征列用于VIF计算
x = all_x_y.drop(data_utils.label, axis=1)
# 计算每个特征的方差膨胀系数(VIF)
# variance_inflation_factor参数说明：
# - exog: 自变量矩阵（特征矩阵）
# - obs: 要计算VIF的特征索引
# 返回值：指定特征的方差膨胀系数，数值越大表示多重共线性越严重
vif = [variance_inflation_factor(x.values, ix) for ix in range(x.shape[1])]
print("各特征的vif值计算结果: \n", dict(zip(x.columns, vif)))

# 筛选阈值小于10的特征
# tolist()的作用是将pandas的索引或序列对象转换为Python的原生列表类型
# 这里通过列表推导式[f < 10 for f in vif]创建布尔索引，筛选出VIF值小于10的特征列
# 然后使用iloc选择这些列，再通过columns获取列名索引，最后用tolist()转换为列表
selected_cols = x.columns[[f < 10 for f in vif]].tolist()
print("设置vif阈值为10, 筛选得到%s个特征: \n" % len(selected_cols), selected_cols)
