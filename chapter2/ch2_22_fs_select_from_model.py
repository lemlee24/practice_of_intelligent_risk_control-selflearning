# -*- coding: utf-8 -*- 

import sys
import toad
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

# 树模型的建立就是一个特征选择的过程。基于树模型的特征选择会根据信息增益或基尼不纯度的准则来选择特征进行建模。输出各个特征的重要度，依次进行特征筛选
# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
y = all_x_y.pop(data_utils.label)
x = all_x_y
# GBDT（Gradient Boosting Decision Tree，梯度提升决策树）是一种基于决策树的集成学习算法
# 它通过迭代的方式构建多个弱学习器（决策树），每个新树都用来纠正前面树的预测错误
# 在特征选择中，GBDT可以评估每个特征在所有树中的重要性得分，从而进行特征筛选
# 通过计算特征在分裂节点时的信息增益或基尼重要性，GBDT能够给出特征的重要度排序

sf = SelectFromModel(GradientBoostingClassifier(random_state=42, n_estimators=100))
x_new = sf.fit_transform(x, y)

selected_cols = x.columns[sf.get_support()].tolist()
print("基于树模型筛选得到%s个特征: \n" % len(selected_cols), selected_cols)
