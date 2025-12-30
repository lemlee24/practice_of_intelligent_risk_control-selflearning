# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from sklearn.ensemble import RandomForestClassifier
from utils import data_utils
from sklearn.metrics import roc_auc_score

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)
# RandomForestClassifier 随机森林分类器参数说明：
# n_estimators=200: 决策树的数量，越多通常效果越好，但计算时间也会增加
# criterion='gini': 分割标准，'gini'表示使用基尼不纯度，'entropy'表示使用信息熵
# max_depth=6: 树的最大深度，用于控制过拟合
# min_samples_leaf=15: 叶节点最少样本数，防止过拟合
# bootstrap=True: 是否使用自助采样法构建树
# oob_score=True: 是否使用袋外样本来评估模型泛化精度
# random_state=88: 随机种子，保证结果可重现
clf = RandomForestClassifier(n_estimators=200,
                             criterion='gini',
                             max_depth=6,
                             min_samples_leaf=15,
                             bootstrap=True,
                             oob_score=True,
                             random_state=88)
clf.fit(train_x, train_y)
auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
print("随机森林模型 AUC: ", auc_score)
