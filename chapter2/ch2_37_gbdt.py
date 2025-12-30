# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from sklearn.ensemble import GradientBoostingClassifier
from utils import data_utils
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)

# GradientBoostingClassifier参数解释：
# n_estimators: 弱学习器的数量，即 boosting 迭代次数，值越大模型越复杂
# learning_rate: 学习率，控制每个弱学习器对最终结果的贡献程度，较小的值需要更多的弱学习器
# subsample: 子样本比例，用于训练每个弱学习器的样本比例，小于1.0时可防止过拟合
# max_depth: 树的最大深度，控制单个弱学习器的复杂度
# min_samples_leaf: 叶节点最小样本数，防止过拟合
# random_state: 随机种子，确保结果可重现
clf = GradientBoostingClassifier(n_estimators=100,
                                 learning_rate=0.1,
                                 subsample=0.9,
                                 max_depth=4,
                                 min_samples_leaf=20,
                                 random_state=88)
clf.fit(train_x, train_y)
auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
print("GBDT模型 AUC: ", auc_score)
