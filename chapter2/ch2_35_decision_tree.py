# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")


from sklearn.tree import DecisionTreeClassifier
from utils import data_utils
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)
# 导入数值型样例数据
# DecisionTreeClassifier 是 scikit-learn 库中的决策树分类器
# 参数说明：
# criterion='gini': 表示使用基尼不纯度作为分割标准，另一种选择是 'entropy'（信息熵）
# max_depth=8: 限制决策树的最大深度为8层，防止过拟合
# min_samples_leaf=15: 每个叶子节点至少包含15个样本，确保叶节点有足够的样本数
# random_state=88: 随机种子，确保结果可重现
clf = DecisionTreeClassifier(criterion='gini',
                             max_depth=8,
                             min_samples_leaf=15,
                             random_state=88)
clf.fit(train_x, train_y)

# 计算ROC AUC分数，用于评估二分类模型的性能
# roc_auc_score函数需要两个参数：
# 1. test_y: 真实的标签值（测试集的y值）
# 2. clf.predict_proba(test_x)[:, 1]: 模型预测的正类概率
#    - predict_proba返回每个样本属于各类别的概率
#    - [:, 1]表示取正类（类别1）的概率值
# AUC值越接近1，说明模型性能越好，0.5表示随机猜测的水平
auc_score = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
print("决策树模型 AUC: ", auc_score)
# print(clf.predict_proba(test_x)[:, 1])