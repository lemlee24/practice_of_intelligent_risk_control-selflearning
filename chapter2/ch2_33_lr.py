# -*- coding: utf-8 -*- 

import sys
import numpy as np
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import toad

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)
print(f"原始训练集大小: {train_x.shape}, 测试集大小: {test_x.shape}")

# 使用toad进行WOE编码
print("开始WOE编码...")
# 合并训练数据用于分箱
train_data = train_x.copy()
train_data['target'] = train_y.values

# 使用toad进行分箱
combiner = toad.transform.Combiner()
combiner.fit(train_data, y='target', method='chi', min_samples=0.05)

# 应用分箱
train_x_bins = combiner.transform(train_x)
test_x_bins = combiner.transform(test_x)

# WOE转换
transer = toad.transform.WOETransformer()
train_x_woe = transer.fit_transform(train_x_bins, train_y)
test_x_woe = transer.transform(test_x_bins)

print(f"WOE编码完成，特征数: {train_x_woe.shape[1]}")

# 利用梯度下降法训练逻辑回归模型
# 梯度下降法是一种优化算法，通过迭代更新模型参数来最小化损失函数
# 在逻辑回归中，梯度下降法通过计算损失函数对模型参数的偏导数（梯度）
# 然后沿着梯度的反方向更新参数，逐步逼近最优解
# SGDClassifier使用随机梯度下降，每次迭代只使用一个样本或小批量样本来计算梯度
# 这样可以加快训练速度并减少内存使用
lr = SGDClassifier(loss="log_loss",  # 新版本使用'log_loss'而不是'log'
                   penalty="l2",
                   learning_rate='optimal',
                   max_iter=1000,  # 增加迭代次数以避免收敛警告
                   tol=0.001,
                   epsilon=0.1,
                   random_state=1)
clf = make_pipeline(StandardScaler(), lr)
clf.fit(train_x_woe, train_y)
y_pred_proba = clf.predict_proba(test_x_woe)[:, 1]
auc_score = roc_auc_score(test_y, y_pred_proba)
print("="*50)
print("梯度下降法训练逻辑回归模型")
print(f"AUC: {auc_score:.4f}")
print("="*50)

# 利用牛顿法训练逻辑回归模型
# 牛顿法是一种二阶优化算法，通过使用损失函数的二阶导数（海塞矩阵）来加速收敛
# 与梯度下降法相比，牛顿法具有更快的收敛速度，特别是接近最优解时
# 在逻辑回归中，牛顿法通过迭代更新参数θ，更新公式为：θ = θ - H^(-1) * g
# 其中H是海塞矩阵（二阶导数矩阵），g是梯度向量（一阶导数）
# 牛顿法的优点是收敛速度快（二次收敛），但计算海塞矩阵及其逆矩阵的计算成本较高
# 在sklearn中，solver='lbfgs'使用了拟牛顿法（L-BFGS），它是牛顿法的改进版本
# L-BFGS通过近似海塞矩阵来减少内存使用，特别适合大规模数据集
lr = LogisticRegression(C=1.0,  # 正则化强度的倒数，C=1.0表示标准的l2正则化
                        solver='lbfgs',
                        max_iter=1000,  # 增加迭代次数
                        tol=0.001,
                        random_state=1)
clf = make_pipeline(StandardScaler(), lr)
clf.fit(train_x_woe, train_y)
y_pred_proba = clf.predict_proba(test_x_woe)[:, 1]
auc_score = roc_auc_score(test_y, y_pred_proba)
print("="*50)
print("牛顿法训练逻辑回归模型")
print(f"AUC: {auc_score:.4f}")
print("="*50)
