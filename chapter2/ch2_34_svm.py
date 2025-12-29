# -*- coding: utf-8 -*- 

import sys
import numpy as np
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils import data_utils
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
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

# 线性SVM, Linear Support Vector Classification
line_svm = LinearSVC(penalty='l2',
                     loss='hinge',
                     C=0.2,
                     max_iter=10000,  # 增加最大迭代次数避免收敛警告
                     tol=0.001)
clf = make_pipeline(StandardScaler(), line_svm)
clf.fit(train_x_woe, train_y)
acc_score = accuracy_score(test_y, clf.predict(test_x_woe))
print("="*50)
print("线性SVM模型")
print(f"ACC: {acc_score:.4f}")
print("="*50)


# 支持核函数的SVM, C-Support Vector Classification
svm = SVC(C=0.2,
          kernel='rbf',
          tol=0.001,
          max_iter=10000,  # 增加最大迭代次数
          probability=True)
clf = make_pipeline(StandardScaler(), svm)
clf.fit(train_x_woe, train_y)
y_pred_proba = clf.predict_proba(test_x_woe)[:, 1]
auc_score = roc_auc_score(test_y, y_pred_proba)
print("="*50)
print("支持核函数SVM模型")
print(f"AUC: {auc_score:.4f}")
print("="*50)
