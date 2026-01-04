# -*- coding: utf-8 -*- 

import sys
sys.path.append("../")

import lightgbm as lgb
from utils import data_utils
from sklearn.metrics import roc_auc_score

# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)

# 创建LightGBM分类器
# 新版本LightGBM的API变化：early_stopping_rounds需要通过callbacks传递
# 优化参数以减少警告：
# - force_row_wise=True: 使用行式多线程，减少overhead
# - max_depth: 增加树深度以允许更多分裂
# - min_child_samples: 设置叶子节点最小样本数
# - min_split_gain: 设置分裂所需的最小增益
# - verbose: 设置为-1以减少输出
clf = lgb.LGBMClassifier(objective='binary',
                         boosting_type='gbdt',
                         max_depth=6,
                         n_estimators=1000,
                         subsample=1,
                         colsample_bytree=1,
                         min_child_samples=20,
                         min_split_gain=0.0,
                         force_row_wise=True,
                         verbose=-1)

# 使用callbacks参数传递early_stopping
lgb_model = clf.fit(train_x, train_y,
                    eval_set=[(test_x, test_y)],
                    eval_metric='auc',
                    callbacks=[lgb.early_stopping(stopping_rounds=30)])

auc_score = roc_auc_score(test_y, lgb_model.predict_proba(test_x)[:, 1])
print("LightGBM模型 AUC: ", auc_score)
sys.path.append("./")
