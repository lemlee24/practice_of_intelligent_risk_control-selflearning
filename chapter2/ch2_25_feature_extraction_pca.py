# -*- coding: utf-8 -*- 

import sys
import toad
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.decomposition import PCA


# 导入数值型样例数据
# PCA将高维的特征向量合并为低维的特征向量，是一种无监督的特征提取方法。
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
# PCA（主成分分析）是一种降维技术，通过线性变换将原始高维特征转换为低维特征
# n_components=0.9 表示保留90%的方差信息，即保留能够解释90%数据变异的主成分
# 这样可以在减少特征维度的同时尽可能保留原始数据的重要信息
pca = PCA(n_components=0.9)
x_new = pca.fit_transform(x)
print(f"PCA降维前特征数量: {x.shape[1]}")
print(f"PCA降维后特征数量: {x_new.shape[1]}")
print(f"解释的方差比例: {pca.explained_variance_ratio_.sum():.4f}")
x_new_df = pd.DataFrame(x_new)
print("利用sklearn进行PCA特征提取, 保留90%信息后结果: \n", x_new_df)
