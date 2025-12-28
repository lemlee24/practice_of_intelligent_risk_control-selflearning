# -*- coding: utf-8 -*- 

import sys
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# LDA是一种基于分类模型进行特征属性合并的操作，是一种有监督的特征提取方法。原理是将带标签的数据投影到维度更低的空间中，使得投影后的点按类别区分，相同类别的点会在投影后的空间中更接近。
# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
y = all_x_y[data_utils.label]
# LinearDiscriminantAnalysis(线性判别分析)是一种经典的有监督降维算法
# 主要参数说明：
# n_components: 指定降维后的维度数量，这里设置为1，即将数据降到1维
# solver: 求解方法，可选'lsqr'、'eigen'等
# shrinkage: 正则化参数，用于处理小样本问题
# tol: 数值计算容差
# store_covariance: 是否存储协方差矩阵
# priors: 指定各类别的先验概率

# 计算最大可降维度：min(特征数, 类别数-1)
n_classes = len(y.unique())
n_features = x.shape[1]
max_components = min(n_features, n_classes - 1)

print(f"数据集信息：")
print(f"  样本数: {x.shape[0]}")
print(f"  特征数: {n_features}")
print(f"  类别数: {n_classes}")
print(f"  最大降维维度: {max_components}\n")

# 对于二分类问题，最多只能降到1维
lda = LinearDiscriminantAnalysis(n_components=max_components if max_components > 0 else None)
x_new = lda.fit_transform(x, y)
x_new_df = pd.DataFrame(x_new, columns=[f'LDA_Component_{i+1}' for i in range(x_new.shape[1])])
print(f"LDA降维后的维度: {x_new.shape}")
print("\n利用sklearn进行LDA特征提取结果(前10行): \n", x_new_df.head(10))
print("\n降维后数据统计信息:")
print(x_new_df.describe())
