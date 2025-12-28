# -*- coding: utf-8 -*- 

import sys
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.manifold import LocallyLinearEmbedding

# LLE 是一种基于“流形学习”的方法，其能够使特征提取后的数据较好地保持原有流形结构。LLE假设数据在较小的局部是线性的，即一个数据点都可以由其临点线性表示。
# 导入数值型样例数据

all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
# LocallyLinearEmbedding (LLE) 是一种非线性降维方法，属于流形学习算法
# 主要参数说明：
# n_neighbors: 邻居数量，用于确定每个样本的局部邻域，这里设置为5
# n_components: 降维后的维度数，这里设置为10，即将原始特征降至10维
# LLE算法假设数据在局部是线性的，通过保持局部邻域关系来实现降维
lle = LocallyLinearEmbedding(n_neighbors=5, n_components=10)
x_new = lle.fit_transform(x)
x_new_df = pd.DataFrame(x_new)
print("利用sklearn进行LLE特征提取结果: \n", x_new_df)
