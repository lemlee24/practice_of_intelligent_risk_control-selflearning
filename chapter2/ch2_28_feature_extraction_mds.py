# -*- coding: utf-8 -*- 

import sys
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.manifold import MDS
'''
MDS是将高维空间中的样本点投影到低维空间中，让样本彼此之间的距离尽可能不变。MDS的具体做法是，
首先计算得到高维空间中样本之间的距离矩阵，接着计算得到低维空间的内积矩阵，然后对低维空间的内积矩阵进行特征值分解，
并按照从大到小的顺序取前d个特征值和特征向量，最后得到在d维空间中的距离矩阵。
'''

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)

# 创建MDS（多维标度法）模型实例，MDS是一种降维技术，用于将高维空间中的样本点投影到低维空间中
# 保持样本点之间的距离关系尽可能不变
# n_components=10：指定降维后的维度为10维
# 对数据进行MDS变换，将高维数据投影到10维低维空间中

mds = MDS(n_components=10)
x_new = mds.fit_transform(x)
x_new_df = pd.DataFrame(x_new)
print("利用sklearn进行MDS特征提取结果: \n", x_new_df)
