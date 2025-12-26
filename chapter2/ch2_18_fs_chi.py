# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
y = all_x_y.pop(data_utils.label)  # pop函数用于移除字典中指定键的项，并返回该键对应的值；这里将标签列从数据中分离出来作为目标变量y
# 选择K个最好的特征，返回选择特征后的数据
# 使用SelectKBest进行特征选择，基于卡方检验(chi2)评分方法，选择k=5个最优特征
# SelectKBest: sklearn中的特征选择器，可以选择评分最高的k个特征
# chi2: 卡方检验，用于评估分类任务中特征与标签之间的相关性，数值越大表示相关性越强
# k=5: 指定选择前5个最优特征
fs_chi = SelectKBest(chi2, k=5)

# 使用fit方法在特征数据(all_x_y)和目标变量(y)上训练特征选择器
# 计算每个特征的卡方统计量和p值，用于后续特征排序和选择
fs_chi.fit(all_x_y, y)

# 使用transform方法将特征选择器应用到原始特征数据上
# 只保留通过卡方检验筛选出的前k个最优特征，返回降维后的特征矩阵
x_new = fs_chi.transform(all_x_y)

# fs_chi.get_support() 返回一个布尔数组，表示哪些特征被选中（True）哪些被过滤掉（False）
# all_x_y.columns 是特征列名的索引对象
# 通过布尔索引 all_x_y.columns[fs_chi.get_support()] 可以获取被选中的特征列名
# tolist() 方法将索引对象转换为列表格式，便于后续处理和显示
selected_cols = all_x_y.columns[fs_chi.get_support()].tolist()
print("卡方检验筛选得到%s个特征: \n" % len(selected_cols), selected_cols)
