# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 使用GBDT算法做特征衡生
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier


def gbdt_fea_gen(train_data, label, n_estimators=100):
    """
    使用GBDT算法生成衍生特征
    
    原理说明:
    -----------
    GBDT(Gradient Boosting Decision Tree)可以自动学习特征组合和非线性关系。
    通过提取样本在GBDT每棵树叶子节点的位置，并进行One-Hot编码，
    可以将高维稀疏的叶子节点特征作为新的输入特征。
    
    参数说明:
    -----------
    train_data : pd.DataFrame or np.ndarray
        训练数据特征矩阵
    label : pd.Series or np.ndarray
        训练数据标签
    n_estimators : int, default=100
        GBDT模型中树的数量，树越多生成的特征越多
    
    返回值:
    -------
    gbc_model : GradientBoostingClassifier
        训练好的GBDT模型
    one_hot_encoder : OneHotEncoder
        训练好的One-Hot编码器
    
    示例:
    -----
    >>> X = pd.DataFrame([[1, 2], [3, 4]])
    >>> y = pd.Series([0, 1])
    >>> model, encoder = gbdt_fea_gen(X, y, n_estimators=10)
    """
    # 训练GBDT模型
    # random_state=1：设置随机种子，确保结果可重现
    gbc_model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=1)
    
    # 转换为numpy数组以避免特征名称警告
    # 使用isinstance函数检查train_data是否为pandas DataFrame类型
    # isinstance是Python内置函数，用于判断对象是否属于指定的类型
    # 语法：isinstance(object, type) -> bool
    # 如果train_data是pd.DataFrame类型则返回True，否则返回False
    # 这里用于统一数据格式：如果是DataFrame则转换为numpy数组，避免sklearn训练时的警告
    if isinstance(train_data, pd.DataFrame):
        train_data_array = train_data.values
    else:
        train_data_array = train_data
    
    # 训练模型
    gbc_model.fit(train_data_array, label)

    # 得到样本元素落在叶节点中的位置
    # apply()方法返回每个样本在每棵树中所在的叶子节点索引
    # reshape(-1, n_estimators)：将结果重塑为(n_samples, n_estimators)的形状
    train_leaf_fea = gbc_model.apply(train_data_array).reshape(-1, n_estimators)

    # 借用One-Hot编码将位置信息转化为0、1的稀疏矩阵
    # 每个叶子节点位置都会被编码为一个独立的二值特征
    one_hot_encoder = OneHotEncoder()
    one_hot_encoder.fit(train_leaf_fea)
    
    return gbc_model, one_hot_encoder


def gbdt_fea_apply(data, model, encoder):
    """
    应用GBDT模型生成衡生特征
    
    功能说明:
    -----------
    将训练好的GBDT模型和One-Hot编码器应用到新数据上，
    生成高维的衡生特征。这些特征捕捉了原始特征之间的
    非线性组合和交互关系。
    
    参数说明:
    -----------
    data : pd.DataFrame or np.ndarray
        需要生成衡生特征的数据
    model : GradientBoostingClassifier
        训练好的GBDT模型
    encoder : OneHotEncoder
        训练好的One-Hot编码器
    
    返回值:
    -------
    new_fea : pd.DataFrame
        生成的衡生特征，列名为fea_1, fea_2, ...
    
    注意事项:
    --------
    - 输入数据的特征顺序和类型必须与训练时一致
    - 生成的特征是稀疏矩阵，大部分值为0
    """
    # 保存原始索引
    original_index = data.index if isinstance(data, pd.DataFrame) else None
    
    # 转换为numpy数组以避免特征名称警告
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data
    
    # 获得GBDT特征
    # 1. model.apply(data_array): 获取每个样本在每棵树的叶子节点位置
    # 2. reshape(-1, model.n_estimators): 重塑为矩阵形状
    # 3. encoder.transform(): One-Hot编码转换
    # 4. toarray(): 将稀疏矩阵转为密集矩阵
    new_feature_train = encoder.transform(
        model.apply(data_array).reshape(-1, model.n_estimators)
    ).toarray()

    # 将生成的新特征转换为DataFrame
    new_fea = pd.DataFrame(new_feature_train)
    
    # 恢复原始索引
    if original_index is not None:
        new_fea.index = original_index
    
    # 设置列名：fea_1, fea_2, ..., fea_N
    new_fea.columns = ['fea_%s' % i for i in range(1, new_fea.shape[1] + 1)]
    
    return new_fea


if __name__ == '__main__':
    # 读取原始特征数据
    all_x_y = pd.read_excel('data/order_feas.xlsx')
    all_x_y.set_index('order_no', inplace=True)
    # 生成训练数据
    x_train = all_x_y.drop(columns='label')
    x_train.fillna(0, inplace=True)
    y = all_x_y['label']
    # 获取特征
    gbr, encode = gbdt_fea_gen(x_train, y, n_estimators=100)
    new_features = gbdt_fea_apply(x_train, gbr, encode)
    print("使用GBDT算法衍生特征结果: \n", new_features.head())
