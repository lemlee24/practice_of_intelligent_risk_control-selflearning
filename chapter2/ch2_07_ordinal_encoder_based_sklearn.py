# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import pandas as pd
from utils import data_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

def label_encode(x):
    """
    将原始分类变量用数字编码
    :param str x: 需要编码的原始变量
    :returns: x_encoded 数字编码后的变量
    """
    le = LabelEncoder()
    x_encoded = le.fit_transform(x.astype(str))
    # 获取LabelEncoder拟合后得到的类别标签数组，按照编码顺序返回原始类别值
    class_ = le.classes_
    # 返回类别标签数组和编码后的数据框
    return class_, pd.DataFrame(x_encoded, columns=x.columns)

def ordinal_encode(x):
    """
    将原始分类变量用数字编码
    :param str x: 需要编码的原始变量，shape为[m,n]
    :returns: x_encoded 数字编码后的变量
    """
    enc = OrdinalEncoder()
    x_encoded = enc.fit_transform(x.astype(str))
    return pd.DataFrame(x_encoded).values

def main():
    """
    主函数
    """
    # 加载数据
    german_credit_data = data_utils.get_data()
    # 以特征purpose为例，进行类别编码
    class_, label_encode_x = label_encode(german_credit_data[['purpose']])
    print("特征'purpose'的类别编码结果: \n", label_encode_x)
    print("特征'purpose'编码顺序为: \n", class_)
    # 以特征purpose、credit.history为例，进行类别编码
    ordinal_encode_x = ordinal_encode(german_credit_data[['purpose', 'credit.history']])
    print("特征'purpose'和'credit.history'的类别编码结果: \n", ordinal_encode_x)

    # 代码好处和作用说明：
    # 1. LabelEncoder用于单列分类变量编码，返回编码后的数据和原始类别标签
    # 2. OrdinalEncoder用于多列分类变量编码，可以同时处理多个特征列
    # 3. 编码将字符串类型的分类变量转换为数值型，便于机器学习算法处理
    # 4. 保持了分类变量的原始映射关系，便于后续解释模型结果


if __name__ == "__main__":
    main()

