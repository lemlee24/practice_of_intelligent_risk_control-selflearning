# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import pandas as pd
import datetime as dt
from dateutil.parser import parse
from chapter3.ch3_00_order_data_preprocess import data_preprocess


def calculate_age(born_day, back_time=None):
    """
    根据出生日期解析年龄
    :param born_day: 出生日期
    :param back_time: 回溯时间，默认当前日期
    :return: 年龄
    """
    if back_time is None:
        today = dt.date.today()
    else:
        today = back_time
    if isinstance(born_day, str):
        born_day = parse(born_day)
    if isinstance(today, str):
        today = parse(today)
    return today.year - born_day.year - ((today.month, today.day) < (born_day.month, born_day.day))


def gen_order_feature_manual(data, time_col, back_time, dtypes_dict, fea_prefix='f'):
    """
    根据业务逻辑生成特征
    :param data: 业务订单原始数据
    :param time_col: 回溯依据的时间列名称
    :param back_time: 回溯时间点
    :param dtypes_dict: 指定列字段类型的字典，如{'col1':int}
    :param fea_prefix: 特征前缀
    :return: features，根据业务逻辑生成的特征
    """
    # 数据预处理函数，见文件ch3_01_order_data_preprocess.py
    data_processed = data_preprocess(data, time_col, back_time, dtypes_dict=dtypes_dict)
    features = {}
    # 从生日解析年龄
    # 使用 %s_age % fea_prefix 的方式为特征名添加前缀，这样可以避免特征名冲突
    # %s 是字符串格式化占位符，fea_prefix 是传入的特征前缀参数
    # 例如，如果 fea_prefix='f'，则最终特征名为 'f_age'
    features['%s_age' % fea_prefix] = calculate_age(data_processed.get('birthday')[0], back_time)
    # 用户历史订单数
    features['%s_history_order_num' % fea_prefix] = data_processed.shape[0]
    # 用户历史逾期次数
    features['%s_overdue_num' % fea_prefix] = data_processed['has_overdue'].sum()
    # 用户历史最大逾期天数
    features['%s_max_overdue_days' % fea_prefix] = data_processed['overdue_days'].max()
    # 用户历史平均逾期天数
    features['%s_mean_overdue_days' % fea_prefix] = data_processed['overdue_days'].mean()

    return features


if __name__ == '__main__':
    # 原始数据读入
    orders = pd.read_excel('data/order_data.xlsx')
    # 取一个用户的历史订单数据
    # 使用eval函数将字符串形式的订单数据转换为DataFrame
    # eval函数可以执行字符串形式的Python表达式，这里用于解析存储在data列中的字符串格式订单数据
    # 将字符串转换为实际的Python对象（如字典或列表），然后构建DataFrame
    raw_data = pd.DataFrame(eval(orders['data'][1]))
    # 获取回溯时间值，用于计算年龄和历史订单统计
    back_time_value = orders['back_time'][1]
    cols_dtypes_dict = {'has_overdue': int, 'application_term': float, 'application_amount': float}

    # 根据业务逻辑生成用户历史订单特征
    features_manual = gen_order_feature_manual(raw_data, 'create_time', back_time_value, cols_dtypes_dict)
    print(features_manual)

    # 批量生成特征
    feature_dict = {}
    # 遍历每个用户的历史订单数据，批量生成特征
    # iterrows() 是pandas DataFrame的方法，用于逐行迭代数据
    # i: 行索引，row: 当前行的数据Series对象
    # 这样可以对每个用户的数据单独处理，生成对应的特征
    for i, row in orders.iterrows():
        feature_dict[i] = gen_order_feature_manual(pd.DataFrame(eval(row['data'])), 'create_time', row['back_time'],
                                                   cols_dtypes_dict, fea_prefix='orderv1')
    feature_df = pd.DataFrame(feature_dict).T
    # feature_df.to_excel('data/features_manual.xlsx', index=True)
