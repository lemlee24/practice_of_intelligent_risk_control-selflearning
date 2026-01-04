# -*- coding: utf-8 -*-

import sys

sys.path.append("./")
sys.path.append("../")

# 根据业务逻辑自动生成用户历史订单特征
import pandas as pd
import numpy as np
from dateutil.parser import parse
from utils.data_utils import stamp_to_date
from chapter3.ch3_00_order_data_preprocess import data_preprocess

func_trans = {'sum': np.sum,
              'mean': np.mean,
              'cnt': np.size,
              'max': np.max,
              'min': np.min,
              'std': np.std,
              }


def apply_func(f, *args):
    return f(*args)


def rfm_cut(data, time_col, back_time, type_dict, comp_dict, time_arr, fea_prefix='f'):
    """
    基于RFM思想切分数据，生成特征
    :param DataFrame data: 待切分的数据，时间列为create_time(timestamp)，距今天数列为gap_days
    :param str time_col: 回溯依据的时间列名称
    :param datetime.datetime back_time: 回溯时间点，datetime.datetime时间格式
    :param dict type_dict: 类别变量，以及其对应的取值类别，用于划分数据，类别列名必须在data中
    :param dict comp_dict: 指定计算字段以及对该字段采用的计算方法, 计算变量名必须在data中
    :param list time_arr: 切分时间列表(近N天)
    :param fea_prefix: 特征前缀
    :return dict: 特征
    """
    data[time_col] = data[time_col].apply(stamp_to_date)
    # 业务时间距back_time天数
    data['gap_days'] = data[time_col].apply(lambda x: (back_time - x).days)
    # 初始化特征字典，用于存储生成的所有特征
    res_feas = {}

    # 1.dict.keys()   → 只返回键(key)
    # 2.dict.values() → 只返回值(value)
    # 3.dict.items()  → 同时返回键值对(key, value)

    # 遍历时间切分数组，对每个时间窗口进行特征生成
    for col_time in time_arr:
        # 遍历计算字段字典，对每个需要计算的字段进行处理
        for col_comp in comp_dict.keys():
            # 遍历类别字段字典，获取类别字段名和对应的取值列表
            for type_k, type_v in type_dict.items():
                # 按类别和时间维度切分数据，对每个类别值进行处理
                for item in type_v:
                    # 根据时间窗口和类别值筛选数据：时间在col_time天内且类别值等于item
                    data_cut = data[(data['gap_days'] < col_time) & (data[type_k] == item)]
                    # 遍历当前计算字段对应的聚合函数列表
                    for func_k in comp_dict[col_comp]:
                        # 根据函数名称获取对应的聚合函数，如果未找到则默认使用np.size
                        func_v = func_trans.get(func_k, np.size)
                        
                        # 构建特征名称：前缀_时间窗口_类别字段_类别值_计算字段_聚合函数
                        fea_name = '%s_%s_%s_%s_%s' % (
                            fea_prefix, col_time, '%s_%s' % (type_k, item), col_comp, func_k)
                        
                        # 判断筛选后的数据是否为空，如果为空则特征值设为NaN，否则进行聚合计算
                        if data_cut.empty:
                            res_feas[fea_name] = np.nan
                        else:
                            # 对筛选出的数据在指定计算字段上应用聚合函数生成特征
                            res_feas[fea_name] = apply_func(func_v, data_cut[col_comp])
    
    # 返回生成的特征字典
    return res_feas


def gen_order_feature_auto(raw_data, time_col, back_time, dtypes_dict, type_dict, comp_dict, time_arr,
                           fea_prefix='f'):
    """
    基于RFM切分，自动生成订单特征
    :param pd.DataFrame raw_data: 原始数据
    :param str time_col: 回溯依据的时间列名称
    :param str back_time: 回溯时间点，字符串格式
    :param dict dtypes_dict: 指定列字段类型的字典，如{'col1':int}
    :param list time_arr: 切分时间列表(近N天)
    :param dict type_dict: 类别变量，以及其对应的取值类别，用于划分数据，类别列名必须在data中
    :param dict comp_dict: 指定计算字段以及对该字段采用的计算方法,计算变量名必须在data中
    :param fea_prefix: 特征前缀
    :return: res_feas 最终生成的特征
    """
    if raw_data.empty:
        return {}
    # 将回溯时间转换为datetime格式，用于后续的时间计算
    # parse函数可以解析各种字符串格式的日期，转换为datetime.datetime对象
    back_time = parse(str(back_time))

    order_df = data_preprocess(raw_data, time_col=time_col, back_time=back_time, dtypes_dict=dtypes_dict)
    if order_df.empty:
        return {}

    # 特征衍生：使用rfm切分
    res_feas = rfm_cut(order_df, time_col, back_time, type_dict, comp_dict, time_arr, fea_prefix)
    return res_feas


if __name__ == '__main__':
    # 原始数据读入
    orders = pd.read_excel('data/order_data.xlsx')
    # 取一个用户的历史订单数据
    raw_orders = pd.DataFrame(eval(orders['data'][1]))

    # 设置自动特征的参数
    # 类别字段及其取值
    type_dict_param = {
        'has_overdue': [0, 1],
        'is_weekend': [0, 1]
    }
    # 计算字段及其计算函数
    comp_dict_param = {
        'order_no': ['cnt'],
        'application_amount': ['sum', 'mean', 'max', 'min']
    }
    time_cut = [30, 90, 180, 365]

    cols_dtypes_dict = {'has_overdue': int, 'application_term': float, 'application_amount': float}

    # 根据业务逻辑生成用户历史订单特征
    features_auto = gen_order_feature_auto(raw_orders, 'create_time', '2020-12-14', cols_dtypes_dict,
                                           type_dict_param, comp_dict_param, time_cut)
    print("特征维度: ", len(features_auto.keys()))
    print(features_auto)

    # 批量生成特征
    feature_dict = {}
    for i, row in orders.iterrows():
        feature_dict[i] = gen_order_feature_auto(pd.DataFrame(eval(row['data'])), 'create_time', row['back_time'],
                                                 cols_dtypes_dict, type_dict_param, comp_dict_param, time_cut,
                                                 'order_auto')
    feature_df_auto = pd.DataFrame(feature_dict).T
    # feature_df_auto.to_excel('data/features_auto.xlsx', index=True)
