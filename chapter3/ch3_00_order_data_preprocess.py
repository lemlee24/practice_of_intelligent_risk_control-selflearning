# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import numpy as np
import pandas as pd
from utils.data_utils import stamp_to_date
from utils.data_utils import date_to_week


def data_preprocess(data, time_col, back_time, dtypes_dict):
    """
    数据预处理函数
    
    参数说明:
    -----------
    data : pd.DataFrame
        待处理的原始数据集
    time_col : str
        回溯依据的时间列名称，用于时间窗口过滤
    back_time : str or datetime.datetime
        特征计算时间，只保留此时间点之前的数据，防止特征穿越
    dtypes_dict : dict
        指定列字段类型的字典，例如 {'col1': int, 'col2': float}
    
    返回值:
    -------
    pd.DataFrame
        清洗完成的数据，包含新增的时间特征字段
    
    处理步骤:
    --------
    1. 删除时间列为空的行
    2. 将时间戳转换为日期格式
    3. 过滤时间窗口，防止数据泄露
    4. 处理缺失值
    5. 去除重复数据
    6. 转换数据类型
    7. 生成时间特征（星期、是否周末）
    
    注意事项:
    --------
    - back_time 必须为有效的日期格式，否则可能引发异常
    - dtypes_dict 中的类型转换必须与数据实际情况匹配
    """
    # 删除time_col为空的行
    # 过滤掉各种形式的空值表示
    data = data[~data[time_col].isin(['nan', np.nan, 'NAN', 'null', 'NULL', 'Null'])]
    
    # 将时间列的时间戳转为日期格式
    # stamp_to_date函数将Unix时间戳转换为datetime对象
    data[time_col] = data[time_col].apply(stamp_to_date)
    
    # 过滤订单创建时间在back_time之后的数据，避免特征穿越
    # 特征穿越: 使用了未来的信息来预测当前，导致模型过分乐观
    data = data[data[time_col] <= back_time]
    
    # 删除整条缺失的数据
    # how='all'表示只删除所有列都为空的行
    # 使用赋值方式替代inplace=True，避免FutureWarning
    data = data.dropna(how='all')
    
    # 空字符串替换为np.nan
    # 统一缺失值的表示方式，便于后续处理
    data = data.replace('', np.nan)
    
    # 单个字段缺失填充0
    # 对于数值型特征，用0填充是常见的策略
    data = data.fillna(0)
    
    # 去除重复数据
    # keep='first'表示保留第一条出现的记录，删除后面的重复项
    data = data.drop_duplicates(keep='first')
    
    # 字段格式转换
    # 将指定列转换为目标数据类型，确保数据类型一致性
    data = data.astype(dtypes_dict)
    
    # 补充时间特征字段
    # create_time_week: 计算该日期是星期几(1-7)
    data['create_time_week'] = data[time_col].apply(date_to_week)
    
    # is_weekend: 判断是否为周末（1表示周末，0表示工作日）
    # 星期六(6)和星期日(7)认为是周末
    data['is_weekend'] = data['create_time_week'].apply(lambda x: 1 if x > 5 else 0)

    return data


if __name__ == '__main__':
    # 原始数据读入
    orders = pd.read_excel('data/order_data.xlsx')
    # 取一个用户的历史订单数据
    raw_data = pd.DataFrame(eval(orders['data'][1]))
    # 数据预处理
    data_processed = data_preprocess(raw_data, time_col='create_time',
                                     back_time='2020-12-14',
                                     dtypes_dict={'has_overdue': int,
                                                  'application_term': float,
                                                  'application_amount': float})
    print(data_processed.shape)
