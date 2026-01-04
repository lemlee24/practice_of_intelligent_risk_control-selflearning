# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 时间序列特征挖掘
import pandas as pd
from tsfresh.feature_extraction import extract_features

if __name__ == '__main__':
    # 读取数据
    # 读取原始订单数据文件
    orders = pd.read_excel('data/order_data.xlsx')
    
    # 初始化处理后的数据列表
    orders_new = []
    
    # 遍历每个订单记录，解析嵌套的data字段
    for i in range(len(orders)):
        # 将字符串格式的data字段转换为DataFrame格式
        sub_data = pd.DataFrame.from_records(eval(orders['data'][i]))
        # 添加用户ID字段，用于后续时间序列特征提取的分组
        sub_data['uid'] = orders['uid'][i]
        # 将处理后的子数据添加到列表中
        orders_new.append(sub_data)
    
    # 合并所有处理后的子数据为一个完整的DataFrame
    orders_new_df = pd.concat(orders_new)
    
    # 数据类型转换：确保数值字段为浮点型，便于后续特征计算
    orders_new_df['application_amount'] = orders_new_df['application_amount'].astype(float)
    orders_new_df['has_overdue'] = orders_new_df['has_overdue'].astype(float)

    # 使用tsfresh库提取时间序列特征
    # column_id指定用户ID列，用于区分不同用户的时间序列
    # column_sort指定时间排序列，确保按时间顺序处理
    order_feas = extract_features(
        orders_new_df[['uid', 'create_time', 'application_amount', 'has_overdue']], 
        column_id="uid", 
        column_sort="create_time"
    )
    
    # 输出特征提取结果信息
    print("时间序列挖掘特征数: \n", order_feas.shape[1])
    print("时间序列特征挖掘结果: \n", order_feas.head())
