# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import pandas as pd
from utils import data_utils
from sklearn.preprocessing import OrdinalEncoder

# 加载数据
german_credit_data = data_utils.get_data()

# 选择要编码的列
cols_to_encode = ['purpose', 'personal.status.and.sex']

# 初始化sklearn的OrdinalEncoder类
# handle_unknown='use_encoded_value'表示未知值使用unknown_value参数指定的值
# unknown_value=-1表示未知值被编码为-1
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', 
                         unknown_value=-1,
                         encoded_missing_value=-2)

# 对指定列进行编码
encoded_values = encoder.fit_transform(german_credit_data[cols_to_encode])

# 创建结果DataFrame
result = german_credit_data.copy()
result[cols_to_encode] = encoded_values

# 构建类别映射关系（模拟category_encoders的输出格式）
category_mapping = []
for i, col in enumerate(cols_to_encode):
    categories = encoder.categories_[i]
    mapping = {cat: idx for idx, cat in enumerate(categories)}
    mapping['缺失值'] = -2
    mapping['未知值'] = -1
    category_mapping.append({
        'col': col,
        'mapping': pd.Series(mapping),
        'categories': list(categories)
    })

print("类别编码结果: \n", result)
print("\n类别编码映射关系:")
for item in category_mapping:
    print(f"\n列名: {item['col']}")
    print(f"映射关系:\n{item['mapping']}")
    print(f"原始类别: {item['categories']}")
