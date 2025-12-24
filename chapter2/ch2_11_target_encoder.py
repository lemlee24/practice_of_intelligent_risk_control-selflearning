# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import pandas as pd
import numpy as np
from utils import data_utils
from sklearn.preprocessing import TargetEncoder


# 加载数据
german_credit_data = data_utils.get_data()
y = german_credit_data['creditability']
x = german_credit_data[['purpose', 'personal.status.and.sex']]

# 使用sklearn的TargetEncoder进行目标编码
# target_type='continuous'表示目标变量是连续型（实际上这里是二分类，但sklearn会自动处理）
# smooth='auto'表示自动平滑，避免过拟合
enc = TargetEncoder(target_type='auto', smooth='auto', random_state=42)

# fit_transform需要传入X和y
result = enc.fit_transform(x, y)

# 将结果转换为DataFrame以便查看
result_df = pd.DataFrame(result, columns=x.columns, index=x.index)

print("目标编码结果: \n", result_df)
print("\n目标编码统计信息:")
for col in x.columns:
    print(f"\n列 '{col}' 的编码值范围: [{result_df[col].min():.4f}, {result_df[col].max():.4f}]")
    print(f"平均值: {result_df[col].mean():.4f}, 标准差: {result_df[col].std():.4f}")
