# -*- coding: utf-8 -*- 

import sys
import toad
import os
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from utils import data_utils

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
# 利用toad库quality()方法计算IV
# 使用toad.quality函数计算变量的IV值和其他质量指标
# toad.quality函数参数说明：
# - data: 输入的数据集
# - target: 目标变量列名，这里设置为'creditability'表示信用能力
# - method: 分箱方法，'quantile'表示等频分箱
# - n_bins: 分箱数量，设置为6个箱子
# - iv_only: 是否只返回IV值，True表示只返回IV值而不返回其他质量指标
var_iv = toad.quality(all_x_y,
                      target='creditability',
                      method='quantile',
                      n_bins=6)

selected_cols = var_iv[var_iv.iv > 0.1].index.tolist()
print("各特征的iv值计算结果: \n", var_iv)
print("设置iv阈值为0.1, 筛选得到%s个特征: \n" % len(selected_cols), selected_cols)

# 将IV值结果导出到Excel
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
risk_model_dir = os.path.join(desktop_path, 'risk_model')
# 创建risk_model文件夹（如果不存在）
if not os.path.exists(risk_model_dir):
    os.makedirs(risk_model_dir)

# 保存到Excel
save_path = os.path.join(risk_model_dir, 'var_iv_results.xlsx')
var_iv_df = pd.DataFrame(var_iv)
var_iv_df.to_excel(save_path, sheet_name='IV值结果', index=True)
print(f"\nIV值结果已保存至: {save_path}")
