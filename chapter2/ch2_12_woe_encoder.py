# -*- coding: utf-8 -*- 
import sys
sys.path.append("./")
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
from utils import data_utils
import scorecardpy as sc
# 加载数据
data = data_utils.get_data()
# 准备数据：只需要特征列和目标列
data_woe = data[['purpose', 'personal.status.and.sex', 'creditability']].copy()
data_woe = data_woe.astype({'purpose': str, 'personal.status.and.sex': str, 'creditability': int})
# WOE分箱
bins = sc.woebin(data_woe, y='creditability')

# 将WOE分箱信息保存到Excel（所有变量在同一个工作表）
import pandas as pd
import os
desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
excel_path = os.path.join(desktop, 'woe_bins_info.xlsx')

# 合并所有分箱信息到一个DataFrame
all_bins = pd.concat([bins[col] for col in ['purpose', 'personal.status.and.sex']], ignore_index=True)
all_bins.to_excel(excel_path, sheet_name='WOE分箱信息', index=False)
print(f"WOE分箱信息已保存到桌面: {excel_path}")

# 显示分箱信息
print("WOE分箱信息:")
for col in ['purpose', 'personal.status.and.sex']:
    print(f"\n变量: {col}")
    print(bins[col][['bin', 'count', 'good', 'bad', 'woe', 'bin_iv', 'total_iv']])
# WOE编码
result = sc.woebin_ply(data_woe, bins)
# 提取WOE编码列
woe_result = result[['purpose_woe', 'personal.status.and.sex_woe']]
woe_result.columns = ['purpose', 'personal.status.and.sex']  # 重命名
print("\n" + "="*50)
print("WOE编码结果: \n", woe_result)
print("\nWOE编码统计:")
for col in woe_result.columns:
    print(f"\n{col}: 范围[{woe_result[col].min():.4f}, {woe_result[col].max():.4f}], "
          f"均值{woe_result[col].mean():.4f}, 唯一值{woe_result[col].nunique()}个")

