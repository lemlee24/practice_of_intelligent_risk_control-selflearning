# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import os
import toad
import numpy as np
import pandas as pd
from utils import data_utils
from toad.plot import bin_plot
from matplotlib import pyplot as plt


def cal_iv(x, y):
    """ 
    IV计算函数  
    :param x: feature 
    :param y: label 
    :return: 
    """
    # 使用pd.crosstab函数创建交叉表，统计x和y的联合分布
    # x: 特征变量，y: 目标变量(标签)
    # margins=True: 添加行和列的总计
    # 交叉表的行表示特征的不同取值或分组，列表示目标变量的取值(0和1，即good和bad)
    # 交叉表的每个单元格表示对应特征值和目标值的样本数量
    crtab = pd.crosstab(x, y, margins=True)
    crtab.columns = ['good', 'bad', 'total']  # 重命名列名为good, bad, total
    crtab['factor_per'] = crtab['total'] / len(y)  # 计算各分组样本占比
    crtab['bad_per'] = crtab['bad'] / crtab['total']  # 计算各分组坏样本占比
    crtab['p'] = crtab['bad'] / crtab.loc['All', 'bad']  # 计算各分组坏样本占总坏样本比例
    crtab['q'] = crtab['good'] / crtab.loc['All', 'good']  # 计算各分组好样本占总好样本比例
    crtab['woe'] = np.log(crtab['p'] / crtab['q'])  # 计算WOE值(Weight of Evidence)
    crtab2 = crtab[abs(crtab.woe) != np.inf]  # 过滤掉WOE值为无穷大的分组

    crtab['IV'] = sum(
        (crtab2['p'] - crtab2['q']) * np.log(crtab2['p'] / crtab2['q']))  # 计算IV值(Information Value)
    crtab.reset_index(inplace=True)
    crtab['varname'] = crtab.columns[0]
    crtab.rename(columns={crtab.columns[0]: 'var_level'}, inplace=True)
    crtab.var_level = crtab.var_level.apply(str)
    return crtab


german_credit_data = data_utils.get_data()

# ============================================================================
# 初始化输出目录
# ============================================================================
output_dir = 'data/rules'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建目录: {output_dir}")

# 生成分箱初始化对象  
bin_transformer = toad.transform.Combiner()

# 采用等距分箱训练  
bin_transformer.fit(german_credit_data,
                    y='creditability',
                    n_bins=6,
                    method='step',
                    empty_separate=True)

# 分箱数据  
trans_data = bin_transformer.transform(german_credit_data, labels=True)

# ============================================================================
# 分箱可视化与保存
# ============================================================================
# 查看Credit amount分箱结果  
bin_plot(trans_data, x='credit.amount', target='creditability')

# 保存分箱图形
fig_path = os.path.join(output_dir, 'credit_amount_binning.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\n分箱图形已保存到: {fig_path}")
print(f"绝对路径: {os.path.abspath(fig_path)}")

# plt.show()show

# ============================================================================
# IV计算与导出
# ============================================================================
# 查看Credit amount分箱数据  
cal_iv = cal_iv(trans_data['credit.amount'], trans_data['creditability'])

# 将列名改为中文
cal_iv_chinese = cal_iv.rename(columns={
    'var_level': '分箱区间',
    'good': '好样本数',
    'bad': '坏样本数',
    'total': '总样本数',
    'factor_per': '样本占比',
    'bad_per': '坏样本率',
    'p': '坏样本分布',
    'q': '好样本分布',
    'woe': 'WOE值',
    'IV': 'IV值',
    'varname': '变量名称'
})

# 导出cal_iv到Excel文件
output_path = os.path.join(output_dir, 'cal_iv.xlsx')
cal_iv_chinese.to_excel(output_path, index=False)
print(f"\nIV计算结果已导出到: {output_path}")
print(f"绝对路径: {os.path.abspath(output_path)}")
print(f"导出数据形状: {cal_iv_chinese.shape}")
print("\ncal_iv预览（中文列名）:")
print(cal_iv_chinese)

# ============================================================================
# 构建单规则
# ============================================================================
german_credit_data['credit.amount.rule'] = np.where(german_credit_data['credit.amount'] > 12366.0, 1, 0)
