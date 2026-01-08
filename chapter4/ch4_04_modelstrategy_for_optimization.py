# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d
from sklearn.metrics import r2_score
from scipy.optimize import minimize


def calculate_pass_loss_decile(score_series, y_series):
    """
    模型分取值变化时通过率与坏账率关系
    :param score_series: 模型分
    :param y_series: Y标签
    :return:  
    """
    decile_df = pd.crosstab(score_series, y_series).rename(columns={0: 'N_nonEvent', 1: 'N_Event'})
    decile_df.loc[:, 'N_sample'] = score_series.value_counts()

    decile_df.loc[:, 'EventRate'] = decile_df.N_Event * 1.0 / decile_df.N_sample
    decile_df.loc[:, 'BadPct'] = decile_df.N_Event * 1.0 / sum(decile_df.N_Event)
    decile_df.loc[:, 'GoodPct'] = decile_df.N_nonEvent * 1.0 / sum(decile_df.N_nonEvent)
    decile_df.loc[:, 'CumBadPct'] = decile_df.BadPct.cumsum()
    decile_df.loc[:, 'CumGoodPct'] = decile_df.GoodPct.cumsum()

    decile_df = decile_df.sort_index(ascending=False)
    decile_df.loc[:, 'ApprovalRate'] = decile_df.N_sample.cumsum() / decile_df.N_sample.sum()
    decile_df.loc[:, 'ApprovedEventRate'] = decile_df.N_Event.cumsum() / decile_df.N_sample.cumsum()
    decile_df = decile_df.sort_index(ascending=True)
    return decile_df


def poly_regression(x_series, y_series, degree, plot=True, save_path=None):
    """
    多项式回归拟合
    :param x_series: x数据
    :param y_series: y数据
    :param degree: 指定多项式次数
    :param plot: 是否作图
    :param save_path: 图片保存路径
    :return:
    """
    coeff = polyfit(x_series, y_series, degree)
    f = poly1d(coeff)
    R2 = r2_score(y_series.values, f(x_series))

    print(f'coef:{coeff},R2: {R2}')

    if plot:
        # 用来正常显示中文标签
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(10, 5))
        plt.plot(x_series, y_series, 'rx', label='实际数据')
        plt.plot(x_series, f(x_series), 'b-', label=f'拟合曲线 (R²={R2:.4f})')
        plt.xlabel('通过率', {'size': 15})
        plt.ylabel('坏账率', {'size': 15})
        plt.title('通过率与坏账率关系拟合', {'size': 16})
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        plt.show()
    return coeff


def find_best_approval_rate(x_to_loss_func, score_df):
    """
    定义最优化函数
    坏账率L(x)与通过率x的关系函数
    :param x_to_loss_func: 坏账率与通过率的函数关系
    :param score_df: 模型分与通过率的对应关系，index为模型分，"ApprovalRate"列为对应的通过率
    :return:
    """

    # 定义目标函数，求解最大值即为负的最小值
    def fun(x_array):
        # 其中x_list[0]为通过率x，x_array[1]为对应的坏账率L(x)
        return -10000 * (0.16 * (1 - x_array[1]) - x_array[1]
                         - 30 / (x_array[0] * 0.6) / 10000)

    # eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0， 下面式子1e-6项确保相应变量不等于0或1
    cons = ({'type': 'eq', 'fun': lambda x_array: x_to_loss_func(x_array[0]) - x_array[1]},
            {'type': 'ineq', 'fun': lambda x_array: x_array[0] - 1e-6},
            {'type': 'ineq', 'fun': lambda x_array: x_array[1] - 1e-6},
            {'type': 'ineq', 'fun': lambda x_array: 1 - x_array[0] - 1e-6},
            {'type': 'ineq', 'fun': lambda x_array: 1 - x_array[0] - 1e-6}
            )

    # 设置初始值
    x_base = np.array((0.10, 0.10))
    # 采用SLSQP进行最优化求解
    res = minimize(fun, x_base, method='SLSQP', constraints=cons)
    print('利润最优：', "{:.2f}".format(-res.fun))
    print('最优解对应通过率：', "{:.2%}".format(res.x[0]), '坏账率：', "{:.2%}".format(res.x[1]))
    print("模型分阈值：", score_df[score_df['ApprovalRate'] >= res.x[0]].index.max())
    print('迭代终止是否成功：', res.success)
    print('迭代终止原因：', res.message)


# ============================================================================
# 主程序
# ============================================================================
print("="*80)
print("模型策略优化分析")
print("="*80)

# 创建输出目录
output_dir = 'data/rules'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建目录: {output_dir}")

german_score = pd.read_csv('data/german_score.csv')
print(f"\n数据加载完成: {german_score.shape}")

decile_df = calculate_pass_loss_decile(german_score['score'],
                                       german_score['creditability'])
print("\n通过率与坏账率关系表（前5行）:")
print(decile_df.head())

# 导出decile_df到Excel
decile_output_path = os.path.join(output_dir, 'approval_rate_loss_rate.xlsx')
decile_df.to_excel(decile_output_path)
print(f"\n通过率坏账率关系表已导出到: {decile_output_path}")
print(f"绝对路径: {os.path.abspath(decile_output_path)}")

# 数据准备
print("\n" + "="*80)
print("多项式回归拟合")
print("="*80)
x = decile_df['ApprovalRate']
# 逾期率折算为坏账率
y = decile_df['ApprovedEventRate'] / 2.5

fig_path = os.path.join(output_dir, 'approval_loss_regression.png')
poly_coef = poly_regression(x, y, 2, plot=True, save_path=fig_path)
# 坏账率L(x)与通过率x的关系
l_x = poly1d(poly_coef)
print("\n拟合函数:")
print(l_x)

print("\n" + "="*80)
print("最优化求解")
print("="*80)
find_best_approval_rate(l_x, decile_df)

print("\n" + "="*80)
print("分析完成！")
print("="*80)
