# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scorecardpy')

import os
import pandas as pd
from utils import data_utils


# 极端值检测，极端值检查方式假定不良客户异于大部分其他客户，他们在特征上的表现为集中在极端值处，即特征取值越小或越大，不良客户的浓度均越高。
def rule_evaluate(selected_df, total_df, target, rate=0.15, amount=10000):
    """
    :param selected_df: 子特征列表
    :param total_df: 特征宽表
    :param target: 目标变量
    :param rate: 息费（%）
    :param amount: 平均每笔借款金额
    :return:
    """
    # 命中规则的子群体指标统计
    # 命中规则的子群体指标统计
    hit_size = selected_df.shape[0]  # 命中样本数量
    hit_bad_size = selected_df[target].sum()  # 命中样本中的坏样本数量
    hit_bad_rate = selected_df[target].mean()  # 命中样本中的坏样本比例
    # 总体指标统计
    total_size = total_df.shape[0]  # 总样本数量
    total_bad_size = total_df[target].sum()  # 总样本中的坏样本数量
    total_bad_rate = total_df[target].mean()  # 总样本中的坏样本比例
    # 命中率
    hit_rate = hit_size / total_size  # 命中率 = 命中样本数量 / 总样本数量
    # 提升度
    lift = hit_bad_rate / total_bad_rate  # 提升度 = 命中样本坏样本比例 / 总样本坏样本比例
    # 收益
    profit = hit_bad_size * amount - (hit_size - hit_bad_size) * rate * amount  # 收益 = 命中坏样本数 * 金额 - 命中好样本数 * 利率 * 金额
    res = [total_size, total_bad_size, total_bad_rate,
           hit_rate, hit_size, hit_bad_size, hit_bad_rate, lift, profit]
    return res

# 规则效果评估
def rule_discover(data_df, var, target, rule_term, rate=0.15, amount=10000):
    """
    :param data_df: 特征宽表
    :param var: 特征名称
    :param target: 目标变量
    :param rule_term: 分位数列表或规则条件
    :param rate: 息费（%）
    :param amount: 平均每笔借款金额
    :return:
    """
    res_list = []
    if rule_term is None:
        rule_term = [0.005, 0.01, 0.02, 0.05, 0.95, 0.98, 0.99, 0.995]
    if isinstance(rule_term, list):
        for q in rule_term:
            threshold = data_df[var].quantile(q).round(2)
            if q < 0.5:
                # 对于小于分位数阈值的极端值进行检测
                # 使用pandas的query方法筛选出特征值小于等于阈值的数据子集
                temp = data_df.query("`{0}` <= @threshold".format(var))
                # 定义规则字符串，表示特征值小于等于该阈值
                rule = "<= {0}".format(threshold)
            else:
                temp = data_df.query("`{0}` >= @threshold".format(var))
                rule = ">= {0}".format(threshold)
            res = rule_evaluate(temp, data_df, target, rate, amount)
            res_list.append([var, rule] + res)
    else:
        temp = data_df.query("`{0}` {1}".format(var, rule_term))
        rule = rule_term
        res = rule_evaluate(temp, data_df, target, rate, amount)
        res_list.append([var, rule] + res)
    columns = ['var', 'rule', 'total_size', 'total_bad_size', 'total_bad_rate',
               'hit_rate', 'hit_size', 'hit_bad_size', 'hit_bad_rate', 'lift',
               'profit']
    result_df = pd.DataFrame(res_list, columns=columns)
    return result_df


if __name__ == '__main__':
    # 数据读入
    german_credit_data = data_utils.get_data()
    german_credit_data.loc[german_credit_data.sample(
        frac=0.2, random_state=0).index, 'sample_set'] = 'Train'
    german_credit_data['sample_set'] = german_credit_data['sample_set'].fillna('OOT')
    # 使用分位数列表构建规则集
    rule_table = rule_discover(data_df=german_credit_data, var='credit.amount',
                               target='creditability',
                               rule_term=[0.005, 0.01, 0.02, 0.05, 0.95, 0.98, 0.99, 0.995])
    print(rule_table)
    
    # 创建输出目录
    output_dir = 'data/rules'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录: {output_dir}")
    
    # 导出rule_table到Excel
    output_path = os.path.join(output_dir, 'rule_table_outliers.xlsx')
    rule_table.to_excel(output_path, index=False)
    print(f"\nrule_table已导出到: {output_path}")
    print(f"绝对路径: {os.path.abspath(output_path)}")
    print(f"导出数据形状: {rule_table.shape}")
    
    # 规则效果评估
    rule_analyze = german_credit_data.groupby('sample_set', group_keys=False).apply(
        lambda x: rule_discover(data_df=x, var='credit.amount',
                                target='creditability', rule_term='>12366.0'),
        include_groups=False)
    print("\nrule_analyze结果:")
    print(rule_analyze)
    
    # 导出rule_analyze到Excel
    analyze_output_path = os.path.join(output_dir, 'rule_analyze_outliers.xlsx')
    rule_analyze.to_excel(analyze_output_path, index=False)
    print(f"\nrule_analyze已导出到: {analyze_output_path}")
    print(f"绝对路径: {os.path.abspath(analyze_output_path)}")
