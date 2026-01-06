# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scorecardpy')

import os
import pandas as pd
from utils import data_utils


# ============================================================================
# 极端值检测 - 规则评估函数
# ============================================================================
# 极端值检测，极端值检查方式假定不良客户异于大部分其他客户，他们在特征上的表现为集中在极端值处，即特征取值越小或越大，不良客户的浓度均越高。
def rule_evaluate(selected_df, total_df, target, rate=0.15, amount=10000):
    """
    评估规则的效果，计算命中率、提升度、收益等指标
    
    参数说明:
    -----------
    selected_df : pd.DataFrame
        命中规则的子群体数据
    total_df : pd.DataFrame
        全量数据集
    target : str
        目标变量名称（标签列）
    rate : float, default=0.15
        息费率（15%）
    amount : int, default=10000
        平均每笔借款金额
    
    返回值:
    -------
    list
        包含以下指标的列表：
        [total_size, total_bad_size, total_bad_rate,
         hit_rate, hit_size, hit_bad_size, hit_bad_rate, lift, profit]
    
    指标说明:
    ---------
    - total_size: 总样本数
    - total_bad_size: 总坏样本数
    - total_bad_rate: 总坏样本率
    - hit_rate: 命中率 = 命中样本数 / 总样本数
    - hit_size: 命中样本数
    - hit_bad_size: 命中坏样本数
    - hit_bad_rate: 命中坏样本率
    - lift: 提升度 = 命中坏样本率 / 总坏样本率
    - profit: 收益 = 拒绝坏客户收益 - 误拒好客户损失
    """
    # 命中规则的子群体指标统计
    hit_size = selected_df.shape[0]  # 命中样本数量
    hit_bad_size = selected_df[target].sum()  # 命中样本中的坏样本数量
    hit_bad_rate = selected_df[target].mean()  # 命中样本中的坏样本比例
    
    # 总体指标统计
    total_size = total_df.shape[0]  # 总样本数量
    total_bad_size = total_df[target].sum()  # 总样本中的坏样本数量
    total_bad_rate = total_df[target].mean()  # 总样本中的坏样本比例
    
    # 命中率：规则覆盖的样本占比
    hit_rate = hit_size / total_size  # 命中率 = 命中样本数量 / 总样本数量
    
    # 提升度：规则对坏样本的识别能力
    lift = hit_bad_rate / total_bad_rate  # 提升度 = 命中样本坏样本比例 / 总样本坏样本比例
    
    # 收益计算：拒绝坏客户避免损失 - 误拒好客户的机会成本
    # profit = 命中坏样本数 * 金额 - 命中好样本数 * 利率 * 金额
    profit = hit_bad_size * amount - (hit_size - hit_bad_size) * rate * amount
    
    res = [total_size, total_bad_size, total_bad_rate,
           hit_rate, hit_size, hit_bad_size, hit_bad_rate, lift, profit]
    return res

# ============================================================================
# 规则发现函数
# ============================================================================
def rule_discover(data_df, var, target, rule_term, rate=0.15, amount=10000):
    """
    根据给定的分位数或条件发现规则并评估其效果
    
    参数说明:
    -----------
    data_df : pd.DataFrame
        特征数据集
    var : str
        特征名称
    target : str
        目标变量名称
    rule_term : list or str
        分位数列表（如[0.01, 0.05, 0.95, 0.99]）或规则条件字符串（如'>12366.0'）
    rate : float, default=0.15
        息费率
    amount : int, default=10000
        平均每笔借款金额
    
    返回值:
    -------
    pd.DataFrame
        包含规则及其评估指标的DataFrame
    
    示例:
    -----
    >>> rule_discover(df, 'credit.amount', 'creditability', [0.01, 0.99])
    >>> rule_discover(df, 'credit.amount', 'creditability', '>12366.0')
    """
    res_list = []
    
    # 如果rule_term为None，使用默认分位数
    if rule_term is None:
        rule_term = [0.005, 0.01, 0.02, 0.05, 0.95, 0.98, 0.99, 0.995]
    
    # 如果rule_term是列表，按分位数生成规则
    if isinstance(rule_term, list):
        for q in rule_term:
            # 计算分位数阈值
            threshold = data_df[var].quantile(q).round(2)
            
            if q < 0.5:
                # 对于小于分位数阈值的极端值进行检测
                # 使用pandas的query方法筛选出特征值小于等于阈值的数据子集
                temp = data_df.query("`{0}` <= @threshold".format(var))
                # 定义规则字符串，表示特征值小于等于该阈值
                rule = "<= {0}".format(threshold)
            else:
                # 对于大于分位数阈值的极端值进行检测
                temp = data_df.query("`{0}` >= @threshold".format(var))
                rule = ">= {0}".format(threshold)
            
            # 评估规则效果
            res = rule_evaluate(temp, data_df, target, rate, amount)
            res_list.append([var, rule] + res)
    else:
        # 如果rule_term是字符串，直接使用作为规则条件
        temp = data_df.query("`{0}` {1}".format(var, rule_term))
        rule = rule_term
        res = rule_evaluate(temp, data_df, target, rate, amount)
        res_list.append([var, rule] + res)
    
    # 构建结果DataFrame
    columns = ['var', 'rule', 'total_size', 'total_bad_size', 'total_bad_rate',
               'hit_rate', 'hit_size', 'hit_bad_size', 'hit_bad_rate', 'lift',
               'profit']
    result_df = pd.DataFrame(res_list, columns=columns)
    return result_df


if __name__ == '__main__':
    # ========================================================================
    # 数据读入与预处理
    # ========================================================================
    german_credit_data = data_utils.get_data()
    
    # 划分数据集：20%为训练集，80%为OOT样本（Out-Of-Time）
    german_credit_data.loc[german_credit_data.sample(
        frac=0.2, random_state=0).index, 'sample_set'] = 'Train'
    german_credit_data['sample_set'] = german_credit_data['sample_set'].fillna('OOT')
    
    # ========================================================================
    # 创建输出目录
    # ========================================================================
    output_dir = 'data/rules'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录: {output_dir}")
    
    # ========================================================================
    # 使用分位数列表构建规则集
    # ========================================================================
    print("\n正在使用分位数方法发现规则...")
    rule_table = rule_discover(data_df=german_credit_data, var='credit.amount',
                               target='creditability',
                               rule_term=[0.005, 0.01, 0.02, 0.05, 0.95, 0.98, 0.99, 0.995])
    print("\nrule_table结果:")
    print(rule_table)
    
    # 导出rule_table到Excel
    output_path = os.path.join(output_dir, 'rule_table_outliers.xlsx')
    rule_table.to_excel(output_path, index=False)
    print(f"\nrule_table已导出到: {output_path}")
    print(f"绝对路径: {os.path.abspath(output_path)}")
    print(f"导出数据形状: {rule_table.shape}")
    
    # ========================================================================
    # 规则效果评估 - 分样本集分析
    # ========================================================================
    print("\n正在分样本集评估规则效果...")
    # 修复: 移除include_groups参数，使用group_keys=False已经足够
    rule_analyze = german_credit_data.groupby('sample_set', group_keys=False).apply(
        lambda x: rule_discover(data_df=x, var='credit.amount',
                                target='creditability', rule_term='>12366.0'))
    
    print("\nrule_analyze结果:")
    print(rule_analyze)
    
    # 导出rule_analyze到Excel
    analyze_output_path = os.path.join(output_dir, 'rule_analyze_outliers.xlsx')
    rule_analyze.to_excel(analyze_output_path, index=False)
    print(f"\nrule_analyze已导出到: {analyze_output_path}")
    print(f"绝对路径: {os.path.abspath(analyze_output_path)}")
    print(f"导出数据形状: {rule_analyze.shape}")
