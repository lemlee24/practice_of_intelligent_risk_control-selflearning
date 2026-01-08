# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scorecardpy')

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from chapter4.ch4_01_rules_for_outliers import rule_discover
from utils import data_utils

# 加载数据
print("="*80)
print("孤立森林异常检测 (Isolation Forest)")
print("="*80)

german_credit_data = data_utils.get_data()

# 构造数据集
X = german_credit_data[data_utils.numeric_cols]
y = german_credit_data['creditability']

print(f"\n数据信息:")
print(f"- 样本数量: {X.shape[0]}")
print(f"- 特征数量: {X.shape[1]}")
print(f"- 特征列表: {list(X.columns)}")

# 初始化模型（使用sklearn的IsolationForest）
print(f"\n正在训练孤立森林模型...")
clf = IsolationForest(
    # behaviour参数在新版本sklearn中已废弃，使用默认值即可
    n_estimators=500,
    max_samples='auto',
    contamination=0.1,  # 假设10%的数据为异常值
    max_features=1.0,
    bootstrap=False,
    random_state=20,
    verbose=0
)

# 训练模型  
clf.fit(X)

# 预测结果（-1表示异常，1表示正常）
predictions = clf.predict(X)
# 获取异常分数（负值越大越异常）
scores = clf.score_samples(X)

# 将分数转换为概率（0-1之间，值越大越异常）
# 使用sigmoid函数转换
german_credit_data['out_pred'] = 1 / (1 + np.exp(scores))

# 将预测概率大于0.7以上的设为异常值  
german_credit_data['iforest_rule'] = np.where(german_credit_data['out_pred'] > 0.7, 1, 0)

print(f"模型训练完成！")
print(f"\n异常检测统计:")
print(f"- 检测到异常样本: {german_credit_data['iforest_rule'].sum()}")
print(f"- 异常比例: {german_credit_data['iforest_rule'].mean()*100:.2f}%")
print(f"- 平均异常分数: {german_credit_data['out_pred'].mean():.4f}")
print(f"- 最大异常分数: {german_credit_data['out_pred'].max():.4f}")
print(f"- 最小异常分数: {german_credit_data['out_pred'].min():.4f}")

# 效果评估  
print(f"="*80)
print("规则效果评估")
print("="*80)
rule_iforest = rule_discover(data_df=german_credit_data, var='iforest_rule', target='creditability', rule_term='==1')
print("\n孤立森林评估结果:")
print(rule_iforest)

# 创建输出目录并导出结果
output_dir = 'data/rules'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\n已创建目录: {output_dir}")

# 导出规则评估结果到Excel
output_path = os.path.join(output_dir, 'rule_isolationforest.xlsx')
rule_iforest.to_excel(output_path, index=False)
print(f"\n规则评估结果已导出到: {output_path}")
print(f"绝对路径: {os.path.abspath(output_path)}")

# 导出带有异常分数的数据
data_with_scores = german_credit_data[['creditability', 'out_pred', 'iforest_rule'] + data_utils.numeric_cols].copy()
scores_output_path = os.path.join(output_dir, 'isolationforest_scores.xlsx')
data_with_scores.to_excel(scores_output_path, index=False)
print(f"异常分数数据已导出到: {scores_output_path}")
print(f"绝对路径: {os.path.abspath(scores_output_path)}")

print(f"="*80)
print("分析完成！")
print("="*80)

