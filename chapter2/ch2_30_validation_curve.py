# -*- coding: utf-8 -*- 
# 绘制验证曲线

import sys
import numpy as np
import pandas as pd
sys.path.append("./")
sys.path.append("../")

import matplotlib
matplotlib.use('Agg')  # 使用Agg后端避免GUI相关问题
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve

X, y = load_digits(return_X_y=True)
print(X)
print(y)

# np.logspace函数用于创建在对数刻度上均匀分布的数值数组
# 参数说明：
# -6: 起始值的指数，即10^(-6) = 0.000001
# -1: 结束值的指数，即10^(-1) = 0.1
# 5: 生成5个均匀分布的数值
# 该函数生成的数组将用于验证曲线中gamma参数的不同取值
param_range = np.logspace(-6, -1, 5)


# validation_curve函数用于绘制验证曲线，帮助分析模型参数对训练和验证性能的影响
# 函数参数说明：
# estimator: 估计器对象（这里是SVC()支持向量机分类器）
# X: 训练数据特征
# y: 训练数据标签
# param_name: 需要验证的参数名称（这里是"gamma"，SVM的核函数系数）
# param_range: 参数的取值范围（这里是从10^(-6)到10^(-1)的5个对数值）
# scoring: 评估指标（这里是"accuracy"准确率）
# n_jobs: 并行计算的任务数（这里设为1，不使用并行计算）
train_scores, test_scores = validation_curve(
    SVC(), X, y, param_name="gamma", param_range=param_range,
    scoring="accuracy", n_jobs=1)
# print(train_scores)
# print(test_scores)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


plt.title("Validation Curve with SVM")
plt.xlabel(r"$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2

plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
# 解释：
# plt.semilogx - 在x轴使用对数刻度绘制线图
# param_range - x轴数据，这里是gamma参数的不同取值（对数值）
# train_scores_mean - y轴数据，这里是训练得分的平均值
# label - 图例标签，显示为"Training score"
# color - 线条颜色，设置为"darkorange"（深橙色）
# lw - 线条宽度，使用变量lw（值为2）


# plt.fill_between - 填充两条线之间的区域，用于显示标准差范围
# param_range - x轴数据，gamma参数的取值范围
# train_scores_mean - train_scores_std - 下界，训练得分平均值减去标准差
# train_scores_mean + train_scores_std - 上界，训练得分平均值加上标准差
# alpha - 透明度，设置为0.2表示半透明效果
# color - 填充颜色，与训练得分线保持一致的"darkorange"色
# lw - 边界线条宽度
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)


plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)  # 使用对数刻度绘制交叉验证得分曲线，颜色设为深蓝色，线宽为2
# 函数说明：
# plt.semilogx - 在x轴使用对数刻度绘制线图，适用于参数范围跨越多个数量级的情况
# param_range - x轴数据，gamma参数的不同取值（对数值范围从10^(-6)到10^(-1)）
# test_scores_mean - y轴数据，交叉验证得分的平均值
# label - 图例标签，显示为"Cross-validation score"
# color - 线条颜色，设置为"navy"（深蓝色），与训练得分线形成对比
# lw - 线条宽度，使用变量lw（值为2），保持与训练曲线一致的样式
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")

# plt.tight_layout() - 自动调整子图参数，防止图表元素重叠或被截断
# 该函数会自动计算并调整图形的边距、标签、标题等元素的位置
# 确保所有内容都能完整显示在图形区域内，避免出现标签被截断或重叠的情况

plt.tight_layout()
# 保存图像而不是显示，避免GUI相关错误
import os
output_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'risk_model')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, 'validation_curve.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"验证曲线已保存至：{output_path}")


