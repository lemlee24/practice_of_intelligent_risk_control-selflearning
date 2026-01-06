# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import os
import sklearn.tree as st
import graphviz
from utils import data_utils


def decision_tree_resolve(train_x, train_y, class_names=None, max_depth=3, fig_path=''):
    """
    基于决策树可视化
    
    参数说明:
    -----------
    train_x : pd.DataFrame
        训练数据特征
    train_y : pd.Series
        训练数据标签
    class_names : list, default=None
        标签名称，默认为['good', 'bad']
    max_depth : int, default=3
        树最大深度
    fig_path : str
        图片保存路径和名称（不包括扩展名）
    
    返回值:
    -------
    graph : graphviz.Source
        决策树可视化对象
    """
    if class_names is None:
        class_names = ['good', 'bad']
    
    # 训练决策树模型
    clf = st.DecisionTreeClassifier(max_depth=max_depth,
                                    min_samples_leaf=0.01,
                                    min_samples_split=0.01,
                                    criterion='gini',
                                    splitter='best',
                                    max_features=None)
    clf = clf.fit(train_x, train_y)

    # 生成比例图
    dot_data = st.export_graphviz(clf, out_file=None,
                                  feature_names=train_x.columns.tolist(),
                                  class_names=class_names,
                                  filled=True,
                                  rounded=True,
                                  node_ids=True,
                                  special_characters=True,
                                  proportion=True,
                                  leaves_parallel=True)
    graph = graphviz.Source(dot_data, filename=fig_path)
    return graph


# ============================================================================
# 数据加载与预处理
# ============================================================================
# 加载数据
german_credit_data = data_utils.get_data()

# 构造数据集
X = german_credit_data[data_utils.numeric_cols].copy()
y = german_credit_data['creditability']

# ============================================================================
# 创建输出目录
# ============================================================================
output_dir = 'data/rules'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建目录: {output_dir}")

# ============================================================================
# 生成决策树可视化
# ============================================================================
print("\n正在生成决策树可视化...")
fig_path = os.path.join(output_dir, 'decision_tree')
graph = decision_tree_resolve(X, y, fig_path=fig_path)

# 保存图片（生成PDF和PNG格式）
print(f"正在保存决策树图片...")
graph.render(cleanup=True)  # 生成PDF并清理中间文件
print(f"\n决策树图片已保存到:")
print(f"  - PDF: {os.path.abspath(fig_path + '.pdf')}")

# 尝试生成PNG格式（如果系统支持）
try:
    graph.format = 'png'
    graph.render(cleanup=True)
    print(f"  - PNG: {os.path.abspath(fig_path + '.png')}")
except Exception as e:
    print(f"  注意: PNG格式生成失败，请查看 PDF文件")

print("\n决策树信息:")
print(f"- 最大深度: 3")
print(f"- 特征数量: {X.shape[1]}")
print(f"- 样本数量: {X.shape[0]}")

# ============================================================================
# 转化为规则
# ============================================================================
print("\n正在生成决策树规则特征...")
# 根据决策树叶子节点条件生成规则特征
X['node_5'] = X.apply(lambda x: 1 if x['duration.in.month'] <= 34.5 and x['credit.amount'] > 8630.5 else 0, axis=1)
X['node_9'] = X.apply(
    lambda x: 1 if x['duration.in.month'] > 34.5 and x['age.in.years'] <= 29.5 and x['credit.amount'] > 4100.0 else 0,
    axis=1)
X['node_12'] = X.apply(lambda x: 1 if x['duration.in.month'] > 34.5 and x['age.in.years'] > 56.5 else 0, axis=1)

print("\n生成的规则特征:")
print(f"- node_5: duration.in.month <= 34.5 AND credit.amount > 8630.5")
print(f"- node_9: duration.in.month > 34.5 AND age.in.years <= 29.5 AND credit.amount > 4100.0")
print(f"- node_12: duration.in.month > 34.5 AND age.in.years > 56.5")

print(f"\n规则命中统计:")
for col in ['node_5', 'node_9', 'node_12']:
    hit_count = X[col].sum()
    hit_rate = X[col].mean() * 100
    print(f"- {col}: 命中 {hit_count} 个样本 ({hit_rate:.2f}%)")
