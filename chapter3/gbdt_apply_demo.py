# -*- coding: utf-8 -*-

"""
GBDT apply() 方法演示
演示如何获取样本在决策树叶子节点的位置
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# ============================================================================
# 示例1：基础演示
# ============================================================================
print("=" * 70)
print("示例1：基础演示 - 理解 apply() 和 reshape() 的作用")
print("=" * 70)

# 创建简单的训练数据：3个样本，2个特征
X = np.array([[1, 2], 
              [3, 4], 
              [5, 6]])
y = np.array([0, 1, 0])

print("\n训练数据 X:")
print(X)
print("\n标签 y:")
print(y)

# 训练GBDT模型（2棵树，便于观察）
model = GradientBoostingClassifier(n_estimators=2, max_depth=3, random_state=42)
model.fit(X, y)

print(f"\n模型信息:")
print(f"- 树的数量: {model.n_estimators}")
print(f"- 每棵树的最大深度: {model.max_depth}")

# 步骤1：使用 apply() 获取叶子节点位置（一维数组）
leaf_positions_1d = model.apply(X)
print(f"\n步骤1 - apply() 返回的一维数组:")
print(f"形状: {leaf_positions_1d.shape}")
print(f"内容: {leaf_positions_1d}")
print(f"说明: 这是 {len(X)} 个样本 × {model.n_estimators} 棵树 = {len(leaf_positions_1d)} 个元素")

# 步骤2：使用 reshape() 重塑为二维矩阵
leaf_positions_2d = leaf_positions_1d.reshape(-1, model.n_estimators)
print(f"\n步骤2 - reshape(-1, {model.n_estimators}) 返回的二维矩阵:")
print(f"形状: {leaf_positions_2d.shape}")
print("内容:")
print(leaf_positions_2d)

print("\n详细解释:")
for i, (sample, positions) in enumerate(zip(X, leaf_positions_2d)):
    print(f"样本{i+1} {sample} → 在树1的叶子节点{positions[0]}, 在树2的叶子节点{positions[1]}")


# ============================================================================
# 示例2：更多棵树的情况
# ============================================================================
print("\n" + "=" * 70)
print("示例2：使用更多棵树 (5棵树)")
print("=" * 70)

# 创建更多样本
X_large = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y_large = np.array([0, 1, 0, 1, 0])

# 训练5棵树的模型
model_large = GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=42)
model_large.fit(X_large, y_large)

print(f"\n训练数据: {len(X_large)} 个样本")
print(f"树的数量: {model_large.n_estimators}")

# 获取叶子节点位置矩阵
leaf_matrix = model_large.apply(X_large).reshape(-1, model_large.n_estimators)
print(f"\n叶子节点位置矩阵形状: {leaf_matrix.shape}")
print(f"(行数 = 样本数 = {leaf_matrix.shape[0]}, 列数 = 树的数量 = {leaf_matrix.shape[1]})")
print("\n叶子节点位置矩阵:")
print(leaf_matrix)

print("\n每个样本在各棵树中的叶子节点:")
for i in range(len(X_large)):
    print(f"样本{i+1}: {leaf_matrix[i]}")


# ============================================================================
# 示例3：统计叶子节点信息
# ============================================================================
print("\n" + "=" * 70)
print("示例3：统计叶子节点信息")
print("=" * 70)

print(f"\n每棵树的叶子节点分布:")
for tree_idx in range(model_large.n_estimators):
    unique_leaves = np.unique(leaf_matrix[:, tree_idx])
    print(f"树{tree_idx+1}: 使用了 {len(unique_leaves)} 个不同的叶子节点 → {unique_leaves}")

print(f"\n所有叶子节点的唯一值数量: {len(np.unique(leaf_matrix))}")
print(f"所有叶子节点的唯一值: {sorted(np.unique(leaf_matrix))}")


# ============================================================================
# 示例4：One-Hot编码预览
# ============================================================================
print("\n" + "=" * 70)
print("示例4：One-Hot编码预览（这是GBDT特征衍生的下一步）")
print("=" * 70)

from sklearn.preprocessing import OneHotEncoder

# 对叶子节点位置进行One-Hot编码
encoder = OneHotEncoder(sparse_output=False)
leaf_onehot = encoder.fit_transform(leaf_matrix)

print(f"\nOne-Hot编码后的特征矩阵形状: {leaf_onehot.shape}")
print(f"原始特征数: {X_large.shape[1]}")
print(f"衍生特征数: {leaf_onehot.shape[1]}")
print(f"特征数增加: {X_large.shape[1]} → {leaf_onehot.shape[1]} (增加了 {leaf_onehot.shape[1] - X_large.shape[1]} 个)")

print(f"\n前3个样本的One-Hot编码特征（前20列）:")
print(leaf_onehot[:3, :20])
print("...")

print(f"\n特征稀疏度:")
print(f"- 非零元素数量: {np.count_nonzero(leaf_onehot)}")
print(f"- 总元素数量: {leaf_onehot.size}")
print(f"- 非零元素比例: {np.count_nonzero(leaf_onehot) / leaf_onehot.size * 100:.2f}%")
print(f"- 每个样本激活的特征数: {np.count_nonzero(leaf_onehot, axis=1)}")


# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print("""
GBDT特征衍生流程:
1. 训练GBDT模型
2. 使用 apply() 获取样本在每棵树的叶子节点位置
3. 使用 reshape() 将结果整理为矩阵形式 (n_samples, n_trees)
4. 使用 One-Hot编码将叶子节点位置转换为高维稀疏特征
5. 得到的新特征捕捉了原始特征的非线性组合关系

优势:
- 自动特征组合，无需人工设计
- 非线性特征转换
- 捕捉特征交互关系
- 可解释性强（每个特征对应一个决策路径）
""")
