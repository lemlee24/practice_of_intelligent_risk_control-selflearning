# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# 使用聚类算法衍生特征
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def cluster_fea_gen(data, selected_cols, n_clusters):
    """
    使用K-Means聚类算法训练聚类模型
    
    原理说明:
    -----------
    K-Means聚类可以将相似的样本分组，每个簇代表一类特定的模式。
    通过聚类可以发现数据的内在结构，生成两类特征：
    1. 簇标签：样本所属的类别
    2. 距离特征：样本到簇中心的距离，反映样本的典型性
    
    参数说明:
    -----------
    data : pd.DataFrame
        原始数据集
    selected_cols : list
        用于聚类的特征列名列表
    n_clusters : int
        聚类的簇数量（K值）
    
    返回值:
    -------
    clf : KMeans
        训练好的K-Means聚类模型
    
    示例:
    -----
    >>> df = pd.DataFrame({'age': [20, 25, 60], 'income': [3000, 4000, 8000]})
    >>> model = cluster_fea_gen(df, ['age', 'income'], n_clusters=2)
    """
    # 提取用于聚类的特征列
    # loc[:, selected_cols] 确保返回DataFrame格式
    x_cluster_feas = data.loc[:, selected_cols]
    
    # 训练K-Means聚类模型
    # n_clusters: 簇的数量
    # n_init='auto': 自动选择初始化次数，避免FutureWarning
    # random_state=1: 设置随机种子，确保结果可重现
    clf = KMeans(n_clusters=n_clusters, n_init='auto', random_state=1)
    clf.fit(x_cluster_feas)
    
    return clf


def cluster_fea_apply(data, selected_cols, clf):
    """
    应用聚类模型生成衍生特征
    
    功能说明:
    -----------
    基于训练好的聚类模型，为每个样本生成两类衍生特征：
    1. 簇标签（group）：样本所属的簇编号
    2. 距离特征（distance）：样本在各特征维度上与簇中心的距离
    
    距离特征的意义：
    - 正值：样本在该特征上高于簇中心
    - 负值：样本在该特征上低于簇中心
    - 绝对值大：样本在该特征上偏离簇中心较远
    
    参数说明:
    -----------
    data : pd.DataFrame
        原始数据集
    selected_cols : list
        用于聚类的特征列名列表
    clf : KMeans
        训练好的K-Means聚类模型
    
    返回值:
    -------
    result_df : pd.DataFrame
        包含衍生特征的DataFrame，列包括：
        - group: 簇标签
        - {feature}_distance: 各特征与簇中心的距离
    
    注意事项:
    --------
    - 此函数会修改原始数据，添加中间计算列
    - 返回的DataFrame只包含衍生特征，不包含原始特征
    """
    # 创建数据副本，避免修改原始数据
    data_copy = data.copy()
    
    # 步骤1：预测样本所属的簇标签
    # predict() 返回每个样本的簇编号（0到n_clusters-1）
    data_copy['group'] = clf.predict(data_copy[selected_cols])
    
    # 步骤2：构建簇中心DataFrame
    # clf.cluster_centers_ 是形状为 (n_clusters, n_features) 的数组
    # 每行代表一个簇的中心坐标
    centers_df = pd.DataFrame(clf.cluster_centers_)
    centers_df.columns = [x + '_center' for x in selected_cols]
    
    # 步骤3：计算每个样本到其所属簇中心的距离
    for item in selected_cols:
        # 为每个样本添加其所属簇在该特征维度的中心值
        # apply(lambda x: ...) 根据簇标签查找对应的簇中心值
        data_copy[item + '_center'] = data_copy['group'].apply(
            lambda x: centers_df.iloc[x, :][item + '_center'])
        
        # 计算距离：原始值 - 簇中心值
        # 正值表示高于平均，负值表示低于平均
        data_copy[item + '_distance'] = data_copy[item] - data_copy[item + '_center']
    
    # 步骤4：选择返回的特征列
    # 返回簇标签 + 所有距离特征
    fea_cols = ['group']
    # extend方法用于将一个列表中的所有元素添加到另一个列表的末尾
    # 这里将所有距离特征列名添加到fea_cols列表中
    # 例如：如果selected_cols=['age', 'income']，则会添加['age_distance', 'income_distance']
    fea_cols.extend([x + '_distance' for x in selected_cols])
    
    return data_copy.loc[:, fea_cols]


if __name__ == '__main__':
    # ========================================================================
    # 数据读取与预处理
    # ========================================================================
    print("正在读取数据...")
    all_x_y = pd.read_excel('data/order_feas.xlsx')
    
    # 将order_no设置为索引
    all_x_y = all_x_y.set_index('order_no')
    
    # 填充缺失值为0
    # 使用赋值方式而非inplace=True，避免FutureWarning
    all_x_y = all_x_y.fillna(0)
    
    # ========================================================================
    # 选择聚类特征
    # ========================================================================
    # 选择以下几个业务特征做聚类
    # 这些特征分别代表：年龄、近90天工作日平均申请金额、历史订单数、最大逾期天数
    chose_cols = [
        'orderv1_age',                                      # 年龄
        'orderv1_90_workday_application_amount_mean',      # 90天工作日平均申请金额
        'orderv1_history_order_num',                       # 历史订单数
        'orderv1_max_overdue_days'                         # 最大逾期天数
    ]
    
    print(f"\n选择的聚类特征: {chose_cols}")
    print(f"样本数量: {len(all_x_y)}")
    
    # ========================================================================
    # 训练聚类模型并生成特征
    # ========================================================================
    print("\n正在训练K-Means聚类模型...")
    # 设置5个簇，将样本分为5类
    model = cluster_fea_gen(all_x_y, chose_cols, n_clusters=5)
    
    print(f"聚类模型信息:")
    print(f"- 簇数量: {model.n_clusters}")
    print(f"- 迭代次数: {model.n_iter_}")
    print(f"- 惯性 (inertia): {model.inertia_:.2f}")
    
    # 应用聚类模型生成衍生特征
    print("\n正在生成聚类衍生特征...")
    fea_cluster = cluster_fea_apply(all_x_y, chose_cols, model)
    
    # ========================================================================
    # 输出结果
    # ========================================================================
    print(f"\n聚类衍生特征数量: {fea_cluster.shape[1]}")
    print(f"特征列详情:")
    print(f"  - group: 簇标签 (0-{model.n_clusters-1})")
    for col in chose_cols:
        print(f"  - {col}_distance: 与簇中心在'{col}'维度的距离")
    
    print("\n聚类结果样例（前5行）:")
    print(fea_cluster.head())
    
    # 统计每个簇的样本数量
    print("\n每个簇的样本分布:")
    cluster_counts = fea_cluster['group'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        percentage = count / len(fea_cluster) * 100
        print(f"  簇 {cluster_id}: {count} 个样本 ({percentage:.1f}%)")
    
    # 显示每个簇的中心特征
    print("\n每个簇的中心特征:")
    centers_df = pd.DataFrame(model.cluster_centers_, columns=chose_cols)
    centers_df.index.name = '簇ID'
    print(centers_df.round(2))
