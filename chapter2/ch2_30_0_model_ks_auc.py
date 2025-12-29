# -*- coding: utf-8 -*- 
# 计算模型KS值和AUC值

import sys
import numpy as np
import pandas as pd
sys.path.append("./")
sys.path.append("../")

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端避免GUI相关问题
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def calculate_ks_auc(y_true, y_pred_proba):
    """
    计算模型的KS值和AUC值
    
    参数:
    y_true: array-like, 真实标签 (0或1)
    y_pred_proba: array-like, 预测概率值
    
    返回:
    dict: 包含KS值、AUC值和KS出现位置的字典
    """
    # 计算AUC值
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # 计算KS值
    # KS = max(TPR - FPR)
    ks = max(tpr - fpr)
    
    # 找到KS值对应的索引位置
    ks_index = np.argmax(tpr - fpr)
    ks_threshold = thresholds[ks_index]
    
    return {
        'KS': ks,
        'AUC': auc,
        'KS_Threshold': ks_threshold,
        'FPR': fpr,
        'TPR': tpr,
        'Thresholds': thresholds,
        'KS_Index': ks_index
    }


def plot_ks_curve(y_true, y_pred_proba, save_path=None):
    """
    绘制KS曲线
    
    参数:
    y_true: array-like, 真实标签
    y_pred_proba: array-like, 预测概率值
    save_path: str, 图像保存路径，默认为None则保存到桌面risk_model文件夹
    """
    # 计算KS和AUC
    result = calculate_ks_auc(y_true, y_pred_proba)
    
    # 创建DataFrame用于分析
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred_proba': y_pred_proba
    })
    
    # 按预测概率降序排序
    df = df.sort_values('y_pred_proba', ascending=False).reset_index(drop=True)
    
    # 计算累积坏样本率和好样本率
    df['bad'] = df['y_true']
    df['good'] = 1 - df['y_true']
    
    total_bad = df['bad'].sum()
    total_good = df['good'].sum()
    
    df['bad_cumsum'] = df['bad'].cumsum()
    df['good_cumsum'] = df['good'].cumsum()
    
    df['bad_rate'] = df['bad_cumsum'] / total_bad
    df['good_rate'] = df['good_cumsum'] / total_good
    df['ks'] = abs(df['bad_rate'] - df['good_rate'])
    
    # 绘制KS曲线
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(df)) / len(df)
    plt.plot(x, df['bad_rate'], label='累积坏样本率 (TPR)', color='red', lw=2)
    plt.plot(x, df['good_rate'], label='累积好样本率 (FPR)', color='blue', lw=2)
    
    # 标注KS值
    ks_index = df['ks'].idxmax()
    ks_value = df['ks'].max()
    
    plt.plot([x[ks_index], x[ks_index]], 
             [df['good_rate'].iloc[ks_index], df['bad_rate'].iloc[ks_index]], 
             'k--', lw=2, label=f'KS={ks_value:.4f}')
    
    plt.scatter([x[ks_index]], [df['good_rate'].iloc[ks_index]], 
                color='blue', s=100, zorder=5)
    plt.scatter([x[ks_index]], [df['bad_rate'].iloc[ks_index]], 
                color='red', s=100, zorder=5)
    
    plt.xlabel('样本占比', fontsize=12)
    plt.ylabel('累积占比', fontsize=12)
    plt.title(f'KS曲线 (KS={ks_value:.4f}, AUC={result["AUC"]:.4f})', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    if save_path is None:
        import os
        output_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'risk_model')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, 'ks_curve.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"KS曲线已保存至：{save_path}")
    plt.close()
    
    return result


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    绘制ROC曲线
    
    参数:
    y_true: array-like, 真实标签
    y_pred_proba: array-like, 预测概率值
    save_path: str, 图像保存路径，默认为None则保存到桌面risk_model文件夹
    """
    # 计算KS和AUC
    result = calculate_ks_auc(y_true, y_pred_proba)
    
    fpr = result['FPR']
    tpr = result['TPR']
    auc = result['AUC']
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='随机模型')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (FPR)', fontsize=12)
    plt.ylabel('真正例率 (TPR)', fontsize=12)
    plt.title('ROC曲线', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    if save_path is None:
        import os
        output_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'risk_model')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, 'roc_curve.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC曲线已保存至：{save_path}")
    plt.close()
    
    return result


# 示例用法
if __name__ == "__main__":
    # 使用German Credit数据集作为示例
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from utils import data_utils

    # 加载数据并转换为二分类问题
    all_x_y = data_utils.get_all_x_y()
    X = all_x_y.drop(data_utils.label, axis=1)
    y = all_x_y.pop(data_utils.label)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 训练逻辑回归模型

    # 创建逻辑回归模型实例
    # max_iter=1000: 设置最大迭代次数为1000，确保模型充分收敛
    # random_state=42: 设置随机种子，确保结果可复现
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测概率
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算KS和AUC
    result = calculate_ks_auc(y_test, y_pred_proba)
    
    print("=" * 50)
    print("模型评估指标:")
    print("=" * 50)
    print(f"AUC: {result['AUC']:.4f}")
    print(f"KS:  {result['KS']:.4f}")
    print(f"KS对应的阈值: {result['KS_Threshold']:.4f}")
    print("=" * 50)
    
    # 绘制KS曲线
    plot_ks_curve(y_test, y_pred_proba)
    
    # 绘制ROC曲线
    plot_roc_curve(y_test, y_pred_proba)
    
    print("\n所有图像已保存完成！")
