# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def calculate_vintage(data, loan_date_col, obs_date_col, status_col, 
                     loan_amount_col=None, overdue_days_col=None,
                     freq='M', max_mob=12, dpd_threshold=30):
    """
    计算分期贷款vintage分析
    
    参数说明:
    ----------
    data : pd.DataFrame
        贷款数据，每行代表一笔贷款
    loan_date_col : str
        放款日期列名
    obs_date_col : str
        观察日期列名（当前状态的观察时点）
    status_col : str
        贷款状态列名（0-正常，1-逾期/违约）
    loan_amount_col : str, optional
        贷款金额列名，用于计算金额维度的vintage
    overdue_days_col : str, optional
        逾期天数列名，用于更精细的逾期判断
    freq : str, default='M'
        放款批次频率，'M'-月度, 'Q'-季度, 'Y'-年度
    max_mob : int, default=12
        最大观察月数（Month on Book）
    dpd_threshold : int, default=30
        逾期天数阈值，超过该天数算作违约（Days Past Due）
    
    返回:
    ----------
    dict : 包含以下键值对
        - vintage_count: 笔数维度的vintage表
        - vintage_rate: 违约率维度的vintage表
        - vintage_amount: 金额维度的vintage表（如果提供了loan_amount_col）
        - vintage_amount_rate: 金额违约率维度的vintage表（如果提供了loan_amount_col）
        - summary: 汇总统计信息
    """
    
    # 数据验证
    df = data.copy()
    required_cols = [loan_date_col, obs_date_col, status_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")
    
    # 转换日期格式
    df[loan_date_col] = pd.to_datetime(df[loan_date_col])
    df[obs_date_col] = pd.to_datetime(df[obs_date_col])
    
    # 计算放款批次（vintage）
    if freq == 'M':
        df['vintage'] = df[loan_date_col].dt.to_period('M')
    elif freq == 'Q':
        df['vintage'] = df[loan_date_col].dt.to_period('Q')
    elif freq == 'Y':
        df['vintage'] = df[loan_date_col].dt.to_period('Y')
    else:
        raise ValueError("freq参数只支持 'M', 'Q', 'Y'")
    
    # 计算MOB (Month on Book) - 在账月数
    df['mob'] = ((df[obs_date_col].dt.year - df[loan_date_col].dt.year) * 12 + 
                 (df[obs_date_col].dt.month - df[loan_date_col].dt.month))
    
    # 确保MOB不为负数
    df['mob'] = df['mob'].clip(lower=0, upper=max_mob)
    
    # 判断是否违约
    if overdue_days_col and overdue_days_col in df.columns:
        df['is_default'] = ((df[status_col] == 1) | 
                           (df[overdue_days_col] >= dpd_threshold)).astype(int)
    else:
        df['is_default'] = df[status_col].astype(int)
    
    # ========================================================================
    # 1. 计算笔数维度的vintage
    # ========================================================================
    vintage_count = df.groupby(['vintage', 'mob']).agg({
        loan_date_col: 'count',  # 总笔数
        'is_default': 'sum'       # 违约笔数
    }).rename(columns={loan_date_col: 'total_count', 'is_default': 'default_count'})
    
    # 计算累计违约笔数
    vintage_count['cumulative_default'] = vintage_count.groupby(level=0)['default_count'].cumsum()
    
    # 重置索引并透视表
    vintage_count_pivot = vintage_count.reset_index().pivot(
        index='vintage', 
        columns='mob', 
        values='cumulative_default'
    )
    
    # ========================================================================
    # 2. 计算违约率维度的vintage
    # ========================================================================
    # 计算各vintage的总笔数
    total_by_vintage = df.groupby('vintage')[loan_date_col].count()
    
    # 计算累计违约率
    vintage_rate_pivot = vintage_count_pivot.div(total_by_vintage, axis=0) * 100
    
    # ========================================================================
    # 3. 如果提供了金额列，计算金额维度的vintage
    # ========================================================================
    vintage_amount_pivot = None
    vintage_amount_rate_pivot = None
    
    if loan_amount_col and loan_amount_col in df.columns:
        # 计算金额维度的vintage
        vintage_amount = df.groupby(['vintage', 'mob']).agg({
            loan_amount_col: 'sum',  # 总金额
            'is_default': lambda x: df.loc[x.index, loan_amount_col][df.loc[x.index, 'is_default'] == 1].sum()  # 违约金额
        }).rename(columns={loan_amount_col: 'total_amount', 'is_default': 'default_amount'})
        
        # 计算累计违约金额
        vintage_amount['cumulative_default_amt'] = vintage_amount.groupby(level=0)['default_amount'].cumsum()
        
        # 透视表
        vintage_amount_pivot = vintage_amount.reset_index().pivot(
            index='vintage',
            columns='mob',
            values='cumulative_default_amt'
        )
        
        # 计算金额违约率
        total_amount_by_vintage = df.groupby('vintage')[loan_amount_col].sum()
        vintage_amount_rate_pivot = vintage_amount_pivot.div(total_amount_by_vintage, axis=0) * 100
    
    # ========================================================================
    # 4. 生成汇总统计信息
    # ========================================================================
    summary = {
        'total_loans': len(df),
        'total_defaults': df['is_default'].sum(),
        'overall_default_rate': (df['is_default'].sum() / len(df) * 100),
        'vintage_periods': df['vintage'].nunique(),
        'max_mob_observed': df['mob'].max(),
        'date_range': f"{df[loan_date_col].min().strftime('%Y-%m-%d')} 至 {df[loan_date_col].max().strftime('%Y-%m-%d')}"
    }
    
    if loan_amount_col and loan_amount_col in df.columns:
        summary['total_amount'] = df[loan_amount_col].sum()
        summary['default_amount'] = df[df['is_default'] == 1][loan_amount_col].sum()
        summary['amount_default_rate'] = (summary['default_amount'] / summary['total_amount'] * 100)
    
    # ========================================================================
    # 返回结果
    # ========================================================================
    result = {
        'vintage_count': vintage_count_pivot,
        'vintage_rate': vintage_rate_pivot,
        'summary': summary
    }
    
    if vintage_amount_pivot is not None:
        result['vintage_amount'] = vintage_amount_pivot
        result['vintage_amount_rate'] = vintage_amount_rate_pivot
    
    return result


def plot_vintage(vintage_result, plot_type='rate', figsize=(14, 8), 
                cmap='YlOrRd', save_path=None):
    """
    可视化vintage分析结果
    
    参数说明:
    ----------
    vintage_result : dict
        calculate_vintage函数的返回结果
    plot_type : str
        绘图类型，'rate'-违约率, 'count'-违约笔数, 'amount'-违约金额, 'amount_rate'-金额违约率
    figsize : tuple
        图形大小
    cmap : str
        颜色映射
    save_path : str, optional
        图形保存路径
    """
    
    if plot_type == 'rate':
        data = vintage_result['vintage_rate']
        title = 'Vintage分析 - 累计违约率(%)'
        fmt = '.2f'
    elif plot_type == 'count':
        data = vintage_result['vintage_count']
        title = 'Vintage分析 - 累计违约笔数'
        fmt = '.0f'
    elif plot_type == 'amount':
        if 'vintage_amount' not in vintage_result:
            raise ValueError("未提供金额数据，无法绘制金额维度图表")
        data = vintage_result['vintage_amount']
        title = 'Vintage分析 - 累计违约金额'
        fmt = '.0f'
    elif plot_type == 'amount_rate':
        if 'vintage_amount_rate' not in vintage_result:
            raise ValueError("未提供金额数据，无法绘制金额违约率图表")
        data = vintage_result['vintage_amount_rate']
        title = 'Vintage分析 - 累计金额违约率(%)'
        fmt = '.2f'
    else:
        raise ValueError("plot_type参数只支持 'rate', 'count', 'amount', 'amount_rate'")
    
    # 绘制热力图
    plt.figure(figsize=figsize)
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5, 
                cbar_kws={'label': title})
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('MOB (Month on Book)', fontsize=12)
    plt.ylabel('放款批次 (Vintage)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图形已保存到: {save_path}")
    
    plt.show()


def plot_vintage_curve(vintage_result, figsize=(14, 8), save_path=None):
    """
    绘制vintage曲线图（折线图）
    
    参数说明:
    ----------
    vintage_result : dict
        calculate_vintage函数的返回结果
    figsize : tuple
        图形大小
    save_path : str, optional
        图形保存路径
    """
    
    data = vintage_result['vintage_rate']
    
    plt.figure(figsize=figsize)
    for idx in data.index:
        plt.plot(data.columns, data.loc[idx], marker='o', label=str(idx))
    
    plt.title('Vintage曲线 - 累计违约率趋势', fontsize=16, pad=20)
    plt.xlabel('MOB (Month on Book)', fontsize=12)
    plt.ylabel('累计违约率 (%)', fontsize=12)
    plt.legend(title='放款批次', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"曲线图已保存到: {save_path}")
    
    plt.show()


def export_vintage_to_excel(vintage_result, output_path):
    """
    将vintage分析结果导出到Excel文件
    
    参数说明:
    ----------
    vintage_result : dict
        calculate_vintage函数的返回结果
    output_path : str
        Excel文件保存路径
    """
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 导出违约率表
        vintage_result['vintage_rate'].to_excel(writer, sheet_name='违约率')
        
        # 导出违约笔数表
        vintage_result['vintage_count'].to_excel(writer, sheet_name='违约笔数')
        
        # 如果有金额数据，也导出
        if 'vintage_amount' in vintage_result:
            vintage_result['vintage_amount'].to_excel(writer, sheet_name='违约金额')
            vintage_result['vintage_amount_rate'].to_excel(writer, sheet_name='金额违约率')
        
        # 导出汇总信息
        summary_df = pd.DataFrame([vintage_result['summary']]).T
        summary_df.columns = ['统计值']
        summary_df.to_excel(writer, sheet_name='汇总统计')
    
    print(f"\nVintage分析结果已导出到: {output_path}")
    print(f"绝对路径: {os.path.abspath(output_path)}")


# ============================================================================
# 示例用法
# ============================================================================
if __name__ == "__main__":
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 5000
    
    # 生成放款日期（2023年1月到2024年12月）
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2024-12-31')
    loan_dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # 生成观察日期（放款后1-12个月）
    obs_dates = loan_dates + pd.to_timedelta(np.random.randint(30, 365, n_samples), unit='D')
    
    # 生成贷款金额
    loan_amounts = np.random.uniform(5000, 100000, n_samples).round(2)
    
    # 生成逾期天数（大部分正常，少部分逾期）
    overdue_days = np.random.choice([0, 15, 45, 90], n_samples, p=[0.85, 0.08, 0.05, 0.02])
    
    # 生成贷款状态（基于逾期天数）
    status = (overdue_days >= 30).astype(int)
    
    # 构建DataFrame
    loan_data = pd.DataFrame({
        'loan_date': loan_dates,
        'observation_date': obs_dates,
        'loan_amount': loan_amounts,
        'overdue_days': overdue_days,
        'status': status,
        'loan_id': range(1, n_samples + 1)
    })
    
    print("=" * 80)
    print("分期贷款Vintage分析示例")
    print("=" * 80)
    print(f"\n数据概览:")
    print(loan_data.head(10))
    print(f"\n数据形状: {loan_data.shape}")
    
    # ========================================================================
    # 计算vintage
    # ========================================================================
    vintage_result = calculate_vintage(
        data=loan_data,
        loan_date_col='loan_date',
        obs_date_col='observation_date',
        status_col='status',
        loan_amount_col='loan_amount',
        overdue_days_col='overdue_days',
        freq='M',
        max_mob=12,
        dpd_threshold=30
    )
    
    # ========================================================================
    # 输出结果
    # ========================================================================
    print("\n" + "=" * 80)
    print("汇总统计信息")
    print("=" * 80)
    for key, value in vintage_result['summary'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 80)
    print("累计违约率表 (Vintage Rate %)")
    print("=" * 80)
    print(vintage_result['vintage_rate'].round(2))
    
    # ========================================================================
    # 导出到Excel
    # ========================================================================
    output_dir = 'data/rules'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'vintage_analysis.xlsx')
    export_vintage_to_excel(vintage_result, output_path)
    
    # ========================================================================
    # 可视化
    # ========================================================================
    # 绘制违约率热力图
    plot_vintage(vintage_result, plot_type='rate', 
                save_path=os.path.join(output_dir, 'vintage_rate_heatmap.png'))
    
    # 绘制违约率曲线图
    plot_vintage_curve(vintage_result, 
                      save_path=os.path.join(output_dir, 'vintage_rate_curve.png'))
    
    print("\n" + "=" * 80)
    print("Vintage分析完成！")
    print("=" * 80)
