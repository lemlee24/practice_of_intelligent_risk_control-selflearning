# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

from utils import data_utils
import matplotlib
matplotlib.use('TkAgg')  # 设置兼容后端，解决tostring_rgb缺失问题
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 黑体、微软雅黑、宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 适合构建相关性系数矩阵

# 导入数值型样例数据
all_x_y = data_utils.get_all_x_y()
x = all_x_y.drop(data_utils.label, axis=1)
# 利用pandas库计算相关系数
# pearson相关系数
pearson_corr = x.corr(method='pearson')
print("pandas库计算 pearson相关系数: \n", pearson_corr)

# 绘制pearson相关系数热力图
plt.figure(figsize=(12, 10))
sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0, cbar_kws={"shrink": 0.8},
            annot_kws={'fontsize': 6})  # 减小数据字体大小，取消格子间空格
plt.title('Pearson相关系数热力图', fontsize=16, fontweight='bold')
plt.tight_layout()

# 保存到桌面
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
save_path = os.path.join(desktop_path, 'pearson_corr_heatmap.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n热力图已保存至: {save_path}")
# plt.show()  # 注释掉plt.show()避免在PyCharm控制台中报错，图片已保存到桌面

# spearman相关系数
spearman_corr = x.corr(method='spearman')  
print("\npandas库计算 spearman相关系数: \n", spearman_corr)
# kendall相关系数
kendall_corr = x.corr(method='kendall')  
print("\npandas库计算 kendall相关系数: \n", kendall_corr)
