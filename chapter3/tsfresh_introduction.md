# tsfresh 详细介绍与使用指南

## 目录
- [1. tsfresh 简介](#1-tsfresh-简介)
- [2. 核心概念](#2-核心概念)
- [3. 安装与配置](#3-安装与配置)
- [4. 基本用法](#4-基本用法)
- [5. 高级功能](#5-高级功能)
- [6. 实际应用案例](#6-实际应用案例)
- [7. 最佳实践](#7-最佳实践)
- [8. 常见问题](#8-常见问题)

---

## 1. tsfresh 简介

### 1.1 什么是 tsfresh？

**tsfresh**（Time Series Feature extraction based on scalable hypothesis tests）是一个用于**时间序列特征提取**的 Python 库。

### 1.2 主要特点

- 🚀 **自动化特征工程**：自动从时间序列中提取大量统计特征
- 📊 **特征选择**：基于假设检验自动筛选相关特征
- ⚡ **高性能**：支持并行计算，处理大规模数据集
- 🎯 **适用场景广泛**：金融、医疗、工业、风控等领域

### 1.3 核心优势

| 优势 | 说明 |
|------|------|
| **自动化** | 无需手动设计特征，自动提取800+种特征 |
| **科学性** | 基于统计假设检验筛选特征，避免过拟合 |
| **可扩展** | 支持分布式计算，处理TB级数据 |
| **易用性** | API简洁，与pandas、sklearn无缝集成 |

---

## 2. 核心概念

### 2.1 时间序列数据格式

tsfresh 要求数据包含以下三个关键列：

```python
| id  | time | value |
|-----|------|-------|
| 1   | 0    | 3.5   |
| 1   | 1    | 4.2   |
| 1   | 2    | 3.8   |
| 2   | 0    | 5.1   |
| 2   | 1    | 4.9   |
```

- **id**：标识不同的时间序列（如用户ID、订单ID）
- **time**：时间戳或序列索引
- **value**：观测值

### 2.2 特征类别

tsfresh 提取的特征分为以下几类：

#### 2.2.1 统计特征
- 均值、中位数、方差、标准差
- 最大值、最小值、极差
- 偏度（skewness）、峰度（kurtosis）
- 分位数（25%, 50%, 75%）

#### 2.2.2 时序特征
- 自相关系数（Autocorrelation）
- 偏自相关系数（Partial Autocorrelation）
- 趋势强度
- 季节性指标

#### 2.2.3 频域特征
- 傅里叶变换系数
- 功率谱密度
- 频谱质心

#### 2.2.4 复杂度特征
- 近似熵（Approximate Entropy）
- 样本熵（Sample Entropy）
- C3统计量
- CID（Complexity-Invariant Distance）

#### 2.2.5 形态特征
- 峰值数量
- 过零点数量
- 长度统计
- 变化率

---

## 3. 安装与配置

### 3.1 安装

```bash
# 基础安装
pip install tsfresh

# 包含所有依赖
pip install tsfresh[all]

# 特定版本
pip install tsfresh==0.20.1
```

### 3.2 依赖库

```python
# 核心依赖
numpy >= 1.15.1
pandas >= 0.25.0
scipy >= 1.2.0
statsmodels >= 0.9.0
scikit-learn >= 0.22.0

# 可选依赖
dask  # 分布式计算
```

### 3.3 验证安装

```python
import tsfresh
print(tsfresh.__version__)
```

---

## 4. 基本用法

### 4.1 快速开始

#### 示例1：最简单的用法

```python
from tsfresh import extract_features
import pandas as pd

# 准备时间序列数据
df = pd.DataFrame({
    'id': [1, 1, 1, 2, 2, 2],
    'time': [0, 1, 2, 0, 1, 2],
    'value': [3.5, 4.2, 3.8, 5.1, 4.9, 5.3]
})

# 提取特征
features = extract_features(df, column_id='id', column_sort='time')
print(features.shape)  # 输出：(2, 794) - 2个ID，794个特征
```

#### 示例2：多变量时间序列

```python
df = pd.DataFrame({
    'id': [1, 1, 1, 2, 2, 2],
    'time': [0, 1, 2, 0, 1, 2],
    'value1': [3.5, 4.2, 3.8, 5.1, 4.9, 5.3],
    'value2': [1.2, 1.5, 1.1, 2.0, 1.8, 2.1]
})

# 自动识别所有数值列作为特征列
features = extract_features(
    df, 
    column_id='id', 
    column_sort='time'
)
```

### 4.2 特征提取配置

#### 4.2.1 使用预定义设置

```python
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters

# 最小特征集（快速）
minimal_features = extract_features(
    df, 
    column_id='id', 
    column_sort='time',
    default_fc_parameters=MinimalFCParameters()
)

# 完整特征集（全面）
comprehensive_features = extract_features(
    df, 
    column_id='id', 
    column_sort='time',
    default_fc_parameters=ComprehensiveFCParameters()
)
```

#### 4.2.2 自定义特征参数

```python
from tsfresh.feature_extraction import EfficientFCParameters

# 自定义特征提取参数
custom_settings = {
    "length": None,  # 序列长度
    "mean": None,    # 均值
    "median": None,  # 中位数
    "variance": None, # 方差
    "standard_deviation": None,  # 标准差
    "maximum": None,  # 最大值
    "minimum": None,  # 最小值
    "sum_values": None,  # 总和
    "quantile": [{"q": 0.25}, {"q": 0.75}],  # 分位数
    "autocorrelation": [{"lag": 1}, {"lag": 2}],  # 自相关
}

features = extract_features(
    df,
    column_id='id',
    column_sort='time',
    default_fc_parameters=custom_settings
)
```

### 4.3 特征选择

```python
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

# 假设我们有目标变量
y = pd.Series([0, 1], index=[1, 2])

# 填充缺失值
features_imputed = impute(features)

# 基于假设检验选择相关特征
features_filtered = select_features(
    features_imputed, 
    y,
    fdr_level=0.05  # 假发现率阈值
)

print(f"原始特征数: {features.shape[1]}")
print(f"筛选后特征数: {features_filtered.shape[1]}")
```

---

## 5. 高级功能

### 5.1 并行计算

```python
from tsfresh import extract_features
from tsfresh.utilities.distribution import MultiprocessingDistributor

# 使用多进程加速
Distributor = MultiprocessingDistributor(
    n_workers=4,  # 使用4个进程
    disable_progressbar=False,
    progressbar_title="Feature Extraction"
)

features = extract_features(
    df,
    column_id='id',
    column_sort='time',
    distributor=Distributor
)
```

### 5.2 Dask 分布式计算

```python
from tsfresh import extract_features
from tsfresh.utilities.distribution import ClusterDaskDistributor
from dask.distributed import Client

# 启动Dask集群
client = Client()

# 使用Dask分布式计算
Distributor = ClusterDaskDistributor(address=client.scheduler.address)

features = extract_features(
    df,
    column_id='id',
    column_sort='time',
    distributor=Distributor
)
```

### 5.3 滚动窗口特征提取

```python
from tsfresh.utilities.dataframe_functions import roll_time_series

# 创建滚动窗口
df_rolled = roll_time_series(
    df,
    column_id='id',
    column_sort='time',
    column_kind=None,
    rolling_direction=1,  # 向前滚动
    max_timeshift=3       # 最大时间窗口
)

# 提取滚动窗口特征
features_rolled = extract_features(
    df_rolled,
    column_id='id',
    column_sort='time'
)
```

### 5.4 与 sklearn 集成

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tsfresh.transformers import FeatureSelector, FeatureAugmenter

# 创建机器学习Pipeline
pipeline = Pipeline([
    ('augmenter', FeatureAugmenter(
        default_fc_parameters=MinimalFCParameters()
    )),
    ('selector', FeatureSelector()),
    ('classifier', RandomForestClassifier())
])

# 训练
pipeline.fit(df, y)

# 预测
predictions = pipeline.predict(df_test)
```

---

## 6. 实际应用案例

### 6.1 风控场景：用户行为特征提取

```python
import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

# 用户订单时间序列数据
orders = pd.DataFrame({
    'user_id': [1, 1, 1, 1, 2, 2, 2],
    'order_time': [0, 1, 3, 7, 0, 2, 5],
    'order_amount': [100, 150, 200, 120, 300, 250, 400],
    'has_overdue': [0, 0, 1, 0, 0, 0, 1]
})

# 提取用户订单金额的时序特征
amount_features = extract_features(
    orders[['user_id', 'order_time', 'order_amount']],
    column_id='user_id',
    column_sort='order_time'
)

# 提取逾期行为的时序特征
overdue_features = extract_features(
    orders[['user_id', 'order_time', 'has_overdue']],
    column_id='user_id',
    column_sort='order_time'
)

# 合并特征
user_features = pd.concat([amount_features, overdue_features], axis=1)

print(f"用户特征维度: {user_features.shape}")
```

### 6.2 金融场景：股票价格特征

```python
# 股票价格时间序列
stock_data = pd.DataFrame({
    'stock_id': ['AAPL'] * 100,
    'date': range(100),
    'close_price': np.random.randn(100).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 100)
})

# 提取价格和成交量特征
stock_features = extract_features(
    stock_data,
    column_id='stock_id',
    column_sort='date',
    default_fc_parameters=ComprehensiveFCParameters()
)

# 查看提取的特征
print(stock_features.columns[:10])
```

### 6.3 工业场景：设备传感器数据

```python
# 传感器时间序列数据
sensor_data = pd.DataFrame({
    'device_id': [1] * 1000 + [2] * 1000,
    'timestamp': list(range(1000)) * 2,
    'temperature': np.random.normal(25, 5, 2000),
    'vibration': np.random.normal(0.5, 0.1, 2000),
    'pressure': np.random.normal(100, 10, 2000)
})

# 提取设备健康度特征
device_features = extract_features(
    sensor_data,
    column_id='device_id',
    column_sort='timestamp',
    n_jobs=4  # 并行处理
)

# 异常检测
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.1)
anomalies = clf.fit_predict(impute(device_features))
```

---

## 7. 最佳实践

### 7.1 数据预处理

```python
# 1. 检查数据质量
print(df.isnull().sum())
print(df.dtypes)

# 2. 确保时间列有序
df = df.sort_values(['id', 'time'])

# 3. 处理异常值
df = df[df['value'].between(df['value'].quantile(0.01), 
                             df['value'].quantile(0.99))]

# 4. 标准化时间间隔（如果需要）
df['time'] = pd.to_datetime(df['time'])
df['time'] = (df['time'] - df.groupby('id')['time'].transform('min')).dt.total_seconds()
```

### 7.2 性能优化

```python
# 1. 使用最小特征集进行快速实验
features = extract_features(
    df, 
    column_id='id',
    default_fc_parameters=MinimalFCParameters()
)

# 2. 分批处理大数据
def extract_features_in_batches(df, batch_size=1000):
    ids = df['id'].unique()
    features_list = []
    
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_df = df[df['id'].isin(batch_ids)]
        batch_features = extract_features(batch_df, column_id='id')
        features_list.append(batch_features)
    
    return pd.concat(features_list)

# 3. 启用多进程
features = extract_features(
    df,
    column_id='id',
    n_jobs=8,
    show_warnings=False
)
```

### 7.3 特征工程技巧

```python
# 1. 组合原始特征和tsfresh特征
original_features = df.groupby('id').agg({
    'value': ['count', 'sum']
}).reset_index()

tsfresh_features = extract_features(df, column_id='id')

combined_features = original_features.merge(
    tsfresh_features, 
    left_on='id', 
    right_index=True
)

# 2. 时间窗口特征
# 提取最近7天、30天、90天的特征
for window in [7, 30, 90]:
    df_window = df[df['time'] >= df['time'].max() - window]
    features_window = extract_features(
        df_window, 
        column_id='id'
    )
    features_window.columns = [f"{col}_last_{window}d" 
                               for col in features_window.columns]
```

---

## 8. 常见问题

### 8.1 特征提取速度慢？

**解决方案：**
- 使用 `MinimalFCParameters()` 减少特征数量
- 启用多进程 `n_jobs=-1`
- 使用Dask进行分布式计算
- 减少时间序列长度或采样

### 8.2 内存不足？

**解决方案：**
```python
# 分批处理
features = extract_features(
    df, 
    column_id='id',
    chunksize=500  # 每次处理500个时间序列
)

# 或使用Dask
import dask.dataframe as dd
ddf = dd.from_pandas(df, npartitions=10)
```

### 8.3 特征包含大量NaN？

**解决方案：**
```python
from tsfresh.utilities.dataframe_functions import impute

# 使用内置的填充方法
features_imputed = impute(features)

# 或自定义填充策略
features.fillna(0, inplace=True)
features.fillna(features.median(), inplace=True)
```

### 8.4 如何选择合适的特征参数？

**建议：**
1. **快速原型**：使用 `MinimalFCParameters()`
2. **精细调优**：使用 `EfficientFCParameters()`
3. **全面探索**：使用 `ComprehensiveFCParameters()`
4. **自定义**：根据业务需求定义特定特征

### 8.5 与其他时序库的对比

| 库 | 优势 | 劣势 | 适用场景 |
|----|------|------|----------|
| **tsfresh** | 全自动、特征丰富 | 计算开销大 | 表格数据+时序特征 |
| **tslearn** | 时序分类/聚类 | 特征提取有限 | 时序模式识别 |
| **statsmodels** | 统计建模专业 | 手动特征工程 | 时序预测 |
| **prophet** | 预测准确 | 不适合特征提取 | 时序预测 |

---

## 9. 参考资源

### 9.1 官方文档
- 官网：https://tsfresh.readthedocs.io/
- GitHub：https://github.com/blue-yonder/tsfresh
- 论文：*tsfresh: A Python package for automatic extraction of relevant features from time series*

### 9.2 相关教程
- [官方示例集](https://tsfresh.readthedocs.io/en/latest/text/quick_start.html)
- [API文档](https://tsfresh.readthedocs.io/en/latest/api/tsfresh.html)
- [特征计算器列表](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)

### 9.3 实战案例
- 金融风控：客户行为序列分析
- 工业制造：设备故障预测
- 医疗健康：生理信号分析
- 零售电商：用户购买模式挖掘

---

## 10. 总结

### 10.1 核心价值

tsfresh 的核心价值在于：
1. **自动化**：无需人工设计特征
2. **全面性**：800+ 种特征覆盖各个维度
3. **科学性**：基于统计学的特征选择
4. **实用性**：与主流ML库无缝集成

### 10.2 使用建议

- ✅ **适合使用**：有大量时间序列数据，需要快速探索特征
- ✅ **适合使用**：时序数据维度高，人工特征工程困难
- ❌ **不适合**：数据量小，简单统计特征即可满足
- ❌ **不适合**：对计算资源和时间要求严格

### 10.3 下一步学习

1. 实践官方教程案例
2. 在实际项目中应用 tsfresh
3. 学习如何调优特征提取参数
4. 探索与深度学习的结合

---

**文档版本：** v1.0  
**更新日期：** 2025-01-01  
**维护者：** AI Assistant
