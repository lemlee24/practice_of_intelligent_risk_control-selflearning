# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

# Keras官方文档: https://keras.io
# Keras是TensorFlow的高级API，用于构建和训练深度学习模型

from utils import data_utils
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, models, callbacks

# ============================================================================
# 数据准备
# ============================================================================
# 加载德国信用数据集，并进行标准化处理
# transform_method='standard': 使用StandardScaler进行标准化，将特征缩放到均值为0、标准差为1
# 标准化对神经网络很重要，因为它可以加速梯度下降的收敛速度
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(transform_method='standard')

# ============================================================================
# 模型构建
# ============================================================================
# 设置随机数种子，确保结果可重现
# 在深度学习中，权重初始化、dropout等都涉及随机性
tf.random.set_seed(1)

# 设置早停回调函数
# monitor='val_loss': 监控验证集的损失值
# patience=30: 如果验证集损失连续30个epoch没有改善，则停止训练
# mode='min': 损失值越小越好
# 早停可以防止过拟合，在验证集性能不再提升时及时停止训练
callback = callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='min')

# 构建深度神经网络(DNN)模型
# Sequential模型是层的线性堆叠，适合大多数问题
# 使用Input层替代input_shape参数，这是Keras 3.x推荐的现代写法
model = models.Sequential([
    # 输入层：接收train_x.shape[1]个特征（即20个特征）
    layers.Input(shape=(train_x.shape[1],)),
    
    # 第一个隐藏层：32个神经元
    # activation=tf.nn.relu: 使用ReLU激活函数，f(x) = max(0, x)
    # ReLU可以解决梯度消失问题，是深度学习中最常用的激活函数
    layers.Dense(32, activation=tf.nn.relu),
    
    # Dropout层：随机丢弃30%的神经元
    # 防止过拟合的正则化技术，训练时随机将部分神经元输出设为0
    # seed=1: 设置随机种子保证可重现性
    layers.Dropout(0.3, seed=1),
    
    # 第二个隐藏层：16个神经元
    # 逐层降低神经元数量，形成漏斗形结构，有助于特征抽象
    layers.Dense(16, activation=tf.nn.relu),
    
    # 输出层：1个神经元
    # activation=tf.nn.sigmoid: sigmoid函数将输出压缩到(0,1)区间
    # 输出可以解释为样本属于正类（违约）的概率
    layers.Dense(1, activation=tf.nn.sigmoid)
])

# 显示模型的结构
# 包括每层的类型、输出形状和参数数量
model.summary()

# ============================================================================
# 模型编译
# ============================================================================
# 设置模型训练参数
# optimizer='SGD': 使用随机梯度下降优化器
#   - SGD是最基础的优化算法，通过梯度反向传播更新权重
# metrics=[tf.metrics.AUC()]: 评估指标为AUC（Area Under Curve）
#   - AUC是衡量二分类模型性能的重要指标，值越接近1越好
# loss='binary_crossentropy': 二分类交叉熵损失函数
#   - 适用于二分类问题，衡量预测概率分布与真实分布的差异
model.compile(optimizer='SGD',
              metrics=[tf.metrics.AUC()],
              loss='binary_crossentropy')

# ============================================================================
# 模型训练
# ============================================================================
# 训练模型
# train_x, train_y: 训练集特征和标签
# validation_data=(test_x, test_y): 验证集数据，用于监控训练过程中的泛化性能
# batch_size=16: 每次梯度更新使用16个样本
#   - 小batch_size可以提供更频繁的权重更新，但训练时间更长
# epochs=240: 最多训练240轮
#   - 实际训练可能因早停而提前结束
# callbacks=[callback]: 使用早停回调，防止过拟合
# verbose=2: 每个epoch打印一行训练进度
model.fit(train_x, train_y, 
          validation_data=(test_x, test_y), 
          batch_size=16, 
          epochs=240, 
          callbacks=[callback], 
          verbose=2)

# ============================================================================
# 模型评估
# ============================================================================
# 在训练集上评估模型性能
# model.predict()返回预测的概率值
# roc_auc_score计算ROC曲线下的面积（AUC值）
auc_score = roc_auc_score(train_y, model.predict(train_x))
print("训练集AUC:", auc_score)

# 在测试集上评估模型性能
# 测试集AUC更能反映模型的泛化能力
# 如果训练集AUC远高于测试集AUC，说明模型可能过拟合
auc_score = roc_auc_score(test_y, model.predict(test_x))
print("测试集AUC:", auc_score)
