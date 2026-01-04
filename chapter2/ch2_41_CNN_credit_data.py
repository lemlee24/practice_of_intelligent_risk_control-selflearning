# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from utils import data_utils
from sklearn.metrics import roc_auc_score

# ============================================================================
# 数据准备
# ============================================================================
# 加载德国信用数据集并进行标准化处理
# transform_method='standard': 使用StandardScaler将特征标准化为均值0、标准差1
# 返回值: train_x, test_x为特征数据(DataFrame), train_y, test_y为标签数据(Series)
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(transform_method='standard')

# ============================================================================
# 数据预处理 - 转换为CNN所需的三维张量格式
# ============================================================================
# CNN(卷积神经网络)需要三维输入张量: (样本数, 序列长度, 特征通道数)
# 对于表格数据:
#   - 样本数: 每个样本是一条信用记录
#   - 序列长度: 将每个特征视为序列中的一个位置(即特征数量)
#   - 特征通道数: 每个位置只有1个通道(类似于灰度图像)
# 
# 例如: train_x.shape从(800, 20)转换为(800, 20, 1)
#       800个样本,每个样本有20个特征,每个特征1个通道
train_x = train_x.to_numpy().reshape((train_x.shape[0], train_x.shape[1], 1))
test_x = test_x.to_numpy().reshape((test_x.shape[0], test_x.shape[1], 1))
# 标签也需要reshape为列向量,便于后续处理
train_y = train_y.values.reshape((train_y.shape[0], 1))
test_y = test_y.values.reshape((test_y.shape[0], 1))

# ============================================================================
# 模型构建
# ============================================================================
# 设置随机数种子,确保模型训练结果可重现
# 影响权重初始化、dropout随机丢弃等随机过程
tf.random.set_seed(1)

# 配置早停回调函数,防止过拟合
# monitor='val_loss': 监控验证集的损失值
# patience=30: 如果验证集损失连续30个epoch没有改善则停止训练
# mode='min': 损失值越小越好,自动保存最佳模型
callback = callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='min')

# 构建一维卷积神经网络(1D CNN)模型
# CNN通过卷积操作提取局部特征模式,适合处理序列数据
# 使用Sequential模型按顺序堆叠各层,使用Input层替代input_shape参数(Keras 3.x推荐写法)
model = models.Sequential([
    # 输入层: 接收shape为(20, 1)的数据
    # 20表示序列长度(特征数),1表示通道数
    layers.Input(shape=(train_x.shape[1], 1)),
    
    # 第一个卷积层: 提取低层次特征模式
    # filters=16: 使用16个卷积核,每个卷积核学习不同的特征模式
    # kernel_size=4: 卷积窗口大小为4,一次观察4个连续特征
    # activation='relu': ReLU激活函数,f(x)=max(0,x),引入非线性,解决梯度消失
    # 输出shape: (None, 17, 16) - 17是卷积后的序列长度,16是特征图数量
    layers.Conv1D(filters=16, kernel_size=4, activation='relu'),
    
    # 第二个卷积层: 在第一层的基础上提取更高层次的特征
    # filters=8: 使用8个卷积核,逐步减少特征图数量
    # kernel_size=1: 1x1卷积,用于通道间的信息融合和降维
    # 类似于全连接层但保持空间结构
    # 输出shape: (None, 17, 8)
    layers.Conv1D(filters=8, kernel_size=1, activation='relu'),
    
    # 展平层: 将多维特征图展平为一维向量
    # 将(None, 17, 8)展平为(None, 136) - 17*8=136
    # 为后续全连接层做准备
    layers.Flatten(),
    
    # Dropout层: 正则化技术,防止过拟合
    # 0.3: 训练时随机丢弃30%的神经元(将输出置为0)
    # 强迫网络学习更鲁棒的特征,不依赖特定神经元
    # seed=1: 设置随机种子保证可重现性
    # 注意: 仅在训练时生效,预测时自动关闭
    layers.Dropout(0.3, seed=1),
    
    # 全连接层: 整合卷积层提取的特征
    # 16个神经元,学习特征间的复杂组合关系
    # activation='relu': ReLU激活函数
    layers.Dense(16, activation='relu'),
    
    # 输出层: 二分类问题的输出
    # 1个神经元输出预测结果
    # activation='sigmoid': Sigmoid函数将输出压缩到(0,1)区间
    #   - 输出可解释为样本属于正类(违约)的概率
    #   - sigmoid(x) = 1/(1+e^(-x))
    layers.Dense(1, activation='sigmoid')
])

# 显示模型的结构信息
# 包括每层的类型、输出形状、参数数量等
# 帮助理解模型复杂度和检查模型结构是否正确
model.summary()

# ============================================================================
# 模型编译 - 配置训练参数
# ============================================================================
# 设置优化器、评估指标和损失函数
# optimizer='SGD': 随机梯度下降优化器
#   - 最基础的优化算法,通过计算梯度并反向传播更新权重
#   - 每次用一个batch的数据计算梯度
# metrics=[tf.metrics.AUC()]: 训练过程中监控的评估指标
#   - AUC(Area Under ROC Curve): 衡量二分类模型性能的重要指标
#   - 值范围[0,1],越接近1表示模型性能越好
#   - AUC=0.5表示随机猜测,AUC=1.0表示完美分类
# loss='binary_crossentropy': 二分类交叉熵损失函数
#   - 衡量预测概率分布与真实标签分布的差异
#   - 公式: -[y*log(p) + (1-y)*log(1-p)]
#   - 其中y是真实标签(0或1),p是预测概率
model.compile(optimizer='SGD',
              metrics=[tf.metrics.AUC()],
              loss='binary_crossentropy')

# ============================================================================
# 模型训练
# ============================================================================
# 使用训练数据训练模型,并在验证集上评估性能
# 参数说明:
# - train_x, train_y: 训练集特征和标签,用于模型学习
# - validation_data=(test_x, test_y): 验证集数据
#     每个epoch结束后在验证集上评估,监控泛化性能
#     验证集不参与训练,只用于评估和早停判断
# - batch_size=16: 小批量梯度下降的batch大小
#     每次使用16个样本计算梯度并更新权重
#     较小的batch_size提供更频繁的权重更新,但训练时间更长
#     较大的batch_size训练更稳定,但可能陷入局部最优
# - epochs=240: 最多训练240轮(遍历整个训练集240次)
#     实际训练可能因早停而提前结束
# - callbacks=[callback]: 使用早停回调函数
#     在验证集性能不再提升时自动停止训练,节省时间并防止过拟合
# - verbose=2: 训练日志详细程度
#     2表示每个epoch打印一行简洁的进度信息
model.fit(train_x, train_y, 
          validation_data=(test_x, test_y), 
          batch_size=16, 
          epochs=240, 
          callbacks=[callback], 
          verbose=2)

# ============================================================================
# 模型评估 - 计算AUC指标
# ============================================================================
# 在训练集上评估模型性能
# model.predict(train_x): 对训练集进行预测,返回每个样本属于正类的概率
# roc_auc_score(): 计算ROC曲线下的面积(AUC值)
#   - 参数1: 真实标签
#   - 参数2: 预测概率
# 训练集AUC通常会比测试集高,因为模型在训练集上学习过
auc_score = roc_auc_score(train_y, model.predict(train_x))
print("训练集AUC:", auc_score)

# 在测试集上评估模型性能
# 测试集AUC更能反映模型的真实泛化能力
# 如果训练集AUC明显高于测试集AUC,说明模型可能过拟合
# 理想情况下两者应该比较接近
auc_score = roc_auc_score(test_y, model.predict(test_x))
print("测试集AUC:", auc_score)
