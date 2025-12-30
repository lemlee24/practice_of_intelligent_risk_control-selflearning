# -*- coding: utf-8 -*- 

import sys
sys.path.append("./")
sys.path.append("../")

# 设置matplotlib后端以兼容新版本
import matplotlib
matplotlib.use('Agg')

import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import bayes_opt as bo
import sklearn.model_selection as sk_ms
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from utils import data_utils
from chapter2.ch2_31_model_deployment_pickle import save_model_as_pkl


# 确定最优树的颗数
def xgb_cv(param, x, y, num_boost_round=10000):
    """
    使用交叉验证确定XGBoost模型的最优迭代次数
    
    参数说明:
    param: XGBoost模型参数字典
    x: 训练特征数据
    y: 训练标签数据
    num_boost_round: 最大 boosting 轮数，默认10000
    
    返回值:
    最优的 boosting 轮数，即交叉验证结果的行数
    """
    dtrain = xgb.DMatrix(x, label=y)
    # 执行交叉验证，early_stopping_rounds=30表示如果30轮内auc没有提升则停止训练
    cv_res = xgb.cv(param, dtrain, num_boost_round=num_boost_round, early_stopping_rounds=30)
    # 返回实际训练的轮数（即达到early stopping时的轮数）
    num_boost_round = cv_res.shape[0]
    return num_boost_round

def train_xgb(params, x_train, y_train, x_test=None, y_test=None, num_boost_round=10000, early_stopping_rounds=30, verbose_eval=50):
    """
    训练XGBoost模型
    
    参数说明:
    params: XGBoost模型参数字典
    x_train: 训练特征数据
    y_train: 训练标签数据
    x_test: 测试特征数据（可选），如果为None则使用交叉验证确定最优迭代次数
    y_test: 测试标签数据（可选），如果x_test为None则此参数也应为None
    num_boost_round: 最大boosting轮数，默认10000
    early_stopping_rounds: 早停轮数，如果连续early_stopping_rounds轮auc没有提升则停止训练，默认30
    verbose_eval: 每verbose_eval轮输出一次训练日志，默认50
    
    返回值:
    训练好的XGBoost模型
    """
    """
    训练xgb模型
    """
    dtrain = xgb.DMatrix(x_train, label=y_train)
    if x_test is None:
        num_boost_round = xgb_cv(params, x_train, y_train)
        early_stopping_rounds = None
        eval_sets = ()
    else:
        dtest = xgb.DMatrix(x_test, label=y_test)
        eval_sets = [(dtest, 'test')]
    model = xgb.train(params, dtrain, num_boost_round, evals=eval_sets, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)
    return model


def xgboost_grid_search(params_space, x_train, y_train, x_test=None, y_test=None, num_boost_round=10000):
    """
    网格调参, 确定其他参数
    
    参数说明:
    params_space: 参数搜索空间，字典格式，键为参数名，值为参数候选值列表
    x_train: 训练特征数据
    y_train: 训练标签数据
    x_test: 测试特征数据（可选），如果为None则从训练集中划分20%作为测试集
    y_test: 测试标签数据（可选），如果x_test为None则此参数也应为None
    num_boost_round: 最大boosting轮数，默认10000
    
    返回值:
    在测试集上表现最好的参数组合
    """
    # 如果没有提供测试集，则从训练集中划分20%作为测试集
    if x_test is None:
        x_train, x_test, y_train, y_test = sk_ms.train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    
    score_list = []  # 存储每个参数组合的评估分数
    test_params = list(ParameterGrid(params_space))  # 生成所有参数组合
    
    # 遍历所有参数组合
    for params_try in test_params:
        # 设置评估指标和随机种子
        params_try['eval_metric'] = "auc"
        params_try['random_state'] = 1
        
        # 训练模型
        clf_obj = train_xgb(params_try, x_train, y_train, x_test, y_test, num_boost_round=num_boost_round,
                            early_stopping_rounds=30, verbose_eval=0)
        
        # 预测并计算AUC分数
        y_pred = clf_obj.predict(xgb.DMatrix(x_test))
        auc_score = roc_auc_score(y_test, y_pred)
        score_list.append(auc_score)
    
    # 创建结果DataFrame并打印
    result_df = pd.DataFrame({
        'AUC_Score': score_list,
        'Parameters': test_params
    })
    print("网格搜索结果:")
    print(result_df)
    
    # 找到最佳参数组合
    # argmax()用于找到数组中最大值的索引位置，这里用来找到AUC分数最高的参数组合的索引
    best_params_idx = np.argmax(score_list)
    best_params = test_params[best_params_idx]
    
    print(f"最佳参数组合: {best_params}")
    print(f"最佳AUC分数: {score_list[best_params_idx]}")
    
    return best_params

'''
这个函数 `xgboost_bayesian_optimization` 的作用是使用贝叶斯优化来调参XGBoost模型。它包含一个内部函数 `xgboost_cv_for_bo`，用于评估不同参数组合的性能（通过AUC分数），然后使用贝叶斯优化算法找到最优参数。

让我重写这段代码，使其更清晰、更易维护：
'''

def xgboost_bayesian_optimization(params_space, x_train, y_train, x_test=None, y_test=None, num_boost_round=10000, nfold=5, init_points=2, n_iter=5, verbose_eval=0, early_stopping_rounds=30):
    """
    贝叶斯调参, 确定其他参数
    
    参数说明:
    params_space: 参数搜索空间，字典格式，键为参数名，值为参数的取值范围元组(min, max)
    x_train: 训练特征数据
    y_train: 训练标签数据
    x_test: 测试特征数据（可选），如果为None则使用交叉验证
    y_test: 测试标签数据（可选），如果x_test为None则此参数也应为None
    num_boost_round: 最大boosting轮数，默认10000
    nfold: 交叉验证折数，默认5
    init_points: 贝叶斯优化初始化点数，默认2
    n_iter: 贝叶斯优化迭代次数，默认5
    verbose_eval: 训练日志输出频率，默认0（不输出）
    early_stopping_rounds: 早停轮数，默认30
    
    返回值:
    最优参数组合
    """
    def evaluate_params(eta, gamma, max_depth, min_child_weight, subsample, colsample_bytree):
        """
        评估给定参数组合的性能
        
        参数说明:
        eta: 学习率
        gamma: 树分裂所需的最小损失减少值
        max_depth: 树的最大深度
        min_child_weight: 子节点中最小的样本权重和
        subsample: 训练样本的子采样比例
        colsample_bytree: 每棵树使用的特征子集比例
        
        返回值:
        AUC分数
        """
        # 构建XGBoost参数字典
        params = {
            'eval_metric': 'auc',
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eta': eta,
            'gamma': gamma,
            'max_depth': int(max_depth),
            'min_child_weight': int(min_child_weight),
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'seed': 1
        }
        
        # 根据是否提供测试集选择评估方式
        if x_test is None:
            # 使用交叉验证评估
            dtrain = xgb.DMatrix(x_train, label=y_train)
            cv_results = xgb.cv(
                params,
                dtrain,
                nfold=nfold,
                metrics='auc',
                early_stopping_rounds=early_stopping_rounds,
                num_boost_round=num_boost_round,
                verbose_eval=verbose_eval
            )
            # 返回交叉验证的最终AUC分数
            test_auc = cv_results['test-auc-mean'].iloc[-1]
        else:
            # 使用测试集评估
            clf_obj = train_xgb(
                params, 
                x_train, 
                y_train, 
                x_test, 
                y_test, 
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds, 
                verbose_eval=verbose_eval
            )
            # 计算测试集上的AUC分数
            test_auc = roc_auc_score(y_test, clf_obj.predict(xgb.DMatrix(x_test)))
        
        return test_auc

    # 使用贝叶斯优化寻找最优参数
    # bayes_opt.BayesianOptimization用于在给定参数空间内搜索最优参数组合
    bayes_optimizer = bo.BayesianOptimization(
        f=evaluate_params,
        pbounds=params_space,
        random_state=1
    )
    
    # 执行贝叶斯优化
    # init_points: 初始随机采样点数
    # n_iter: 后续贝叶斯优化迭代次数
    bayes_optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    # 获取最优参数
    best_params = bayes_optimizer.max['params']
    
    # 将需要整数的参数转换为整数类型
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    
    # 添加固定的XGBoost参数
    best_params['eval_metric'] = 'auc'
    best_params['booster'] = 'gbtree'
    best_params['objective'] = 'binary:logistic'
    best_params['seed'] = 1
    
    return best_params

    # 指定需要调节参数的取值范围
    xgb_bo_obj = bo.BayesianOptimization(xgboost_cv_for_bo, params_space, random_state=1)
    xgb_bo_obj.maximize(init_points=init_points, n_iter=n_iter)
    best_params = xgb_bo_obj.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    best_params['eval_metric'] = 'auc'
    best_params['booster'] = 'gbtree'
    best_params['objective'] = 'binary:logistic'
    best_params['seed'] = 1
    return best_params


# 导入数值型样例数据
train_x, test_x, train_y, test_y = data_utils.get_x_y_split(test_rate=0.2)

# 经验参数
exp_params = {
    'eval_metric': 'auc',
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eta': 0.1,
    'gamma': 0.01,
    'max_depth': 4,
    'min_child_weight': 1,
    'subsample': 1,
    'colsample_bytree': 1,
    'seed': 1
}
final_xgb_model = train_xgb(exp_params, train_x, train_y, test_x, test_y)
auc_score = roc_auc_score(test_y, final_xgb_model.predict(xgb.DMatrix(test_x)))
print("经验参数模型AUC: ", auc_score)

# 随机搜索调参
choose_tuner = 'bayesian'  # bayesian grid_search
if choose_tuner == 'grid_search':
    params_test = {
        'learning_rate': [0.1, 0.15],
        'gamma': [0.01, 0],
        'max_depth': [4, 3],
        'min_child_weight': [1, 2],
        'subsample': [0.95, 1],
        'colsample_bytree': [1]
    }
    optimal_params = xgboost_grid_search(params_test, train_x, train_y, test_x, test_y)
elif choose_tuner == 'bayesian':
    # 贝叶斯调参
    params_test = {'eta': (0.05, 0.2),
                   'gamma': (0.005, 0.05),
                   'max_depth': (3, 5),
                   'min_child_weight': (0, 3),
                   'subsample': (0.9, 1.0),
                   'colsample_bytree': (0.9, 1.0)}
    optimal_params = xgboost_bayesian_optimization(params_test, train_x, train_y, test_x, test_y, init_points=5, n_iter=8)

print("随机搜索调参最优参数: ", optimal_params)

final_xgb_model = train_xgb(optimal_params, train_x, train_y, test_x, test_y)
auc_score = roc_auc_score(test_y, final_xgb_model.predict(xgb.DMatrix(test_x)))
print("随机搜索调参模型AUC: ", auc_score)

# 保存模型
import os
data_dir = "./data/model" if os.path.exists("./data/model") else "../data/model"
os.makedirs(data_dir, exist_ok=True)
save_model_as_pkl(final_xgb_model, os.path.join(data_dir, "xgb_model.pkl"))

# SHAP计算
explainer = shap.TreeExplainer(final_xgb_model)
shap_values = explainer.shap_values(train_x)
# SHAP可视化并保存到文件
import matplotlib.pyplot as plt
shap.summary_plot(shap_values, train_x, max_display=5, show=False)
plt.savefig(os.path.join(data_dir, 'shap_summary_plot.png'), dpi=150, bbox_inches='tight')
print(f"SHAP可视化图已保存到: {os.path.join(data_dir, 'shap_summary_plot.png')}")
plt.close()
