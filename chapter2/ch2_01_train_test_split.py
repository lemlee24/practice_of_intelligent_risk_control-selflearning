# -*- coding: utf-8 -*- 

from pathlib import Path
import sys

# 保证无论从哪个工作目录运行，都能正确导入项目内的 utils
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(PROJECT_ROOT))

from utils import data_utils
from sklearn.model_selection import train_test_split


# 导入添加 month 列的数据
model_data = data_utils.get_data()
# 选取 OOT 样本
oot_set = model_data[model_data["month"] == "2020-05"]
# 划分训练集和测试集（仅在非 OOT 月份上切分）
train_valid_set = model_data[model_data["month"] != "2020-05"]
X = train_valid_set[data_utils.x_cols]
Y = train_valid_set[data_utils.label]
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, test_size=0.3, random_state=88
)
# 标记样本集合
model_data.loc[oot_set.index, "sample_set"] = "oot"
model_data.loc[X_train.index, "sample_set"] = "train"
model_data.loc[X_valid.index, "sample_set"] = "valid"
# 输出各样本集数量，便于确认切分是否正确
print(model_data["sample_set"].value_counts(dropna=False))

