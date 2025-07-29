import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../../input/titanic/train.csv")

    # 我们创建一个名为 kfold 的新列，并用 -1 填充
    df["kfold"] = -1

    # 接下来的步骤是随机打乱数据的行
    df = df.sample(frac=1).reset_index(drop=True)

    # 从 model_selection 模块初始化 kfold 类
    kf = model_selection.KFold(n_splits=5)

    # 填充新的 kfold 列（enumerate的作用是返回一个迭代器）
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold

    # 保存带有 kfold 列的新 CSV 文件
    df.to_csv("../../input/titanic/train_folds.csv", index=False)