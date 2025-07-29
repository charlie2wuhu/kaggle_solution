import joblib
import pandas as pd
import os
import glob
import config
import argparse
import model_dispatcher
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import datetime

def load_best_params(model_name):
    """加载指定模型的最新最佳参数"""
    # 查找该模型的所有最佳参数文件
    pattern = os.path.join(config.MODEL_OUTPUT, f"{model_name}_best_params_*.joblib")
    files = glob.glob(pattern)
    
    if not files:
        print(f"⚠️ 没有找到模型 {model_name} 的最佳参数文件，使用默认参数")
        return None
    
    # 按时间戳排序，获取最新的文件
    latest_file = max(files, key=os.path.getctime)
    
    print(f"📁 加载最佳参数文件: {latest_file}")
    
    # 加载参数
    results = joblib.load(latest_file)
    
    print(f"🎯 最佳得分: {results['best_score']:.4f}")
    print(f"📊 最佳参数: {results['best_params']}")
    print(f"⏰ 时间戳: {results['timestamp']}")
    
    return results['best_params']

def create_model_with_best_params(model_name, best_params):
    """根据模型名称和最佳参数创建模型"""
    if best_params is None:
        # 如果没有最佳参数，使用默认模型
        return model_dispatcher.models[model_name]
    
    if model_name == "rf":
        from sklearn import ensemble
        return ensemble.RandomForestClassifier(**best_params, random_state=42)
    elif model_name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(**best_params, random_state=42)
    elif model_name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**best_params, random_state=42, verbose=-1)
    elif model_name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**best_params, random_state=42, verbose=False)
    elif model_name in ["decision_tree_gini", "decision_tree_entropy"]:
        from sklearn import tree
        criterion = "gini" if model_name == "decision_tree_gini" else "entropy"
        return tree.DecisionTreeClassifier(**best_params, criterion=criterion, random_state=42)
    else:
        # 如果模型不支持参数优化，使用默认模型
        return model_dispatcher.models[model_name]

def run(fold, model):
    # 读取数据文件
    df = pd.read_csv(config.TRAINING_FILE)
    # 选取df中kfold列不等于fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # 选取df中kfold列等于fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # 训练集输入，删除label列

    # 只使用部分特征进行训练和预测
    features = ["Pclass", "Sex", "SibSp", "Parch"]

    # 定义预处理，将Sex字段进行独热编码
    categorical_features = ["Sex"]
    numeric_features = ["Pclass", "SibSp", "Parch"]

    use_model = model_dispatcher.models[model]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    # 构建pipeline
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", use_model)
        ]
    )

    x_train = df_train[features]
    y_train = df_train.Survived.values
    x_valid = df_valid[features]
    y_valid = df_valid.Survived.values

    print(f"Training {use_model} for fold {fold}")
    # 使用pipeline训练
    clf.fit(x_train, y_train)
    # 使用验证集输入得到预测结果
    preds = clf.predict(x_valid)
    # 计算验证集准确率
    accuracy = metrics.accuracy_score(y_valid, preds)
    # 打印fold信息和准确率
    print(f"Fold={fold}, Accuracy={accuracy}")
    # 保存模型
    # 获取当前时间，格式为YYYYMMDD_HHMMSS
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 获取算法名称
    algo = model
    # 获取参数摘要（这里只做简单摘要，可以根据需要自定义）
    param_summary = ""
    if hasattr(use_model, "get_params"):
        params = use_model.get_params()
        # 取部分参数做摘要，比如n_estimators、max_depth等
        keys = [k for k in ["n_estimators", "max_depth", "criterion"] if k in params]
        param_summary = "_".join([f"{k}{params[k]}" for k in keys])
    # 拼接文件名
    filename = f"{algo}_{param_summary}_{now}_{fold}.joblib"
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, filename))

def train_and_predict(model_name):
    """使用最佳参数训练完整数据集并进行预测"""
    print(f"🚀 开始使用 {model_name} 模型进行训练和预测...")
    
    # 加载最佳参数
    best_params = load_best_params(model_name)
    
    # 读取训练数据
    train_df = pd.read_csv(config.TRAINING_FILE)
    
    # 只使用部分特征进行训练和预测
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    
    # 定义预处理，将Sex字段进行独热编码
    categorical_features = ["Sex"]
    numeric_features = ["Pclass", "SibSp", "Parch"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )
    
    # 创建使用最佳参数的模型
    use_model = create_model_with_best_params(model_name, best_params)
    
    # 构建pipeline
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", use_model)
        ]
    )
    
    # 准备训练数据
    X_train = train_df[features]
    y_train = train_df["Survived"].values
    
    print(f"🏋️ 使用完整数据集训练模型...")
    print(f"📊 训练样本数量: {len(X_train)}")
    
    # 使用完整数据集训练
    clf.fit(X_train, y_train)
    
    # 读取测试数据
    test_df = pd.read_csv(config.TEST_FILE)
    X_test = test_df[features]
    
    print(f"🔮 对测试集进行预测...")
    print(f"📊 测试样本数量: {len(X_test)}")
    
    # 进行预测
    predictions = clf.predict(X_test)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    
    # 确保output目录存在
    os.makedirs("output", exist_ok=True)
    
    # 保存预测结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{model_name}_{timestamp}.csv"
    output_file = os.path.join(config.OUTPUT_FILE, filename)
    submission.to_csv(output_file, index=False)
    
    print(f"💾 预测结果已保存到: {output_file}")
    print(f"📈 预测样本数量: {len(submission)}")
    print(f"🎯 存活预测数量: {predictions.sum()}")
    print(f"💀 死亡预测数量: {len(predictions) - predictions.sum()}")
    
    # 保存训练好的模型
    filename = f"{model_name}_final_model_{timestamp}.joblib"
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, filename))
    print(f"🗃️ 最终模型已保存到: {filename}")
    return submission, clf

if __name__ == "__main__":
    # 实例化参数环境
    parser = argparse.ArgumentParser()
    # fold参数
    parser.add_argument("--fold", type=int, help="交叉验证fold编号")
    # 模型参数
    parser.add_argument("--model", type=str, required=True, help="要使用的模型")
    # 新增预测模式参数
    parser.add_argument("--predict", action="store_true", help="使用最佳参数训练完整数据集并预测")
    # 读取参数
    args = parser.parse_args()
    
    if args.predict:
        # 预测模式：使用最佳参数训练完整数据集并预测
        train_and_predict(args.model)
    else:
        # 传统交叉验证模式
        if args.fold is None:
            print("❌ 交叉验证模式需要指定 --fold 参数")
            exit(1)
        run(fold=args.fold, model=args.model)