import joblib
import pandas as pd
import os
import glob
import config
import argparse
import model_dispatcher
import preprocessing
import datetime
from sklearn import metrics, ensemble
from sklearn import tree
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def load_best_params(model_name, feature_config='baseline'):
    """加载指定模型的最新最佳参数"""

    if feature_config == 'baseline':
        return None
    
    # 优先查找指定特征配置的参数文件
    pattern = os.path.join(config.MODEL_OUTPUT, f"{model_name}_best_params_{feature_config}_*.joblib")
    files = glob.glob(pattern)
    
    # 如果没有找到指定配置的文件，查找所有该模型的参数文件
    if not files:
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
    print(f"📊 最佳参数: {results['best_params']}")
    return results['best_params']

def create_model_with_best_params(model_name, best_params):
    """根据模型名称和最佳参数创建模型"""
    if best_params is None:
        # 如果没有最佳参数，使用默认模型
        return model_dispatcher.models[model_name]
    
    if model_name == "rf":
        return ensemble.RandomForestClassifier(**best_params, random_state=42)
    elif model_name == "xgb":
        return XGBClassifier(**best_params, random_state=42)
    elif model_name == "lgbm":
        return LGBMClassifier(**best_params, random_state=42, verbose=-1)
    elif model_name == "cat":
        return CatBoostClassifier(**best_params, random_state=42, verbose=False)
    elif model_name in ["decision_tree_gini", "decision_tree_entropy"]:
        criterion = "gini" if model_name == "decision_tree_gini" else "entropy"
        return tree.DecisionTreeClassifier(**best_params, criterion=criterion, random_state=42)
    else:
        # 如果模型不支持参数优化，使用默认模型
        return model_dispatcher.models[model_name]

def run_cv_fold(fold, model_name, feature_config='baseline'):
    """执行交叉验证中的单个fold"""

    best_params = load_best_params(model_name, feature_config)
    
    if feature_config not in config.FEATURE_CONFIG:
        raise ValueError(f"❌ 未知的特征配置: {feature_config}")
    
    features = config.FEATURE_CONFIG[feature_config]
    print(f"📝 使用特征: {features}")
    df = pd.read_csv(config.TRAINING_FILE)
    
    if feature_config == 'baseline':
        # baseline模式：不使用特征工程，直接使用原始数据
        print("⚙️ 使用baseline模式，不进行特征工程")
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        
        use_model = create_model_with_best_params(model_name, best_params)
        clf, _ = preprocessing.create_full_pipeline(use_model, feature_config)
        
    else:
        train_raw = pd.read_csv("../../input/titanic/train.csv")
        preprocessor_fe = preprocessing.TitanicPreprocessor()
        train_engineered = preprocessor_fe.fit_transform(train_raw)
        
        # 将kfold信息合并回去
        train_engineered['kfold'] = df['kfold']
        df = train_engineered
        
        # 分割训练集和验证集
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        
        use_model = create_model_with_best_params(model_name, best_params)
        clf, _ = preprocessing.create_full_pipeline(use_model, feature_config)
        clf = use_model
    
    # 准备数据
    x_train = df_train[features]
    y_train = df_train['Survived'].values
    x_valid = df_valid[features]
    y_valid = df_valid['Survived'].values
    
    print(f"🏋️ 训练样本数: {len(x_train)}")
    print(f"🧪 验证样本数: {len(x_valid)}")
    
    # 训练模型
    clf.fit(x_train, y_train)
    
    # 验证集预测
    preds = clf.predict(x_valid)
    accuracy = metrics.accuracy_score(y_valid, preds)
    
    print(f"✅ Fold={fold}, Accuracy={accuracy:.4f}")
    return accuracy

def train_full_dataset(model_name, feature_config='baseline'):
    """使用完整数据集训练并进行预测"""
    print(f"🚀 开始使用 {model_name} 模型进行训练和预测...")
    print(f"🎯 特征配置: {feature_config}")
    
    # 加载最佳参数
    best_params = load_best_params(model_name, feature_config)
    
    # 获取特征列表
    if feature_config not in config.FEATURE_CONFIG:
        raise ValueError(f"❌ 未知的特征配置: {feature_config}")
    
    features = config.FEATURE_CONFIG[feature_config]
    print(f"📝 使用特征: {features}")
    
    if feature_config == 'baseline':
        # baseline模式：不使用特征工程
        print("⚙️ 使用baseline模式，不进行特征工程")
        train_df = pd.read_csv(config.TRAINING_FILE)
        test_df = pd.read_csv(config.TEST_FILE)
        
        # 创建使用最佳参数的模型
        use_model = create_model_with_best_params(model_name, best_params)
        
        # 获取预处理pipeline
        clf, _ = preprocessing.create_full_pipeline(use_model, feature_config)
        
        # 准备数据
        X_train = train_df[features]
        y_train = train_df["Survived"].values
        X_test = test_df[features]
        
        # 用于创建提交文件的ID
        test_ids = test_df['PassengerId']
        
    else:
        # 其他模式：使用特征工程
        print("🔧 使用特征工程模式")
        
        # 读取原始数据
        train_raw = pd.read_csv("../../input/titanic/train.csv")
        test_raw = pd.read_csv("../../input/titanic/test.csv")
        
        print("🔧 开始对训练数据应用特征工程...")
        preprocessor_fe = preprocessing.TitanicPreprocessor()
        train_engineered = preprocessor_fe.fit_transform(train_raw)
        
        print("🔧 开始对测试数据应用特征工程...")
        test_engineered = preprocessor_fe.transform(test_raw, is_training=False)
        
        # 创建使用最佳参数的模型
        use_model = create_model_with_best_params(model_name, best_params)
        
        # 直接使用模型，因为特征工程已经完成
        clf = use_model
        
        # 准备数据
        X_train = train_engineered[features]
        y_train = train_engineered["Survived"].values
        X_test = test_engineered[features]
        
        # 用于创建提交文件的ID
        test_ids = test_raw['PassengerId']
    
    print(f"🏋️ 使用完整数据集训练模型...")
    print(f"📊 训练样本数量: {len(X_train)}")
    
    # 使用完整数据集训练
    clf.fit(X_train, y_train)
    
    print(f"🔮 对测试集进行预测...")
    print(f"📊 测试样本数量: {len(X_test)}")
    
    # 进行预测
    predictions = clf.predict(X_test)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions
    })
    
    # 确保output目录存在
    os.makedirs("../../output/titanic", exist_ok=True)
    
    # 保存预测结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{model_name}_{feature_config}_{timestamp}.csv"
    output_file = os.path.join("../../output/titanic", filename)
    submission.to_csv(output_file, index=False)
    
    print(f"💾 预测结果已保存到: {output_file}")
    print(f"📈 预测样本数量: {len(submission)}")
    print(f"🎯 存活预测数量: {predictions.sum()}")
    print(f"💀 死亡预测数量: {len(predictions) - predictions.sum()}")
    print(f"📊 存活率预测: {predictions.mean():.1%}")
    
    # 保存训练好的模型
    model_filename = f"{model_name}_final_{feature_config}_{timestamp}.joblib"
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, model_filename))
    print(f"🗃️ 最终模型已保存到: {model_filename}")
    
    return submission, clf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="要使用的模型")
    parser.add_argument("--cv", type=int, required=True, choices=[0, 1], 
                       help="训练模式: 1=交叉验证, 0=完整数据集训练并预测")
    parser.add_argument("--fold", type=int, help="交叉验证fold编号 (cv=1时必需)")
    parser.add_argument("--features", type=str, default="baseline", 
                       help="特征配置: baseline(原始特征，不使用特征工程)")
    args = parser.parse_args()
    
    print(f"🎯 模型: {args.model}")
    print(f"🎛️ 训练模式: {'交叉验证' if args.cv == 1 else '完整数据集训练'}")
    print(f"📝 特征配置: {args.features}")
    
    # 验证参数
    if args.cv == 1 and args.fold is None:
        print("❌ 交叉验证模式 (--cv 1) 需要指定 --fold 参数")
        exit(1)
    
    # 执行相应的训练模式
    if args.cv == 1:
        # 交叉验证模式
        print(f"📊 执行交叉验证，fold: {args.fold}")
        accuracy = run_cv_fold(fold=args.fold, model_name=args.model, feature_config=args.features)
        print(f"🎯 本次验证准确率: {accuracy:.4f}")
    else:
        # 完整数据集训练并预测
        print("🏆 使用完整数据集训练并生成预测")
        submission, model = train_full_dataset(model_name=args.model, feature_config=args.features)
        print("✅ 训练和预测完成！")