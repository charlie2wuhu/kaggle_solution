import optuna
import pandas as pd
import numpy as np
import joblib
import os
import config
import datetime
import preprocessing
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn import ensemble, tree

# 安全导入可选依赖
def safe_import(module_name, class_name):
    """安全导入可选依赖"""
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        print(f"⚠️ 警告: {module_name} 未安装，{class_name} 模型不可用")
        return None

XGBClassifier = safe_import('xgboost', 'XGBClassifier')
LGBMClassifier = safe_import('lightgbm', 'LGBMClassifier') 
CatBoostClassifier = safe_import('catboost', 'CatBoostClassifier')

# 禁用optuna的日志输出
optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_model_with_params(model_name, trial):
    """根据模型名称和trial获取具有优化参数的模型"""
    
    if model_name == "rf":
        return ensemble.RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=42
        )
    
    elif model_name == "xgb":
        if XGBClassifier is None:
            raise ValueError("XGBoost未安装，无法使用xgb模型")
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=42
        )
    
    elif model_name == "lgbm":
        if LGBMClassifier is None:
            raise ValueError("LightGBM未安装，无法使用lgbm模型")
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=42,
            verbose=-1
        )
    
    elif model_name == "cat":
        if CatBoostClassifier is None:
            raise ValueError("CatBoost未安装，无法使用cat模型")
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 50, 300),
            depth=trial.suggest_int("depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 10),
            random_state=42,
            verbose=False
        )
    
    elif model_name == "decision_tree_gini":
        return tree.DecisionTreeClassifier(
            criterion="gini",
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=42
        )
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def prepare_data(feature_config='recommended'):
    """准备数据"""
    print(f"🔧 准备数据，特征配置: {feature_config}")
    
    # 读取训练数据（带fold信息）
    df = pd.read_csv(config.TRAINING_FILE)
    
    # 根据特征配置选择数据处理方式
    if feature_config == 'baseline':
        # 使用原始baseline特征
        features = ["Pclass", "Sex", "SibSp", "Parch"]
        
        # 定义预处理器
        categorical_features = ["Sex"]
        numeric_features = ["Pclass", "SibSp", "Parch"]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ("num", "passthrough", numeric_features),
            ]
        )
        
        X = df[features]
        
    else:
        # 使用工程化特征
        if 'TitleGroup' not in df.columns:
            print("📊 检测到原始数据，开始应用特征工程...")
            # 读取原始训练数据进行特征工程
            train_raw = pd.read_csv("../../input/titanic/train.csv")
            
            # 应用特征工程
            preprocessor_fe = preprocessing.TitanicPreprocessor()
            train_engineered = preprocessor_fe.fit_transform(train_raw)
            
            # 将kfold信息合并回去
            train_engineered['kfold'] = df['kfold']
            df = train_engineered
        
        # 获取预处理pipeline和特征列表
        preprocessor, features = preprocessing.create_preprocessing_pipeline(feature_config)
        X = df[features]
    
    y = df["Survived"].values
    folds = df["kfold"].values
    
    print(f"📝 使用特征: {features}")
    print(f"📊 数据形状: {X.shape}")
    
    return X, y, folds, preprocessor, features

def objective(trial, model_name, X, y, folds, preprocessor):
    """optuna的目标函数"""
    
    # 获取模型
    model = get_model_with_params(model_name, trial)
    
    # 构建pipeline
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # 手动进行交叉验证
    scores = []
    for fold in range(5):
        # 分割数据
        train_idx = folds != fold
        val_idx = folds == fold
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 训练和预测
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        
        # 计算准确率
        score = metrics.accuracy_score(y_val, y_pred)
        scores.append(score)
    
    return np.mean(scores)

def optimize_hyperparameters(model_name, n_trials=100, feature_config='recommended'):
    """优化超参数"""
    
    print(f"🚀 开始优化 {model_name} 的超参数...")
    print(f"🎯 特征配置: {feature_config}")
    
    # 准备数据
    X, y, folds, preprocessor, features = prepare_data(feature_config)
    
    # 创建study
    study = optuna.create_study(direction='maximize')
    
    # 优化
    study.optimize(
        lambda trial: objective(trial, model_name, X, y, folds, preprocessor),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # 输出结果
    print(f"✅ {model_name} 优化完成!")
    print(f"📊 最佳得分: {study.best_value:.4f}")
    print(f"🎯 最佳参数: {study.best_params}")
    
    # 保存结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'model_name': model_name,
        'feature_config': feature_config,
        'best_score': study.best_value,
        'best_params': study.best_params,
        'features_used': features,
        'timestamp': timestamp
    }
    
    # 确保目录存在
    os.makedirs(config.MODEL_OUTPUT, exist_ok=True)
    
    # 保存最佳参数
    joblib.dump(results, os.path.join(config.MODEL_OUTPUT, f"{model_name}_best_params_{feature_config}_{timestamp}.joblib"))
    
    return study.best_params, study.best_value

def train_best_model(model_name, best_params, feature_config='recommended'):
    """使用最佳参数训练所有fold的模型"""
    
    print(f"🏋️ 使用最佳参数训练 {model_name} 模型...")
    print(f"🎯 特征配置: {feature_config}")
    
    # 准备数据
    X, y, folds, preprocessor, features = prepare_data(feature_config)
    
    results = []
    
    for fold in range(5):
        print(f"  训练 Fold {fold}...")
        
        # 分割数据
        train_idx = folds != fold
        val_idx = folds == fold
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 创建模型
        if model_name == "rf":
            model = ensemble.RandomForestClassifier(**best_params, random_state=42)
        elif model_name == "xgb":
            if XGBClassifier is None:
                raise ValueError("XGBoost未安装")
            model = XGBClassifier(**best_params, random_state=42)
        elif model_name == "lgbm":
            if LGBMClassifier is None:
                raise ValueError("LightGBM未安装")
            model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
        elif model_name == "cat":
            if CatBoostClassifier is None:
                raise ValueError("CatBoost未安装")
            model = CatBoostClassifier(**best_params, random_state=42, verbose=False)
        elif model_name == "decision_tree_gini":
            model = tree.DecisionTreeClassifier(**best_params, criterion="gini", random_state=42)
        
        # 构建pipeline
        clf = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        # 训练
        clf.fit(X_train, y_train)
        
        # 预测
        y_pred = clf.predict(X_val)
        accuracy = metrics.accuracy_score(y_val, y_pred)
        
        print(f"    Fold {fold} 准确率: {accuracy:.4f}")
        results.append(accuracy)
        
        # 保存模型
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_optimized_{feature_config}_{timestamp}_fold{fold}.joblib"
        joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, filename))
    
    print(f"📈 平均准确率: {np.mean(results):.4f} ± {np.std(results):.4f}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="超参数优化")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["rf", "xgb", "lgbm", "cat", "decision_tree_gini"],
                       help="要优化的模型")
    parser.add_argument("--trials", type=int, default=100, 
                       help="优化试验次数")
    parser.add_argument("--train", action="store_true",
                       help="是否使用最佳参数训练模型")
    parser.add_argument("--features", type=str, default="baseline",
                       choices=["baseline", "core", "recommended", "all"],
                       help="特征配置: baseline(原始4特征), core(核心3特征), recommended(推荐8特征), all(全部特征)")
    
    args = parser.parse_args()
    
    print(f"🔍 正在优化模型: {args.model}")
    print(f"🎲 试验次数: {args.trials}")
    print(f"📝 特征配置: {args.features}")
    print("=" * 50)
    
    # 优化超参数
    best_params, best_score = optimize_hyperparameters(args.model, args.trials, args.features)
    
    if args.train:
        print("\n" + "=" * 50)
        # 使用最佳参数训练模型
        train_best_model(args.model, best_params, args.features)
    
    print("\n🎉 优化完成!") 