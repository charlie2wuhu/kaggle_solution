"""
泰坦尼克号数据预处理模块
特征工程过程统一，特征选择根据配置进行
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import re
import config


def extract_title(name):
    """从姓名中提取称谓"""
    title = re.search(' ([A-Za-z]+)\.', name)
    if title:
        return title.group(1)
    return 'Unknown'


def group_title(title):
    """将Title分组为有意义的类别"""
    if title in ['Mr']:
        return 'Mr'
    elif title in ['Miss', 'Mlle', 'Ms']:
        return 'Miss'  
    elif title in ['Mrs', 'Mme']:
        return 'Mrs'
    elif title in ['Master']:
        return 'Master'  # 小男孩
    elif title in ['Dr', 'Rev', 'Col', 'Major', 'Capt']:
        return 'Officer'  # 专业人士/军官
    elif title in ['Sir', 'Lady', 'Countess', 'Don', 'Dona', 'Jonkheer']:
        return 'Noble'  # 贵族
    else:
        return 'Rare'  # 罕见称谓


def create_age_group(age):
    """基于数据洞察的年龄分组"""
    if pd.isna(age):
        return 'Unknown'
    elif age <= 16:
        return 'Child'      # 儿童：受保护群体
    elif age <= 32:
        return 'Young'      # 青年：主要劳动力
    elif age <= 50:
        return 'Middle'     # 中年：家庭责任重
    else:
        return 'Senior'     # 老年：行动不便


def family_type(size):
    """家庭类型分组"""
    if size == 1:
        return 'Alone'
    elif size <= 4:
        return 'Small'
    else:
        return 'Large'


def fare_group(fare, pclass_fares):
    """票价分组（基于同等级的分位数）"""
    if pd.isna(fare):
        return 'Unknown'
    
    q33 = pclass_fares.quantile(0.33)
    q67 = pclass_fares.quantile(0.67)
    
    if fare <= q33:
        return 'Low'
    elif fare <= q67:
        return 'Mid'
    else:
        return 'High'


class TitanicPreprocessor:
    """泰坦尼克号数据预处理器 - 统一的特征工程"""
    
    def __init__(self):
        self.train_data = None
        self.embarked_mode = None
        self.fare_median = None
        self.age_medians = {}  # 存储不同Title的年龄中位数
        self.class_fare_stats = {}  # 存储各等级的票价统计
        
    def fit(self, train_data):
        """在训练数据上拟合预处理器"""
        df = train_data.copy()
        
        # 先进行基础特征工程以便后续计算统计量
        df['Title'] = df['Name'].apply(extract_title)
        df['TitleGroup'] = df['Title'].apply(group_title)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # 保存训练数据用于统计
        self.train_data = df
        
        # 计算并保存统计量
        self.embarked_mode = df['Embarked'].mode()[0] if not df['Embarked'].mode().empty else 'S'
        self.fare_median = df['Fare'].median()
        
        # 计算各Title组的年龄中位数
        for title in df['TitleGroup'].unique():
            title_ages = df[df['TitleGroup'] == title]['Age'].dropna()
            if len(title_ages) > 0:
                self.age_medians[title] = title_ages.median()
            else:
                self.age_medians[title] = df['Age'].median()
        
        # 计算各等级的票价统计
        for pclass in [1, 2, 3]:
            pclass_fares = df[df['Pclass'] == pclass]['Fare'].dropna()
            if len(pclass_fares) > 0:
                self.class_fare_stats[pclass] = {
                    'mean': pclass_fares.mean(),
                    'fares': pclass_fares
                }
            else:
                self.class_fare_stats[pclass] = {
                    'mean': self.fare_median,
                    'fares': pd.Series([self.fare_median])
                }
        
        return self
    
    def transform(self, data, is_training=True):
        """应用完整的特征工程"""
        df = data.copy()
        
        print(f"🔧 开始特征工程处理...")
        print(f"📊 输入数据形状: {df.shape}")
        
        # 1. Title特征提取与分组
        df['Title'] = df['Name'].apply(extract_title)
        df['TitleGroup'] = df['Title'].apply(group_title)
        
        # 2. 智能Age填充
        def fill_age(row):
            if pd.isna(row['Age']):
                title_group = row['TitleGroup']
                if title_group in self.age_medians:
                    return self.age_medians[title_group]
                else:
                    # 如果是新的title，使用总体中位数
                    return self.train_data['Age'].median()
            return row['Age']
        
        df['FilledAge'] = df.apply(fill_age, axis=1)
        
        # 3. 年龄分组（原始年龄和填充后年龄）
        df['AgeGroup'] = df['Age'].apply(create_age_group)
        df['FilledAgeGroup'] = df['FilledAge'].apply(create_age_group)
        
        # 4. Cabin特征工程
        df['CabinLetter'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else np.nan)
        df['HasCabin'] = df['Cabin'].notna().astype(int)
        
        # 5. 家庭特征工程
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['FamilySizeGroup'] = df['FamilySize'].apply(family_type)
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # 6. 票价特征工程
        # 填充缺失的票价
        df['Fare'] = df['Fare'].fillna(self.fare_median)
        
        # 计算相对票价
        def calc_fare_per_class(row):
            pclass = row['Pclass']
            fare = row['Fare']
            class_mean = self.class_fare_stats[pclass]['mean']
            return fare / class_mean if class_mean > 0 else 1.0
        
        df['FarePerClass'] = df.apply(calc_fare_per_class, axis=1)
        
        # 票价分组
        def assign_fare_group(row):
            pclass = row['Pclass']
            fare = row['Fare']
            pclass_fares = self.class_fare_stats[pclass]['fares']
            return fare_group(fare, pclass_fares)
        
        df['FareGroup'] = df.apply(assign_fare_group, axis=1)
        
        # 7. Embarked缺失值处理
        df['FilledEmbarked'] = df['Embarked'].fillna(self.embarked_mode)
        
        # 8. 保留原始特征（用于baseline配置）
        # Age, Fare, Embarked 已经在上面处理了缺失值
        
        print(f"✅ 特征工程完成，输出数据形状: {df.shape}")
        
        # 显示生成的新特征
        new_features = [
            'Title', 'TitleGroup', 'FilledAge', 'AgeGroup', 'FilledAgeGroup',
            'CabinLetter', 'HasCabin', 'FamilySize', 'FamilySizeGroup', 'IsAlone',
            'FarePerClass', 'FareGroup', 'FilledEmbarked'
        ]
        print(f"📝 生成的新特征: {new_features}")
        
        return df
    
    def fit_transform(self, data):
        """拟合并转换训练数据"""
        return self.fit(data).transform(data, is_training=True)


def create_preprocessing_pipeline(feature_config='baseline'):
    """根据特征配置创建预处理pipeline"""
    
    # 从config获取特征列表
    if feature_config not in config.FEATURE_CONFIG:
        raise ValueError(f"❌ 未知的特征配置: {feature_config}")
    
    features = config.FEATURE_CONFIG[feature_config]
    
    print(f"🎯 使用特征配置: {feature_config}")
    print(f"📝 选择的特征: {features}")
    
    # 确定分类特征和数值特征
    categorical_features = []
    numerical_features = []
    
    # 预定义的特征类型
    known_categorical = {
        'Sex', 'Embarked', 'FilledEmbarked', 'TitleGroup', 'AgeGroup', 
        'FilledAgeGroup', 'CabinLetter', 'FamilySizeGroup', 'FareGroup'
    }
    
    known_numerical = {
        'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FilledAge', 
        'FamilySize', 'HasCabin', 'IsAlone', 'FarePerClass'
    }
    
    for feature in features:
        if feature in known_categorical:
            categorical_features.append(feature)
        elif feature in known_numerical:
            numerical_features.append(feature)
        else:
            print(f"⚠️ 未知特征类型: {feature}，默认作为数值特征处理")
            numerical_features.append(feature)
    
    print(f"📊 分类特征: {categorical_features}")
    print(f"🔢 数值特征: {numerical_features}")
    
    # 构建预处理器
    transformers = []
    
    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        )
    
    if numerical_features:
        transformers.append(
            ("num", "passthrough", numerical_features)
        )
    
    if not transformers:
        raise ValueError(f"❌ 没有有效的特征可以使用")
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # 删除未指定的特征
    )
    
    return preprocessor, features


def create_full_pipeline(model, feature_config='baseline'):
    """创建完整的预处理+模型pipeline"""
    
    if feature_config == 'baseline':
        # baseline配置使用简单的预处理（不需要特征工程）
        print("⚙️ 使用baseline配置，仅进行简单预处理")
        
        features = config.FEATURE_CONFIG['baseline']
        
        # 简单的预处理：只处理分类变量
        categorical_features = ['Sex', 'Embarked']
        numerical_features = ['Pclass', 'Age', 'Fare']
        
        # 确保特征存在于列表中
        categorical_features = [f for f in categorical_features if f in features]
        numerical_features = [f for f in numerical_features if f in features]
        
        transformers = []
        if categorical_features:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
            )
        if numerical_features:
            transformers.append(
                ("num", "passthrough", numerical_features)
            )
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
    else:
        # 其他配置使用特征工程后的预处理
        preprocessor, features = create_preprocessing_pipeline(feature_config)
    
    # 构建完整pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )
    
    return pipeline, features


# 向后兼容函数
def get_baseline_features():
    """获取baseline特征"""
    return config.FEATURE_CONFIG['baseline']


def get_engineered_features(feature_config='recommended'):
    """获取工程化特征"""
    return config.FEATURE_CONFIG.get(feature_config, config.FEATURE_CONFIG['recommended']) 