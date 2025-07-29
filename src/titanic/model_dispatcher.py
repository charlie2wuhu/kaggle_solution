from sklearn import ensemble
from sklearn import tree
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

models = {
    # 以gini系数度量的决策树
    "decision_tree_gini": tree.DecisionTreeClassifier(
        criterion="gini"
    ),
    # 以entropy系数度量的决策树
    "decision_tree_entropy": tree.DecisionTreeClassifier(
        criterion="entropy"
    ),
    "rf": ensemble.RandomForestClassifier(),
    "xgb": XGBClassifier(),
    "lgbm": LGBMClassifier(),
    "cat": CatBoostClassifier(verbose=False),
}