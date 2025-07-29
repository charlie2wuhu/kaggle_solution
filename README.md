# Kaggle Solution

## 🎯 项目结构

```
kaggle/
├── input/                      # 输入数据
│   └── titanic/
│       ├── train.csv          # 训练数据
│       ├── test.csv           # 测试数据
│       └── train_folds.csv    # 带有交叉验证fold的训练数据
├── models/                     # 保存的模型
│   └── titanic/
├── output/                     # 输出结果
│   └── titanic/
├── src/                        # 源代码
│   └── titanic/
│       ├── config.py          # 配置文件
│       ├── create_folds.py    # 创建交叉验证fold
│       ├── train.py           # 模型训练主脚本
│       ├── hyperopt.py        # 超参数优化
│       ├── model_dispatcher.py # 模型调度器
│       ├── submit.py          # Kaggle提交工具
│       ├── optimize.sh        # 优化脚本
│       └── run.sh             # 运行脚本
└── requirement.yml            # 依赖环境
```

## 🚀 快速开始

### 1. 环境设置

```bash
# 创建conda环境
conda env create -f requirement.yml
conda activate kaggle
```

### 2. 准备数据

```bash
cd src/titanic
python create_folds.py  # 创建交叉验证fold
```

### 3. 训练模型

#### 基础训练（交叉验证）
```bash
# 训练单个fold
python train.py --fold 0 --model rf

# 训练所有fold
./run.sh rf
```

#### 超参数优化
```bash
# 优化单个模型
python hyperopt.py --model rf --trials 100 --train

# 使用优化脚本
./optimize.sh --model rf --trials 100
./optimize.sh --all --trials 50  # 优化所有模型
```

#### 使用最佳参数训练并预测
```bash
python train.py --model rf --predict
```

### 4. 提交到Kaggle

#### 前置条件
1. 安装Kaggle CLI：`pip install kaggle`
2. 配置API Key：下载`kaggle.json`到`~/.kaggle/`目录

#### 提交最新预测结果
```bash
python submit.py -m "Random Forest with optimized parameters"
```

#### 更多提交选项
```bash
# 列出所有可用的提交文件
python submit.py --list

# 提交指定文件
python submit.py --file output/submission_rf_xxx.csv -m "Specific submission"

# 显示帮助
python submit.py --help
```

## 📊 支持的模型

- **rf** - Random Forest (随机森林)
- **xgb** - XGBoost (需要安装xgboost)
- **lgbm** - LightGBM (需要安装lightgbm)
- **cat** - CatBoost (需要安装catboost)
- **decision_tree_gini** - Decision Tree with Gini criterion
- **decision_tree_entropy** - Decision Tree with Entropy criterion

## 🔧 主要功能

### 超参数优化
使用Optuna进行贝叶斯优化，自动寻找最佳超参数：

```bash
# 优化随机森林，100次试验
python hyperopt.py --model rf --trials 100 --train

# 快速测试所有模型
./optimize.sh --quick
```

### 自动提交工具
自动选择最新生成的预测文件并提交到Kaggle：

- 自动文件选择（按修改时间）
- 完整错误处理和友好提示
- 支持列出所有可用文件
- 验证Kaggle CLI配置

### 模型管理
- 自动保存训练好的模型
- 保存最佳超参数配置
- 支持时间戳命名
- 交叉验证结果统计

## 📈 工作流程示例

```bash
# 1. 优化超参数
./optimize.sh --model rf --trials 100

# 2. 使用最佳参数训练并预测
python train.py --model rf --predict

# 3. 提交结果
python submit.py -m "RF n_estimators=250 max_depth=4 CV=0.8114"

# 4. 查看提交历史
python submit.py --list
```

## ⚙️ 配置说明

`src/titanic/config.py` 包含所有路径配置：

```python
TRAINING_FILE = "../../input/titanic/train_folds.csv"
TEST_FILE = "../../input/titanic/test.csv"
MODEL_OUTPUT = "../../models/titanic/"
OUTPUT_FILE = "../../output/titanic/"
```

## 🎯 提交工具详细说明

### 基本用法
```bash
# 提交最新预测文件
python submit.py -m "提交信息"

# 列出所有可用文件
python submit.py --list

# 提交指定文件
python submit.py --file path/to/file.csv -m "提交信息"
```

### 参数说明
- `-m, --message`：提交信息（必需）
- `--file`：指定具体的提交文件路径
- `--output-dir`：指定输出目录（默认使用config中的配置）
- `--list`：列出所有可用的提交文件

### 特性
- **自动文件选择**：如果不指定文件，自动选择最新修改的文件
- **路径配置**：使用`config.py`中定义的路径
- **错误处理**：检查Kaggle CLI、文件存在性等
- **友好输出**：显示文件信息、修改时间等

### 常见问题

**Q: 提交失败，显示"403 Forbidden"**
A: 请检查：
1. Kaggle API配置是否正确
2. 是否接受了竞赛规则
3. 竞赛是否还在进行中

**Q: 找不到提交文件**
A: 请确保：
1. 已运行训练脚本：`python train.py --model rf --predict`
2. 文件在正确的输出目录下
3. 文件名以"submission_"开头