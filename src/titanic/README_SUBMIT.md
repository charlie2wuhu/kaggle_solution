# Kaggle Titanic 提交工具使用说明

本目录提供了两个提交脚本，用于自动提交预测结果到Kaggle Titanic竞赛：

- `submit.sh` - Shell脚本版本（推荐）
- `submit.py` - Python脚本版本

## 前置条件

1. **安装Kaggle CLI**
   ```bash
   pip install kaggle
   ```

2. **配置Kaggle API**
   - 登录 [Kaggle](https://www.kaggle.com/)
   - 进入 Account 页面，点击 "Create New API Token"
   - 下载 `kaggle.json` 文件
   - 将文件放到 `~/.kaggle/kaggle.json`
   - 设置权限：`chmod 600 ~/.kaggle/kaggle.json`

## 使用方法

### Shell脚本版本（推荐）

#### 基本用法
```bash
# 提交最新的预测文件
./submit.sh "Random Forest with optimized parameters"

# 提交指定文件
./submit.sh -f output/submission_rf_20250729_215635.csv "RF model submission"

# 列出所有可用的提交文件
./submit.sh --list

# 显示帮助信息
./submit.sh --help
```

#### 参数说明
- `-f FILE` : 指定具体的提交文件路径
- `-d DIR` : 指定输出目录（默认：output）
- `--list` : 列出所有可用的提交文件
- `--help` : 显示帮助信息

### Python脚本版本

#### 基本用法
```bash
# 提交最新的预测文件
python submit.py -m "Random Forest with optimized parameters"

# 提交指定文件
python submit.py --file output/submission_rf_20250729_215635.csv -m "RF model submission"

# 列出所有可用的提交文件
python submit.py --list

# 显示帮助信息
python submit.py --help
```

#### 参数说明
- `-m, --message` : 提交信息（必需）
- `--file` : 指定具体的提交文件路径
- `--output-dir` : 指定输出目录（默认：output）
- `--list` : 列出所有可用的提交文件
- `--help` : 显示帮助信息

## 工作流程示例

### 1. 训练模型并生成预测
```bash
# 使用最佳参数训练随机森林模型
python train.py --model rf --predict
```

### 2. 查看生成的预测文件
```bash
./submit.sh --list
```

### 3. 提交到Kaggle
```bash
# 提交最新文件
./submit.sh "Random Forest with best hyperparameters from optuna optimization"

# 或提交指定文件
./submit.sh -f output/submission_rf_20250729_215635.csv "Final RF submission"
```

## 脚本特性

### 自动文件选择
- 如果不指定文件，脚本会自动选择最新修改的提交文件
- 支持按修改时间排序显示所有可用文件

### 错误处理
- 检查Kaggle CLI是否安装和配置
- 验证文件是否存在
- 提供详细的错误信息和解决建议

### 友好的输出
- 彩色emoji输出，易于阅读
- 显示文件信息和修改时间
- 提供Kaggle竞赛页面链接

## 常见问题

### Q: 提交失败，显示"403 Forbidden"
A: 请检查：
1. Kaggle API配置是否正确
2. 是否接受了竞赛规则
3. 竞赛是否还在进行中

### Q: 找不到提交文件
A: 请确保：
1. 已经运行了训练脚本生成预测：`python train.py --model rf --predict`
2. 文件确实在output目录下
3. 文件名以"submission_"开头

### Q: 网络连接问题
A: 请检查：
1. 网络连接是否正常
2. 是否需要代理设置
3. 防火墙是否阻止了连接

## 提交历史管理

脚本会自动选择最新的文件，但你也可以：

1. **查看所有提交文件**：
   ```bash
   ./submit.sh --list
   ```

2. **提交特定版本**：
   ```bash
   ./submit.sh -f output/submission_rf_20250729_215412.csv "Previous version for comparison"
   ```

3. **在Kaggle网站查看提交历史**：
   [https://www.kaggle.com/competitions/titanic/submissions](https://www.kaggle.com/competitions/titanic/submissions)