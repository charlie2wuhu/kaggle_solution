#!/bin/bash

# 超参数优化运行脚本
echo "🎯 Titanic 超参数优化工具"
echo "=========================="

# 检查conda环境
if [[ "$CONDA_DEFAULT_ENV" != "kaggle" ]]; then
    echo "⚠️  请先激活conda环境: conda activate kaggle"
    exit 1
fi

# 创建模型输出目录
mkdir -p ../../models/titanic

# 函数：运行单个模型的超参数优化
optimize_model() {
    local model=$1
    local trials=${2:-50}
    
    echo ""
    echo "🚀 开始优化 $model 模型 (试验次数: $trials)"
    echo "----------------------------------------"
    
    python hyperopt.py --model $model --trials $trials --train
    
    if [ $? -eq 0 ]; then
        echo "✅ $model 优化完成"
    else
        echo "❌ $model 优化失败"
    fi
}

# 函数：批量优化所有模型
optimize_all() {
    local trials=${1:-50}
    
    echo "🔥 批量优化所有模型 (每个模型 $trials 次试验)"
    echo "============================================"
    
    models=("rf" "xgb" "lgbm" "decision_tree_gini")
    
    for model in "${models[@]}"; do
        optimize_model $model $trials
        echo ""
    done
    
    echo "🎉 所有模型优化完成!"
}

# 函数：显示帮助信息
show_help() {
    echo "使用方法:"
    echo "  $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --model MODEL     优化指定模型 (rf, xgb, lgbm, cat, decision_tree_gini)"
    echo "  --trials N        试验次数 (默认: 50)"
    echo "  --all             优化所有模型"
    echo "  --quick           快速测试 (每个模型10次试验)"
    echo "  --help            显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --model rf --trials 100    # 优化随机森林，100次试验"
    echo "  $0 --all --trials 50          # 优化所有模型，每个50次试验"
    echo "  $0 --quick                    # 快速测试所有模型"
}

# 解析命令行参数
MODEL=""
TRIALS=50
ALL=false
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --all)
            ALL=true
            shift
            ;;
        --quick)
            QUICK=true
            TRIALS=10
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "❌ 未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 执行相应的操作
if [ "$QUICK" = true ]; then
    echo "⚡ 快速测试模式"
    optimize_all $TRIALS
elif [ "$ALL" = true ]; then
    optimize_all $TRIALS
elif [ ! -z "$MODEL" ]; then
    # 验证模型名称
    if [[ ! "$MODEL" =~ ^(rf|xgb|lgbm|cat|decision_tree_gini)$ ]]; then
        echo "❌ 无效的模型名称: $MODEL"
        echo "支持的模型: rf, xgb, lgbm, cat, decision_tree_gini"
        exit 1
    fi
    optimize_model $MODEL $TRIALS
else
    echo "❌ 请指定要优化的模型或使用 --all 优化所有模型"
    show_help
    exit 1
fi 