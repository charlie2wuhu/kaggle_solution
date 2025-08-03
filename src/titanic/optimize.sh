#!/bin/bash

# 超参数优化运行脚本
echo "🎯 Titanic 超参数优化工具"
echo "=========================="

# 检查conda环境（除非是显示帮助）
if [[ "$1" != "--help" && "$1" != "-h" && "$CONDA_DEFAULT_ENV" != "kaggle" ]]; then
    echo "⚠️  请先激活conda环境: conda activate kaggle"
    exit 1
fi

# 创建模型输出目录
mkdir -p ../../models/titanic

# 函数：运行单个模型的超参数优化
optimize_model() {
    local model=$1
    local trials=${2:-50}
    local features=${3:-recommended}
    
    echo ""
    echo "🚀 开始优化 $model 模型 (试验次数: $trials, 特征配置: $features)"
    echo "----------------------------------------"
    
    # 捕获输出并解析最佳得分
    local output
    output=$(python hyperopt.py --model $model --trials $trials --features $features --train 2>&1)
    local exit_code=$?
    
    echo "$output"
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $model 优化完成"
        
        # 解析最佳得分和平均准确率
        local best_score=$(echo "$output" | grep "最佳得分:" | tail -1 | sed 's/.*最佳得分: \([0-9.]*\).*/\1/')
        local avg_accuracy=$(echo "$output" | grep "平均准确率:" | tail -1 | sed 's/.*平均准确率: \([0-9.]*\).*/\1/')
        
        # 保存结果到临时文件
        if [ ! -z "$best_score" ]; then
            echo "$model,$best_score,$avg_accuracy" >> /tmp/titanic_results.csv
        else
            echo "$model,FAILED,FAILED" >> /tmp/titanic_results.csv
        fi
        
        return 0
    else
        echo "❌ $model 优化失败"
        echo "$model,FAILED,FAILED" >> /tmp/titanic_results.csv
        return 1
    fi
}

# 函数：批量优化所有模型
optimize_all() {
    local trials=${1:-50}
    local features=${2:-recommended}
    
    echo "🔥 批量优化所有模型 (每个模型 $trials 次试验, 特征配置: $features)"
    echo "============================================"
    
    # 初始化结果文件
    echo "Model,Best_Score,Avg_Accuracy" > /tmp/titanic_results.csv
    
    models=("rf" "decision_tree_gini")
    available_models=()
    
    # 检查可用模型（只包含肯定可用的模型）
    for model in "${models[@]}"; do
        available_models+=("$model")
    done
    
    # 检查可选模型
    if python -c "import xgboost" 2>/dev/null; then
        available_models+=("xgb")
    else
        echo "⚠️ XGBoost 未安装，跳过 xgb 模型"
    fi
    
    if python -c "import lightgbm" 2>/dev/null; then
        available_models+=("lgbm")
    else
        echo "⚠️ LightGBM 未安装，跳过 lgbm 模型"
    fi
    
    if python -c "import catboost" 2>/dev/null; then
        available_models+=("cat")
    else
        echo "⚠️ CatBoost 未安装，跳过 cat 模型"
    fi
    
    echo "📋 将要优化的模型: ${available_models[*]}"
    echo ""
    
    local start_time=$(date +%s)
    local successful_models=0
    local failed_models=0
    
    for model in "${available_models[@]}"; do
        if optimize_model $model $trials $features; then
            ((successful_models++))
        else
            ((failed_models++))
        fi
        echo ""
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "🎉 所有模型优化完成!"
    echo "⏱️  总耗时: ${duration}秒"
    echo "✅ 成功: $successful_models 个模型"
    echo "❌ 失败: $failed_models 个模型"
    echo ""
    
    # 显示结果对比表格
    show_results_comparison $features
}

# 函数：显示结果对比表格
show_results_comparison() {
    local features=${1:-recommended}
    
    if [ ! -f /tmp/titanic_results.csv ] || [ $(wc -l < /tmp/titanic_results.csv) -le 1 ]; then
        echo "⚠️ 没有找到有效的结果数据"
        return 1
    fi
    
    echo "📊 模型性能对比 (特征配置: $features)"
    echo "=========================================="
    
    # 表格头部
    printf "%-20s %-15s %-15s %-10s\n" "模型名称" "最佳CV得分" "平均准确率" "状态"
    echo "--------------------------------------------------------------"
    
    # 读取结果并排序（按最佳得分降序）
    local best_model=""
    local best_score=0
    local model_count=0
    local success_count=0
    
    # 跳过标题行并按分数排序
    tail -n +2 /tmp/titanic_results.csv | sort -t',' -k2 -nr | while IFS=',' read -r model score avg_acc; do
        if [ "$score" != "FAILED" ]; then
            local status="✅"
            local score_display=$(printf "%.4f" "$score")
            local avg_display=$(printf "%.4f" "$avg_acc")
        else
            local status="❌"
            local score_display="FAILED"
            local avg_display="FAILED"
        fi
        
        printf "%-20s %-15s %-15s %-10s\n" "$model" "$score_display" "$avg_display" "$status"
    done
    
    echo "--------------------------------------------------------------"
    
    # 找出最佳模型
    local best_line=$(tail -n +2 /tmp/titanic_results.csv | grep -v "FAILED" | sort -t',' -k2 -nr | head -1)
    if [ ! -z "$best_line" ]; then
        local best_model=$(echo "$best_line" | cut -d',' -f1)
        local best_score=$(echo "$best_line" | cut -d',' -f2)
        local best_avg=$(echo "$best_line" | cut -d',' -f3)
        
        echo ""
        echo "🏆 最佳模型: $best_model"
        echo "🎯 最佳CV得分: $(printf "%.4f" "$best_score")"
        echo "📈 平均准确率: $(printf "%.4f" "$best_avg")"
        
        # 计算提升百分比（相对于baseline假设值0.77）
        local baseline=0.77
        local improvement=$(echo "scale=4; ($best_score - $baseline) * 100 / $baseline" | bc -l 2>/dev/null || echo "N/A")
        if [ "$improvement" != "N/A" ]; then
            echo "📊 相对提升: +$(printf "%.2f" "$improvement")% (相对于0.77基准)"
        fi
    fi
    
    echo ""
    echo "💡 提示:"
    echo "   - 可以使用 'python train.py --model <best_model> --predict --features $features' 生成提交文件"
    echo "   - 最佳参数已保存到 models/ 目录中"
    
    # 清理临时文件
    rm -f /tmp/titanic_results.csv
}

# 函数：显示帮助信息
show_help() {
    echo "使用方法:"
    echo "  $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --model MODEL     优化指定模型 (rf, xgb, lgbm, cat, decision_tree_gini)"
    echo "  --trials N        试验次数 (默认: 50)"
    echo "  --features CONF   特征配置 (baseline, core, recommended, all) 默认: recommended"
    echo "  --all             优化所有可用模型"
    echo "  --quick           快速测试 (每个模型10次试验)"
    echo "  --help            显示此帮助信息"
    echo ""
    echo "特征配置说明:"
    echo "  baseline      使用原始4个特征 (Pclass, Sex, SibSp, Parch)"
    echo "  core          使用3个核心工程化特征 (TitleGroup, Pclass, HasCabin)"
    echo "  recommended   使用8个推荐特征 (默认，包含核心+重要特征)"
    echo "  all           使用所有9个工程化特征"
    echo ""
    echo "示例:"
    echo "  $0 --model rf --trials 100 --features recommended    # 优化随机森林，推荐特征"
    echo "  $0 --all --trials 50 --features core                # 用核心特征优化所有模型"
    echo "  $0 --quick --features baseline                      # 用baseline特征快速测试"
    echo "  $0 --all                                            # 用推荐特征优化所有模型"
}

# 解析命令行参数
MODEL=""
TRIALS=50
FEATURES="recommended"
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
        --features)
            FEATURES="$2"
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

# 验证特征配置参数
if [[ ! "$FEATURES" =~ ^(baseline|core|recommended|all)$ ]]; then
    echo "❌ 无效的特征配置: $FEATURES"
    echo "支持的特征配置: baseline, core, recommended, all"
    exit 1
fi

# 执行相应的操作
if [ "$QUICK" = true ]; then
    echo "⚡ 快速测试模式 (特征配置: $FEATURES)"
    optimize_all $TRIALS $FEATURES
elif [ "$ALL" = true ]; then
    optimize_all $TRIALS $FEATURES
elif [ ! -z "$MODEL" ]; then
    # 验证模型名称
    if [[ ! "$MODEL" =~ ^(rf|xgb|lgbm|cat|decision_tree_gini)$ ]]; then
        echo "❌ 无效的模型名称: $MODEL"
        echo "支持的模型: rf, xgb, lgbm, cat, decision_tree_gini"
        exit 1
    fi
    optimize_model $MODEL $TRIALS $FEATURES
else
    echo "❌ 请指定要优化的模型或使用 --all 优化所有模型"
    show_help
    exit 1
fi 