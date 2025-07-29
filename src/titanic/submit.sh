#!/bin/bash

# Kaggle Titanic 提交工具
echo "🎯 Kaggle Titanic 提交工具"
echo "=========================="

# 函数：显示帮助信息
show_help() {
    echo "使用方法:"
    echo "  $0 [选项] \"提交信息\""
    echo ""
    echo "选项:"
    echo "  -f FILE       指定具体的提交文件"
    echo "  -d DIR        指定输出目录 (默认: output)"
    echo "  --list        列出所有可用的提交文件"
    echo "  --help        显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 \"Random Forest with best params\"     # 提交最新文件"
    echo "  $0 -f output/submission_rf_xxx.csv \"RF submission\"  # 提交指定文件"
    echo "  $0 --list                            # 列出所有文件"
}

# 函数：查找最新的提交文件
find_latest_file() {
    local output_dir=${1:-"output"}
    
    # 查找所有submission文件
    files=($(ls -t "$output_dir"/submission_*.csv 2>/dev/null))
    
    if [ ${#files[@]} -eq 0 ]; then
        echo "❌ 在 $output_dir 目录下没有找到任何submission文件"
        echo "📁 请先运行训练脚本生成预测文件:"
        echo "   python train.py --model rf --predict"
        return 1
    fi
    
    # 返回最新的文件
    latest_file="${files[0]}"
    echo "📁 找到最新提交文件: $(basename "$latest_file")"
    echo "⏰ 修改时间: $(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$latest_file" 2>/dev/null || date -r "$latest_file" "+%Y-%m-%d %H:%M:%S")"
    echo "$latest_file"
}

# 函数：列出所有提交文件
list_files() {
    local output_dir=${1:-"output"}
    
    files=($(ls -t "$output_dir"/submission_*.csv 2>/dev/null))
    
    if [ ${#files[@]} -eq 0 ]; then
        echo "❌ 没有找到任何提交文件"
        return 1
    fi
    
    echo "📁 在 $output_dir 目录下找到 ${#files[@]} 个提交文件:"
    
    for i in "${!files[@]}"; do
        file="${files[$i]}"
        mtime=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$file" 2>/dev/null || date -r "$file" "+%Y-%m-%d %H:%M:%S")
        echo "  $((i+1)). $(basename "$file") ($mtime)"
    done
}

# 函数：提交到Kaggle
submit_to_kaggle() {
    local file_path="$1"
    local message="$2"
    
    # 检查kaggle命令是否可用
    if ! command -v kaggle &> /dev/null; then
        echo "❌ Kaggle CLI 未安装或未配置"
        echo "📖 请参考以下步骤配置Kaggle CLI:"
        echo "   1. 安装: pip install kaggle"
        echo "   2. 配置API Key: https://www.kaggle.com/docs/api"
        return 1
    fi
    
    # 检查文件是否存在
    if [ ! -f "$file_path" ]; then
        echo "❌ 文件不存在: $file_path"
        return 1
    fi
    
    filename=$(basename "$file_path")
    
    echo "🚀 正在提交文件: $filename"
    echo "💬 提交信息: $message"
    echo "🔧 执行命令: kaggle competitions submit -c titanic -f \"$file_path\" -m \"$message\""
    echo "$(printf '%*s' 50 '' | tr ' ' '-')"
    
    # 执行提交命令
    if kaggle competitions submit -c titanic -f "$file_path" -m "$message"; then
        echo "✅ 提交成功!"
        echo "🌐 可以在以下网址查看提交结果:"
        echo "   https://www.kaggle.com/competitions/titanic/submissions"
        return 0
    else
        echo "❌ 提交失败!"
        echo "💡 可能的解决方案:"
        echo "   1. 检查网络连接"
        echo "   2. 验证Kaggle API配置"
        echo "   3. 确认竞赛是否还在进行中"
        return 1
    fi
}

# 解析命令行参数
OUTPUT_DIR="output"
SUBMIT_FILE=""
LIST_MODE=false
MESSAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            SUBMIT_FILE="$2"
            shift 2
            ;;
        -d|--dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --list)
            LIST_MODE=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        -*)
            echo "❌ 未知参数: $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$MESSAGE" ]; then
                MESSAGE="$1"
            else
                echo "❌ 多余的参数: $1"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# 如果是列表模式
if [ "$LIST_MODE" = true ]; then
    list_files "$OUTPUT_DIR"
    exit 0
fi

# 检查是否提供了提交信息
if [ -z "$MESSAGE" ]; then
    echo "❌ 请提供提交信息"
    show_help
    exit 1
fi

# 确定要提交的文件
if [ -n "$SUBMIT_FILE" ]; then
    if [ ! -f "$SUBMIT_FILE" ]; then
        echo "❌ 指定的文件不存在: $SUBMIT_FILE"
        exit 1
    fi
    echo "📁 使用指定文件: $(basename "$SUBMIT_FILE")"
    file_path="$SUBMIT_FILE"
else
    file_path=$(find_latest_file "$OUTPUT_DIR")
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

# 提交到Kaggle
submit_to_kaggle "$file_path" "$MESSAGE"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 提交完成!"
else
    echo ""
    echo "💔 提交失败，请检查错误信息并重试"
    exit 1
fi 