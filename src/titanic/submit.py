#!/usr/bin/env python3
import os
import glob
import argparse
import subprocess
import sys
from datetime import datetime
import config

def find_latest_submission_file(output_dir=None):
    """查找最新的提交文件"""
    
    # 使用config中定义的输出目录
    if output_dir is None:
        output_dir = config.OUTPUT_FILE
    
    # 确保目录存在
    if not os.path.exists(output_dir):
        print(f"❌ 输出目录不存在: {output_dir}")
        print("📁 请先运行训练脚本生成预测文件:")
        print("   python train.py --model rf --predict")
        return None
    
    # 查找所有submission文件
    pattern = os.path.join(output_dir, "submission_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ 在{output_dir}目录下没有找到任何submission文件")
        print("📁 请先运行训练脚本生成预测文件:")
        print("   python train.py --model rf --predict")
        return None
    
    # 按修改时间排序，获取最新的文件
    latest_file = max(files, key=os.path.getmtime)
    
    # 获取文件修改时间
    mtime = os.path.getmtime(latest_file)
    mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"📁 找到最新提交文件: {os.path.basename(latest_file)}")
    print(f"⏰ 修改时间: {mtime_str}")
    
    return latest_file

def submit_to_kaggle(file_path, message):
    """提交文件到Kaggle"""
    
    # 检查kaggle命令是否可用
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Kaggle CLI 未安装或未配置")
        print("📖 请参考以下步骤配置Kaggle CLI:")
        print("   1. 安装: pip install kaggle")
        print("   2. 配置API Key: https://www.kaggle.com/docs/api")
        return False
    
    # 构建提交命令
    filename = os.path.basename(file_path)
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", "titanic",
        "-f", file_path,
        "-m", message
    ]
    
    print(f"🚀 正在提交文件: {filename}")
    print(f"💬 提交信息: {message}")
    print(f"🔧 执行命令: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        # 执行提交命令
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("✅ 提交成功!")
        print("📊 Kaggle响应:")
        print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ 提交失败!")
        print("🔍 错误信息:")
        print(e.stderr)
        print("💡 可能的解决方案:")
        print("   1. 检查网络连接")
        print("   2. 验证Kaggle API配置")
        print("   3. 确认竞赛是否还在进行中")
        return False

def main():
    parser = argparse.ArgumentParser(description="Kaggle Titanic 提交工具")
    parser.add_argument(
        "-m", "--message", 
        type=str, 
        help="提交信息"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"输出文件目录 (默认: {config.OUTPUT_FILE})"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="指定具体的提交文件路径 (如果不指定，则自动选择最新文件)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的提交文件"
    )
    
    args = parser.parse_args()
    
    print("🎯 Kaggle Titanic 提交工具")
    print("=" * 40)
    
    # 如果只是列出文件
    if args.list:
        output_dir = args.output_dir if args.output_dir else config.OUTPUT_FILE
        pattern = os.path.join(output_dir, "submission_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            print("❌ 没有找到任何提交文件")
            return
        
        print(f"📁 在 {output_dir} 目录下找到 {len(files)} 个提交文件:")
        
        # 按修改时间排序
        files.sort(key=os.path.getmtime, reverse=True)
        
        for i, file_path in enumerate(files, 1):
            mtime = os.path.getmtime(file_path)
            mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            filename = os.path.basename(file_path)
            print(f"  {i}. {filename} ({mtime_str})")
        
        return
    
    # 检查是否提供了提交信息
    if not args.message:
        print("❌ 请提供提交信息 (-m)")
        sys.exit(1)
    
    # 确定要提交的文件
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ 指定的文件不存在: {args.file}")
            sys.exit(1)
        file_path = args.file
        print(f"📁 使用指定文件: {os.path.basename(file_path)}")
    else:
        file_path = find_latest_submission_file(args.output_dir)
        if not file_path:
            sys.exit(1)
    
    # 提交到Kaggle
    success = submit_to_kaggle(file_path, args.message)
    
    if success:
        print("\n🎉 提交完成!")
        print("🌐 可以在以下网址查看提交结果:")
        print("   https://www.kaggle.com/competitions/titanic/submissions")
    else:
        print("\n💔 提交失败，请检查错误信息并重试")
        sys.exit(1)

if __name__ == "__main__":
    main() 