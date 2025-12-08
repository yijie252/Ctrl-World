#!/bin/bash

# Flexiv数据转换脚本
# 将HDF5格式转换为Ctrl-World训练格式

echo "================================================"
echo "🔄 Flexiv数据转换为Ctrl-World格式"
echo "================================================"

# 配置参数
INPUT_DIR="/workspace/chenyj36@xiaopeng.com/ex1"  # HDF5文件目录
OUTPUT_DIR="/workspace/chenyj36@xiaopeng.com/Ctrl-World/dataset_example/flexiv_data"
VAE_PATH="/workspace/chenyj36@xiaopeng.com/models/stable-video-diffusion-img2vid"
DATASET_NAME="flexiv_1113"


if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

echo ""
echo "📁 输入目录: $INPUT_DIR"
echo "📁 输出目录: $OUTPUT_DIR"
echo "🤖 VAE模型: $VAE_PATH"
echo "📊 数据集名: $DATASET_NAME"
echo ""

# 检查VAE模型
if [ ! -d "$VAE_PATH" ]; then
    echo "⚠️  警告: VAE模型不存在: $VAE_PATH"
    echo "   请先下载或配置VAE模型路径"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 开始转换
echo "🚀 开始转换..."
echo "------------------------------------------------"

cd /workspace/chenyj36@xiaopeng.com/Ctrl-World

python3 flexiv_to_ctrlworld.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --vae_path "$VAE_PATH" \
    --dataset_name "$DATASET_NAME" \
    --device cuda

# 检查转换结果
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✅ 转换完成!"
    echo "================================================"
    echo ""
    echo "📁 数据保存位置:"
    echo "   Episodes: $OUTPUT_DIR/"
    echo "   Meta info: ${OUTPUT_DIR}_meta_info/$DATASET_NAME/"
    echo ""
    echo "📊 目录结构:"
    tree -L 3 "$OUTPUT_DIR" 2>/dev/null || ls -lh "$OUTPUT_DIR"
    echo ""
    echo "📈 统计信息:"
    echo "   Episodes: $(ls -d $OUTPUT_DIR/*/ 2>/dev/null | wc -l)"
    echo "   Annotations: $(find $OUTPUT_DIR -name '*.json' | wc -l)"
    echo "   Latents: $(find $OUTPUT_DIR -name '*.pt' | wc -l)"
    echo ""
    echo "✅ 可以开始训练了!"
    echo "   运行: bash train_flexiv.sh"
else
    echo ""
    echo "❌ 转换失败，请检查错误信息"
    exit 1
fi

