#!/bin/bash

# Flexiv数据集训练脚本 - 8卡A100 80GB

echo "================================================"
echo "✨ Ctrl-World Flexiv微调训练 (A100 80GB)"
echo "================================================"

cd /root/workspace/chenyj36@xiaopeng.com/Ctrl-World

# A100配置（80GB显存，使用全部8卡）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 禁用SwanLab，使用WandB
export SWANLAB_MODE=disabled
export SWANLAB_LOG_LEVEL=ERROR

# WandB配置（在线模式，自动上传）
# 如需离线: export WANDB_MODE=offline
export WANDB_PROJECT=ctrl_world_flexiv
# export WANDB_MODE=offline  # 取消注释以启用离线模式

# PyTorch性能优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 数据加载优化
export TOKENIZERS_PARALLELISM=false  # 避免tokenizer警告

# 单节点训练配置（A100 8卡）
NUM_GPUS=8

echo ""
echo "✨ A100 8卡加速配置:"
echo "  GPU类型: A100 (80GB显存)"
echo "  GPU数量: ${NUM_GPUS}"
echo "  训练策略: 微调 checkpoint-10000.pt"
echo "  数据集: Flexiv (21 episodes)"
echo "  Batch Size: 2(每卡) × ${NUM_GPUS} × 2(累积) = 32"
echo "  梯度累积: 2步（速度优化）"
echo "  验证视频: 3个"
echo "  验证频率: 每25000步 (共4次)"
echo "  Checkpoint: 每10000步保存 (共10次)"
echo "  Worker数: 8 (持久化workers)"
echo "  学习率: 5e-6"
echo "  🔥 Max Steps: 100,000 (充分训练)"
echo "  ⚡ 加速优化: pin_memory + prefetch + TF32"
echo "  预估显存: ~25-30 GB/GPU (速度优化配置)"
echo ""

# 检查config.py是否已配置为Flexiv
if ! grep -q "flexiv_data" config.py 2>/dev/null; then
    if [ -f "config_flexiv.py" ]; then
        echo "📝 使用 config_flexiv.py 配置"
    else
        echo "⚠️  警告: config.py和config_flexiv.py都未配置为Flexiv"
        echo "   请确保配置文件正确"
    fi
fi

# 检查数据是否存在
if [ ! -d "dataset_example/flexiv_data" ]; then
    echo "❌ 错误: 数据集不存在!"
    echo "   数据应该在: dataset_example/flexiv_data/"
    exit 1
fi

if [ ! -f "dataset_example/flexiv_data_meta_info/flexiv_1113/stat.json" ]; then
    echo "❌ 错误: Meta信息不存在!"
    echo "   Meta信息应该在: dataset_example/flexiv_data_meta_info/flexiv_1113/"
    exit 1
fi

echo "✅ 数据检查通过"
echo ""

# 检查模型文件
if [ ! -f "/workspace/chenyj36@xiaopeng.com/models/checkpoint-10000.pt" ]; then
    echo "⚠️  警告: checkpoint-10000.pt 不存在"
    echo "   路径: /workspace/chenyj36@xiaopeng.com/models/checkpoint-10000.pt"
fi

# 启动训练
echo "🎯 开始训练..."
echo "------------------------------------------------"

# 使用config_flexiv.py
export PYTHONPATH=/root/workspace/chenyj36@xiaopeng.com/Ctrl-World:$PYTHONPATH

accelerate launch \
    --mixed_precision fp16 \
    --num_processes ${NUM_GPUS} \
    --main_process_port 29501 \
    scripts/train_wm.py \
    --config config_flexiv.py

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "🎉 训练完成!"
    echo "================================================"
    echo ""
    echo "📁 Checkpoint保存位置:"
    echo "   model_ckpt/flexiv_finetune/"
    echo ""
    echo "📹 验证视频保存位置:"
    echo "   model_ckpt/flexiv_finetune/samples/"
    echo ""
    ls -lh model_ckpt/flexiv_finetune/*.pt 2>/dev/null | tail -10
else
    echo ""
    echo "❌ 训练失败，请检查错误信息"
    exit 1
fi
