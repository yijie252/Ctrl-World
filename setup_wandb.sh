#!/bin/bash

# WandB初始化脚本

echo "================================================"
echo "🔧 WandB 初始化设置"
echo "================================================"
echo ""

# 检查wandb是否安装
if ! python3 -c "import wandb" 2>/dev/null; then
    echo "❌ WandB未安装，正在安装..."
    pip3 install wandb -q
    echo "✅ WandB安装完成"
else
    echo "✅ WandB已安装"
fi

echo ""
echo "📝 WandB登录选项："
echo "   1. 在线模式 - 自动上传到wandb.ai（推荐）"
echo "   2. 离线模式 - 仅本地保存"
echo "   3. 已登录，跳过"
echo ""

read -p "请选择 (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "🌐 在线模式 - 请登录WandB"
        echo "   访问 https://wandb.ai/authorize 获取API密钥"
        wandb login
        ;;
    2)
        echo ""
        echo "💾 离线模式"
        export WANDB_MODE=offline
        echo "✅ 已设置WANDB_MODE=offline"
        ;;
    3)
        echo ""
        echo "✅ 跳过登录"
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "✅ WandB设置完成"
echo "================================================"
echo ""
echo "📊 训练将记录到项目: ctrl_world_flexiv"
echo ""
echo "🚀 现在可以开始训练:"
echo "   bash train_flexiv.sh"
echo ""

