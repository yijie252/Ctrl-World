# ✅ Ctrl-World Flexiv训练环境恢复完成

## 📝 已恢复的所有文件

### 1. 配置文件
- ✅ `config_flexiv.py` - A100 8卡 10万步训练配置
- ✅ `train_flexiv.sh` - 训练启动脚本

### 2. 数据转换
- ✅ `flexiv_to_ctrlworld.py` - Flexiv HDF5转Ctrl-World格式
- ✅ `convert_flexiv_data.sh` - 数据转换便捷脚本

### 3. 核心修复
- ✅ `models/ctrl_world.py` - 修复UNet类型问题
- ✅ `scripts/train_wm.py` - 支持config_flexiv.py，修复验证bug
- ✅ `dataset/dataset_droid_exp33.py` - 修复stat.json路径

---

## 🚀 完整使用流程

### Step 1: 转换Flexiv数据

```bash
cd /root/workspace/chenyj36@xiaopeng.com/Ctrl-World

# 转换HDF5数据为Ctrl-World格式
bash convert_flexiv_data.sh /path/to/flexiv/hdf5/files

# 或者直接使用Python脚本
python3 flexiv_to_ctrlworld.py \
    --input_dir /path/to/flexiv/hdf5/files \
    --output_dir dataset_example/flexiv_data \
    --vae_path /workspace/chenyj36@xiaopeng.com/models/stable-video-diffusion-img2vid \
    --dataset_name flexiv_1113 \
    --device cuda
```

**输出结构**:
```
dataset_example/
├── flexiv_data/              # 转换后的数据
│   ├── 0/                    # Episode 0
│   │   ├── latent_videos/
│   │   │   ├── 0/           # 视角0 (cam_high)
│   │   │   │   └── 0.pt
│   │   │   ├── 1/           # 视角1 (cam_left_wrist)
│   │   │   │   └── 0.pt
│   │   │   └── 2/           # 视角2 (cam_right_wrist)
│   │   │       └── 0.pt
│   │   └── annotation/
│   │       ├── 0.json
│   │       ├── 1.json
│   │       └── 2.json
│   ├── 1/                    # Episode 1
│   └── ...
└── flexiv_data_meta_info/    # Meta信息
    └── flexiv_1113/
        ├── stat.json         # 统计信息
        ├── train_sample.json # 训练集列表
        └── val_sample.json   # 验证集列表
```

### Step 2: 开始训练

```bash
cd /root/workspace/chenyj36@xiaopeng.com/Ctrl-World
bash train_flexiv.sh
```

**训练配置**:
- GPU: 8 × A100 80GB
- Batch Size: 32 (4×8)
- Total Steps: 100,000
- Checkpoint: 每10,000步
- Validation: 每25,000步（生成3个视频）
- 预估时间: ~40小时

### Step 3: 监控训练

**WandB监控**:
- 项目名: `ctrl_world_flexiv`
- 指标: loss, learning_rate, validation_videos

**文件输出**:
```
model_ckpt/flexiv_finetune/
├── checkpoint-10000.pt
├── checkpoint-20000.pt
├── ...
├── checkpoint-100000.pt
└── samples/
    ├── train_steps_25000_0.mp4
    ├── train_steps_25000_1.mp4
    └── ...
```

---

## 🔧 关键修复说明

### 1. UNet类型问题
**问题**: `Expected types for unet: models.unet... got diffusers.models.unet...`

**修复**: 
- `models/ctrl_world.py`: 使用 `pipeline.register_modules(unet=unet)`
- `scripts/train_wm.py`: checkpoint加载时使用 `strict=False`

### 2. 验证时机错误
**问题**: 训练第5步就开始验证

**修复**: 
- `scripts/train_wm.py`: 改为 `global_step % args.validation_steps == 0 and global_step > 0`

### 3. stat.json路径错误
**问题**: `FileNotFoundError: .../{dataset_name}/stat.json`

**修复**:
- `dataset/dataset_droid_exp33.py`: 使用 `dataset_cfg` 而不是 `dataset_name`

### 4. 支持自定义配置
**新增**:
- `scripts/train_wm.py`: 支持 `--config config_flexiv` 参数

---

## 📊 训练参数详解

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_train_steps` | 100,000 | 总训练步数 |
| `train_batch_size` | 4 | 每卡batch size |
| `gradient_accumulation_steps` | 1 | 无需累积 |
| `num_workers` | 8 | 数据加载worker数 |
| `learning_rate` | 5e-6 | 微调学习率 |
| `checkpointing_steps` | 10,000 | checkpoint频率 |
| `validation_steps` | 25,000 | 验证频率 |
| `video_num` | 3 | 验证视频数量 |
| `width` | 320 | 图像宽度 |
| `height` | 192 | 图像高度 |
| `num_frames` | 5 | 预测帧数 |
| `num_history` | 6 | 历史帧数 |
| `action_dim` | 7 | Flexiv动作维度 |

---

## ⚠️ 常见问题

### Q1: checkpoint-10000.pt 在哪里？
A: `/workspace/chenyj36@xiaopeng.com/models/checkpoint-10000.pt`

如果没有，需要从共享目录复制或下载。

### Q2: 显存不够怎么办？
A: 当前配置已针对A100 80GB优化，预估使用30-35GB/GPU。如果还不够：
1. 减少 `train_batch_size` (4 → 2)
2. 增加 `gradient_accumulation_steps` (1 → 2)
3. 减少 `num_frames` (5 → 3)

### Q3: 如何从中断处继续训练？
A: 修改 `config_flexiv.py`:
```python
ckpt_path = 'model_ckpt/flexiv_finetune/checkpoint-50000.pt'
```

### Q4: 如何调整验证频率？
A: 修改 `config_flexiv.py`:
```python
validation_steps = 50000  # 每5万步验证
video_num = 1             # 只生成1个视频
# 或
video_num = 0             # 完全不验证
```

---

## 📁 模型文件位置

确保以下文件存在:
```
/workspace/chenyj36@xiaopeng.com/models/
├── stable-video-diffusion-img2vid/  # SVD模型
│   ├── vae/
│   ├── image_encoder/
│   └── ...
├── clip-vit-base-patch32/          # CLIP模型
│   └── ...
└── checkpoint-10000.pt             # 预训练checkpoint
```

---

## ✅ 恢复清单

- [x] 配置文件 (config_flexiv.py)
- [x] 训练脚本 (train_flexiv.sh)
- [x] 数据转换 (flexiv_to_ctrlworld.py)
- [x] UNet类型修复
- [x] 验证时机修复
- [x] Dataset路径修复
- [x] 支持自定义配置

**所有修复完成，可以开始训练！** 🚀

---

## 📞 获取帮助

如有问题，检查:
1. 训练日志中的错误信息
2. WandB监控面板
3. 本文档的常见问题部分

祝训练顺利！🎉

