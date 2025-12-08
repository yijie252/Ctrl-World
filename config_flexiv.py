"""
Flexiv数据集训练配置
基于checkpoint-10000.pt微调，学习Flexiv特定任务
A100 80GB × 8 - 10万步训练配置
"""

import torch
import os
from dataclasses import dataclass


@dataclass
class wm_args:
    ########################### training args ##############################
    # model paths (A100机器本地路径)
    svd_model_path = "/workspace/chenyj36@xiaopeng.com/models/stable-video-diffusion-img2vid"
    clip_model_path = "/workspace/chenyj36@xiaopeng.com/models/clip-vit-base-patch32"
    ckpt_path = '/workspace/chenyj36@xiaopeng.com/models/checkpoint-10000.pt'  # 从DROID预训练模型微调
    pi_ckpt = None  # 不需要policy checkpoint

    # dataset parameters - Flexiv数据集
    dataset_root_path = "/workspace/chenyj36@xiaopeng.com/Ctrl-World/dataset_example"
    dataset_names = 'flexiv_data'
    dataset_meta_info_path = '/workspace/chenyj36@xiaopeng.com/Ctrl-World/dataset_example/flexiv_data_meta_info'
    dataset_cfgs = 'flexiv_1113'
    prob = [1.0]
    annotation_name = 'annotation'
    num_workers = 8  # ✨ A100优化：充分利用CPU
    down_sample = 3  # 降采样 15Hz → 5Hz
    skip_step = 1

    # logs parameters
    debug = False
    tag = 'flexiv_finetune'
    output_dir = f"model_ckpt/{tag}"
    wandb_run_name = f"{tag}_21eps"
    wandb_project_name = "ctrl_world_flexiv"

    # training parameters - ✨ A100 80GB优化配置（加速版）
    learning_rate = 5e-6  # 微调用更小的学习率
    gradient_accumulation_steps = 1  # ⚡ 梯度累积2步（平衡速度和显存）
    mixed_precision = 'fp16'  # 混合精度训练
    train_batch_size = 4  # ⚡ 每卡batch=2 (2×8×2=32, 最佳平衡)
    shuffle = True
    num_train_epochs = 1000  # 数据少，多训练
    max_train_steps = 100000  # 🔥 10万步充分训练
    checkpointing_steps = 10000  # 🔥 每1万步保存（共10个checkpoint）
    validation_steps = 25000  # 🔥 每2.5万步验证（共4次）
    max_grad_norm = 1.0
    
    # validation - ✨ A100可以做验证
    video_num = 3  # ✨ 生成3个验证视频

    ############################ model args ##############################
    # model parameters
    motion_bucket_id = 127
    fps = 7
    guidance_scale = 2
    num_inference_steps = 50
    decode_chunk_size = 7
    width = 320  # 恢复原始分辨率（匹配已转换的数据）
    height = 192
    num_frames = 5  # 恢复原始帧数
    
    # action and history
    action_dim = 7  # Flexiv 7维控制
    num_history = 6  # 恢复原始历史帧数
    pred_step = 5  # 恢复原始预测步数
    his_cond_zero = False  # 是否将历史条件设为零（用于消融实验）
    frame_level_cond = True  # 是否使用帧级别的动作条件
    
    # text conditioning
    text_cond = True
    text_max_length = 77

    ########################### rollout args (for validation) ############################
    task_type: str = "replay"
    gripper_max_dict = {'replay': 1.0}
    policy_type = 'pi05'
    action_adapter = None
    policy_skip_step = 2
    interact_num = 12
    
    # wm validation
    data_stat_path = '/workspace/chenyj36@xiaopeng.com/Ctrl-World/dataset_example/flexiv_data_meta_info/flexiv_1113/stat.json'
    val_model_path = ckpt_path
    history_idx = [0, 0, -12, -9, -6, -3]
    
    # validation dataset
    val_dataset_dir = '/workspace/chenyj36@xiaopeng.com/Ctrl-World/dataset_example/flexiv_data'
    val_id = ['0']  # 验证用episode 0
    start_idx = [0]
    instruction = [""]  # 从annotation自动读取

    ########################### optimizer args ##############################
    optimizer_type = "adamw"
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-8
    
    ########################### scheduler args ##############################
    lr_scheduler = "constant"
    lr_warmup_steps = 500
    lr_num_cycles = 1
    lr_power = 1.0

    ########################### accelerate args ##############################
    allow_tf32 = True

