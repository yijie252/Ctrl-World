#!/usr/bin/env python3
"""
测试完整轨迹的视频生成（参考rollout_replay_traj.py）
滚动预测整个episode，生成完整的对比视频
"""

import sys
sys.path.append('.')

import torch
import os
import numpy as np
from pathlib import Path
from config_flexiv import wm_args
import json
import einops
import mediapy
from tqdm import tqdm
from models.ctrl_world import CrtlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

print("="*60)
print("测试完整轨迹视频生成")
print("="*60)

args = wm_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 加载模型
print("\n1️⃣ 加载模型...")
try:
    model = CrtlWorld(args).to(device)
    
    if os.path.exists(args.ckpt_path):
        print(f"   加载checkpoint: {args.ckpt_path}")
        state_dict = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("   ✓ Checkpoint加载完成")
    else:
        print(f"   ⚠️  使用随机初始化（测试用）")
    
    model.eval()
    pipeline = model.pipeline
    print(f"✅ 模型加载成功")
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. 加载数据统计信息
print("\n2️⃣ 加载数据统计信息...")
with open(args.data_stat_path, 'r') as f:
    data_stat = json.load(f)
    state_p01 = np.array(data_stat['state_01'])[None, :]
    state_p99 = np.array(data_stat['state_99'])[None, :]
print(f"✅ 数据统计加载成功")

def normalize_bound(data, data_min, data_max, clip_min=-1, clip_max=1, eps=1e-8):
    ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
    return np.clip(ndata, clip_min, clip_max)

# 3. 读取验证集的一个episode
print("\n3️⃣ 读取验证episode...")
val_dataset_dir = args.val_dataset_dir

# 获取验证集中的所有episode
val_annotation_dir = Path(f"{val_dataset_dir}/annotation/val")
val_episodes = sorted([f.stem for f in val_annotation_dir.glob("*.json")])
print(f"   可用的验证集episodes: {val_episodes}")

# 使用第一个可用的验证episode
val_id = val_episodes[0]
start_idx = 0  # 从第0帧开始

annotation_path = f"{val_dataset_dir}/annotation/val/{val_id}.json"
print(f"   选择Episode: {val_id}")
print(f"   Annotation: {annotation_path}")

with open(annotation_path) as f:
    anno = json.load(f)
    instruction = anno['texts'][0]
    
print(f"   Episode ID: {val_id}")
print(f"   Instruction: {instruction}")
print(f"   Start index: {start_idx}")

# 4. 加载完整的latent videos
print("\n4️⃣ 加载latent videos...")
video_latents = []
for latent_info in anno['latent_videos']:
    latent_path = f"{val_dataset_dir}/{latent_info['latent_video_path']}"
    latent = torch.load(latent_path, map_location='cpu')
    video_latents.append(latent)
    print(f"   ✓ {latent_path}: {latent.shape}")

print(f"✅ 加载了 {len(video_latents)} 个视角的latent")

# 5. 加载states/actions
print("\n5️⃣ 加载states和actions...")
cartesian_pose = np.array(anno['observation.state.cartesian_position'])
gripper_pose = np.array(anno['observation.state.gripper_position'])

print(f"   Cartesian pose shape: {cartesian_pose.shape}")
print(f"   Gripper pose shape: {gripper_pose.shape}")

# 拼接成完整的state (cartesian + gripper)
if len(gripper_pose.shape) == 1:
    gripper_pose = gripper_pose[:, np.newaxis]  # (T,) -> (T, 1)
    
states = np.concatenate([cartesian_pose, gripper_pose], axis=-1)
print(f"   States shape (cartesian+gripper): {states.shape}")
print(f"   总帧数: {states.shape[0]}")

# 6. 滚动预测设置
print("\n6️⃣ 配置滚动预测...")
pred_step = args.pred_step
num_history = args.num_history
num_frames = args.num_frames

# 计算最大可预测的交互次数（避免超出episode长度）
max_interact_num = (states.shape[0] - start_idx - num_history) // (pred_step - 1)

# 限制生成时长（默认20秒）
target_duration = 20  # 秒
target_frames = int(target_duration * args.fps)
target_interact_num = target_frames // (pred_step - 1)
interact_num = min(target_interact_num, max_interact_num)

total_frames = interact_num * (pred_step - 1)
duration_seconds = total_frames / args.fps

print(f"   pred_step: {pred_step}")
print(f"   num_history: {num_history}")
print(f"   num_frames: {num_frames}")
print(f"   目标时长: {target_duration} 秒")
print(f"   interact_num: {interact_num} (最大{max_interact_num})")
print(f"   总共预测帧数: {total_frames} 帧")
print(f"   实际视频时长: {duration_seconds:.1f} 秒")
print(f"   原始episode长度: {states.shape[0]} 帧 ({states.shape[0]/args.fps:.1f} 秒)")

# 7. 开始滚动预测
print("\n7️⃣ 开始滚动预测...")

# 初始化history buffer
his_cond = []
his_states = []

# 拼接第一帧的latent（Flexiv: 2个视角 → 复用第3个）
num_views = len(video_latents)
latent_list = [v[start_idx:start_idx+1] for v in video_latents]

# 如果只有2个视角，复用第2个作为第3个
if num_views == 2:
    latent_list.append(latent_list[1])  # 复用第2个视角
    print(f"   检测到{num_views}个视角，复用第2个作为第3个")
    
first_latent = torch.cat(latent_list, dim=2).to(device)  # (1, 4, 72, 40)
print(f"   first_latent shape: {first_latent.shape}")
assert first_latent.shape[2] == 72, f"Expected height=72, got {first_latent.shape[2]}"

# 填充history buffer
for i in range(num_history * 4):
    his_cond.append(first_latent)
    his_states.append(states[start_idx:start_idx+1])

# 存储结果
video_to_save = []

# 滚动预测循环
for i in tqdm(range(interact_num), desc="滚动预测", unit="步"):
    if i % 10 == 0:  # 每10步打印一次详细信息
        print(f"\n   ===== 预测步骤 {i+1}/{interact_num} =====")
    
    # 当前步骤的帧范围
    step_start = start_idx + int(i * (pred_step - 1))
    step_end = step_start + pred_step
    
    if i % 10 == 0:
        print(f"   帧范围: {step_start} ~ {step_end}")
    
    # 准备ground truth latents（2个视角 → 复用第3个）
    video_latent_true = [v[step_start:step_end].to(device) for v in video_latents]
    if len(video_latent_true) == 2:
        video_latent_true.append(video_latent_true[1])  # 复用第2个视角
    
    # 准备action condition
    history_idx = [0, 0, -8, -6, -4, -2]
    his_pose = np.concatenate([his_states[idx] for idx in history_idx], axis=0)
    action_seq = states[step_start:step_end]
    action_cond = np.concatenate([his_pose, action_seq], axis=0)
    
    # 归一化action
    action_cond = normalize_bound(action_cond, state_p01, state_p99)
    action_cond = torch.tensor(action_cond).unsqueeze(0).to(device).to(torch.float32)
    
    # 准备history和current latent
    his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
    current_latent = his_cond[-1]
    
    # 前向传播
    with torch.no_grad():
        # Encode action
        action_latent = model.action_encoder(
            action_cond, [instruction], model.tokenizer, model.text_encoder, args.frame_level_cond
        )
        
        # Generate video
        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,
            text=action_latent,
            width=args.width,
            height=int(3*args.height),
            num_frames=args.num_frames,
            history=his_cond_input,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
            output_type='latent',
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )
    
    # Rearrange
    pred_latents = einops.rearrange(pred_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)
    
    # Decode ground truth
    true_video = torch.stack(video_latent_true, dim=0)  # (3, T, 4, 24, 40)
    decoded_true = []
    bsz, frame_num = true_video.shape[:2]
    true_video_flat = true_video.flatten(0, 1)
    
    decode_kwargs = {}
    for j in range(0, true_video_flat.shape[0], args.decode_chunk_size):
        chunk = true_video_flat[j:j+args.decode_chunk_size] / pipeline.vae.config.scaling_factor
        decode_kwargs["num_frames"] = chunk.shape[0]
        decoded_true.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
    
    true_video = torch.cat(decoded_true, dim=0)
    true_video = true_video.reshape(bsz, frame_num, *true_video.shape[1:])
    true_video = ((true_video / 2.0 + 0.5).clamp(0, 1) * 255)
    true_video = true_video.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)
    
    # Decode prediction
    decoded_pred = []
    bsz, frame_num = pred_latents.shape[:2]
    pred_latents_flat = pred_latents.flatten(0, 1)
    
    for j in range(0, pred_latents_flat.shape[0], args.decode_chunk_size):
        chunk = pred_latents_flat[j:j+args.decode_chunk_size] / pipeline.vae.config.scaling_factor
        decode_kwargs["num_frames"] = chunk.shape[0]
        decoded_pred.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
    
    pred_video = torch.cat(decoded_pred, dim=0)
    pred_video = pred_video.reshape(bsz, frame_num, *pred_video.shape[1:])
    pred_video = ((pred_video / 2.0 + 0.5).clamp(0, 1) * 255)
    pred_video = pred_video.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)
    
    # 拼接GT和预测
    videos_cat = np.concatenate([true_video, pred_video], axis=-3)  # (3, T, H*2, W, 3)
    videos_cat = np.concatenate([video for video in videos_cat], axis=-2).astype(np.uint8)  # (T, H*2, W*3, 3)
    
    # 保存到buffer（去掉最后一帧避免重复）
    if i == interact_num - 1:
        video_to_save.append(videos_cat)
    else:
        video_to_save.append(videos_cat[:pred_step-1])
    
    # 更新history（拼接3个视角的预测结果）
    his_states.append(action_seq[pred_step-1:pred_step])
    # pred_latents已经是3个视角了（rearrange后是6个：batch0-view0/1/2, batch1-view0/1/2）
    # 对于batch 0，取前3个视角
    pred_last_latent = torch.cat([pred_latents[j, pred_step-1:pred_step] for j in range(3)], dim=2)  # (1, 4, 72, 40)
    his_cond.append(pred_last_latent)

# 8. 拼接并保存完整视频
print("\n8️⃣ 保存完整轨迹视频...")
output_dir = Path("test_validation_output")
output_dir.mkdir(exist_ok=True)

video_full = np.concatenate(video_to_save, axis=0)
print(f"   完整视频shape: {video_full.shape}")

output_path = output_dir / f"test_rollout_val{val_id}_start{start_idx}.mp4"
mediapy.write_video(str(output_path), video_full, fps=args.fps)

print(f"✅ 完整轨迹视频保存: {output_path}")
print(f"   总帧数: {video_full.shape[0]}")
print(f"   分辨率: {video_full.shape[1]}×{video_full.shape[2]}")
print(f"   布局: GT上 + 预测下 + 3视角横排")

print("\n" + "="*60)
print("🎉 完整轨迹视频生成完成！")
print("="*60)

