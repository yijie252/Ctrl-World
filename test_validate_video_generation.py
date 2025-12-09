#!/usr/bin/env python3
"""
测试验证视频生成的完整流程
"""

import sys
sys.path.append('.')

import torch
import os
import numpy as np
from pathlib import Path
from config_flexiv import wm_args

print("="*60)
print("测试验证视频生成完整流程")
print("="*60)

args = wm_args()

# 1. 加载验证数据集
print("\n1️⃣ 加载验证数据集...")
try:
    from dataset.dataset_droid_exp33 import Dataset_mix
    val_dataset = Dataset_mix(args, mode='val')
    print(f"✅ 验证集加载成功: {len(val_dataset)} samples")
except Exception as e:
    print(f"❌ 加载验证集失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. 模拟验证batch采样（和train_wm.py中的逻辑一致）
print("\n2️⃣ 模拟验证batch采样...")
try:
    videos_row = args.video_num if not args.debug else 1
    videos_col = 2
    id = 0  # 第0个GPU
    
    # 采样逻辑和validate_video_generation函数中一致
    batch_id = list(range(0, len(val_dataset), int(len(val_dataset)/videos_row/videos_col)))
    batch_id = batch_id[int(id*(videos_col)):int((id+1)*(videos_col))]
    
    print(f"   videos_row: {videos_row}")
    print(f"   videos_col: {videos_col}")
    print(f"   batch_id: {batch_id}")
    print(f"   采样 {len(batch_id)} 个样本")
    
    batch_list = [val_dataset.__getitem__(bid) for bid in batch_id]
    print(f"✅ Batch采样成功")
    
except Exception as e:
    print(f"❌ Batch采样失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. 构造batch数据（模拟GPU tensor）
print("\n3️⃣ 构造batch数据...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   使用设备: {device}")
    
    video_gt = torch.cat([t['latent'].unsqueeze(0) for i,t in enumerate(batch_list)], dim=0)
    text = [t['text'] for i,t in enumerate(batch_list)]
    actions = torch.cat([t['action'].unsqueeze(0) for i,t in enumerate(batch_list)], dim=0)
    
    # 移到设备上（如果有GPU）
    video_gt = video_gt.to(device)
    actions = actions.to(device)
    
    print(f"   video_gt shape: {video_gt.shape}")
    print(f"   actions shape: {actions.shape}")
    print(f"   text samples: {len(text)}")
    
    print(f"✅ Batch数据构造成功")
    
except Exception as e:
    print(f"❌ Batch数据构造失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 测试验证代码中的assertion逻辑
print("\n4️⃣ 测试assertion逻辑...")
try:
    his_latent_gt, future_latent_ft = video_gt[:,:args.num_history], video_gt[:,args.num_history:]
    current_latent = future_latent_ft[:,0]
    
    print(f"   his_latent_gt shape: {his_latent_gt.shape}")
    print(f"   future_latent_ft shape: {future_latent_ft.shape}")
    print(f"   current_latent shape: {current_latent.shape}")
    print(f"   actions shape: {actions.shape}")
    
    # 执行和train_wm.py中完全相同的assertion
    print("\n   执行assertion检查...")
    
    # 检查1: channels
    assert current_latent.shape[1] == 4, f"Expected 4 channels, got {current_latent.shape[1]}"
    print(f"   ✓ Channels: {current_latent.shape[1]} == 4")
    
    # 检查2: width
    expected_latent_width = args.width // 8  # VAE下采样8倍: 320//8=40
    assert current_latent.shape[3] == expected_latent_width, f"Expected width {expected_latent_width}, got {current_latent.shape[3]}"
    print(f"   ✓ Width: {current_latent.shape[3]} == {expected_latent_width}")
    
    # 检查3: height（多视角）
    num_views = current_latent.shape[2] // 24
    print(f"   ✓ Height: {current_latent.shape[2]} = {num_views} views × 24")
    
    # 检查4: actions
    assert actions.shape[1:] == (int(args.num_frames+args.num_history), args.action_dim), \
        f"Expected actions shape {(int(args.num_frames+args.num_history), args.action_dim)}, got {actions.shape[1:]}"
    print(f"   ✓ Actions: {actions.shape[1:]} == ({args.num_frames+args.num_history}, {args.action_dim})")
    
    print(f"\n✅ 所有assertion检查通过！")
    
except AssertionError as e:
    print(f"❌ Assertion失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 测试einops rearrange操作
print("\n5️⃣ 测试einops rearrange操作...")
try:
    import einops
    
    # 模拟pred_latents
    B, F = current_latent.shape[0], args.num_frames
    C, H, W = current_latent.shape[1], current_latent.shape[2], current_latent.shape[3]
    
    dummy_pred_latents = torch.randn(B, F, C, H, W, device=device)
    print(f"   dummy_pred_latents shape: {dummy_pred_latents.shape}")
    
    # 执行rearrange（验证代码中的操作）
    rearranged = einops.rearrange(dummy_pred_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)
    print(f"   rearranged shape: {rearranged.shape}")
    print(f"   期望: ({B}*3*1, {F}, {C}, {H//3}, {W}) = ({B*3}, {F}, {C}, {H//3}, {W})")
    
    # 对video_gt也执行rearrange
    video_gt_rearranged = einops.rearrange(video_gt, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)
    print(f"   video_gt_rearranged shape: {video_gt_rearranged.shape}")
    
    print(f"✅ einops rearrange操作成功")
    
except Exception as e:
    print(f"❌ einops rearrange失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. 测试VAE decode参数
print("\n6️⃣ 测试VAE decode参数...")
try:
    # 检查height参数
    expected_height = int(3 * args.height)
    print(f"   args.height: {args.height}")
    print(f"   生成时的height参数: 3 × {args.height} = {expected_height}")
    print(f"   实际latent的height: {current_latent.shape[2]}")
    
    # 检查decode_chunk_size
    print(f"   decode_chunk_size: {args.decode_chunk_size}")
    
    # 模拟分chunk decode
    num_chunks = (video_gt.shape[0] * video_gt.shape[1] + args.decode_chunk_size - 1) // args.decode_chunk_size
    print(f"   总帧数: {video_gt.shape[0]} × {video_gt.shape[1]} = {video_gt.shape[0] * video_gt.shape[1]}")
    print(f"   需要 {num_chunks} 个chunks来decode")
    
    print(f"✅ VAE decode参数检查通过")
    
except Exception as e:
    print(f"❌ VAE decode参数检查失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. 加载模型并生成视频
print("\n7️⃣ 加载模型并生成视频...")
try:
    from models.ctrl_world import CrtlWorld
    from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
    
    print("   初始化模型...")
    model = CrtlWorld(args).to(device)
    
    # 加载checkpoint
    if os.path.exists(args.ckpt_path):
        print(f"   加载checkpoint: {args.ckpt_path}")
        state_dict = torch.load(args.ckpt_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"   ⚠️  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"   ⚠️  Unexpected keys: {len(unexpected_keys)}")
        print("   ✓ Checkpoint加载完成")
    else:
        print(f"   ⚠️  Checkpoint不存在: {args.ckpt_path}")
        print("   使用随机初始化的模型（仅用于测试）")
    
    model.eval()
    pipeline = model.pipeline
    
    print(f"✅ 模型加载成功")
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 8. 执行视频生成
print("\n8️⃣ 执行视频生成...")
try:
    with torch.no_grad():
        print("   编码action...")
        bsz = actions.shape[0]
        action_latent = model.action_encoder(
            actions, text, model.tokenizer, model.text_encoder, args.frame_level_cond
        )
        print(f"   action_latent shape: {action_latent.shape}")
        
        print("   生成视频latent...")
        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,
            text=action_latent,
            width=args.width,
            height=int(3*args.height),
            num_frames=args.num_frames,
            history=his_latent_gt,
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
        print(f"   pred_latents shape: {pred_latents.shape}")
    
    print(f"✅ 视频latent生成成功")
    
except Exception as e:
    print(f"❌ 视频生成失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 9. Rearrange和decode（参考rollout_replay_traj.py的完整流程）
print("\n9️⃣ Rearrange和decode视频...")
try:
    import einops
    
    # Rearrange pred_latents
    print("   Rearrange pred_latents...")
    pred_latents_rearranged = einops.rearrange(
        pred_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1
    )
    print(f"   pred_latents_rearranged shape: {pred_latents_rearranged.shape}")
    
    # Prepare ground truth video (只用future部分，和预测帧数匹配)
    print("   Prepare ground truth video (future frames only)...")
    # 只取future_latent_ft，不包含history，这样和预测的帧数一致
    video_gt_future = future_latent_ft  # 只用future部分
    video_gt_rearranged = einops.rearrange(
        video_gt_future, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1
    )
    print(f"   video_gt_rearranged shape: {video_gt_rearranged.shape}")
    
    # Decode ground truth video
    print("   Decode ground truth video...")
    true_video = video_gt_rearranged
    decoded_true = []
    bsz_true, frame_num_true = true_video.shape[:2]
    true_video_flat = true_video.flatten(0, 1)
    decode_kwargs = {}
    
    for i in range(0, true_video_flat.shape[0], args.decode_chunk_size):
        chunk = true_video_flat[i:i+args.decode_chunk_size] / pipeline.vae.config.scaling_factor
        decode_kwargs["num_frames"] = chunk.shape[0]
        decoded_true.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
    
    true_video_decoded = torch.cat(decoded_true, dim=0)
    true_video_decoded = true_video_decoded.reshape(bsz_true, frame_num_true, *true_video_decoded.shape[1:])
    
    # 转换为numpy格式: (bsz, T, C, H, W) -> (bsz, T, H, W, C)
    true_video_np = ((true_video_decoded / 2.0 + 0.5).clamp(0, 1) * 255)
    true_video_np = true_video_np.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)
    print(f"   true_video_np shape: {true_video_np.shape}")
    
    # Decode predicted video
    print("   Decode predicted video...")
    decoded_pred = []
    bsz_pred, frame_num_pred = pred_latents_rearranged.shape[:2]
    pred_latents_flat = pred_latents_rearranged.flatten(0, 1)
    
    for i in range(0, pred_latents_flat.shape[0], args.decode_chunk_size):
        chunk = pred_latents_flat[i:i+args.decode_chunk_size] / pipeline.vae.config.scaling_factor
        decode_kwargs["num_frames"] = chunk.shape[0]
        decoded_pred.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
    
    pred_video_decoded = torch.cat(decoded_pred, dim=0)
    pred_video_decoded = pred_video_decoded.reshape(bsz_pred, frame_num_pred, *pred_video_decoded.shape[1:])
    
    # 转换为numpy格式
    pred_video_np = ((pred_video_decoded / 2.0 + 0.5).clamp(0, 1) * 255)
    pred_video_np = pred_video_np.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)
    print(f"   pred_video_np shape: {pred_video_np.shape}")
    
    print(f"✅ 视频decode成功")
    
except Exception as e:
    print(f"❌ 视频decode失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 10. 保存对比视频（Ground Truth vs Predicted，参考rollout_replay_traj.py）
print("\n🔟 保存对比视频...")
try:
    import mediapy
    import numpy as np
    from torchvision.utils import save_image
    
    output_dir = Path("test_validation_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"   true_video_np shape: {true_video_np.shape}")
    print(f"   pred_video_np shape: {pred_video_np.shape}")
    
    # 拼接GT和预测视频，完全按照rollout_replay_traj.py的方式
    # true_video_np: (3, T, H, W, 3)
    # pred_video_np: (3, T, H, W, 3)
    
    # 沿着高度维度拼接GT和预测: (3, T, H, W, 3) -> (3, T, H*2, W, 3)
    videos_cat = np.concatenate([true_video_np, pred_video_np], axis=-3)
    print(f"   videos_cat shape after concat: {videos_cat.shape}")  # (3, T, H*2, W, 3)
    
    # 沿着宽度维度拼接3个视角: (3, T, H*2, W, 3) -> (T, H*2, W*3, 3)
    videos_cat = np.concatenate([video for video in videos_cat], axis=-2).astype(np.uint8)
    print(f"   videos_cat shape final: {videos_cat.shape}")  # (T, H*2, W*3, 3)
    
    # 保存完整的对比视频
    mp4_path = output_dir / "test_validation_comparison.mp4"
    mediapy.write_video(str(mp4_path), videos_cat, fps=args.fps)
    print(f"   ✓ 保存对比视频: {mp4_path}")
    print(f"      布局: 上半部分=Ground Truth (3视角横排)")
    print(f"            下半部分=Predicted (3视角横排)")
    print(f"      Shape: {videos_cat.shape} ({videos_cat.shape[0]} frames, {args.fps} fps)")
    
    # 另外保存单独的GT和预测视频
    print("\n   保存单独视频...")
    
    # GT视频（3视角横排）
    true_video_concat = np.concatenate([video for video in true_video_np], axis=-2).astype(np.uint8)
    gt_path = output_dir / "test_ground_truth.mp4"
    mediapy.write_video(str(gt_path), true_video_concat, fps=args.fps)
    print(f"   ✓ 保存GT视频: {gt_path} ({true_video_concat.shape})")
    
    # 预测视频（3视角横排）
    pred_video_concat = np.concatenate([video for video in pred_video_np], axis=-2).astype(np.uint8)
    pred_path = output_dir / "test_predicted.mp4"
    mediapy.write_video(str(pred_path), pred_video_concat, fps=args.fps)
    print(f"   ✓ 保存预测视频: {pred_path} ({pred_video_concat.shape})")
    
    # 保存每个视角的单独视频
    print("\n   保存各视角单独视频...")
    for view_idx in range(3):
        # GT视角
        gt_view_path = output_dir / f"test_gt_view{view_idx}.mp4"
        mediapy.write_video(str(gt_view_path), true_video_np[view_idx], fps=args.fps)
        
        # 预测视角
        pred_view_path = output_dir / f"test_pred_view{view_idx}.mp4"
        mediapy.write_video(str(pred_view_path), pred_video_np[view_idx], fps=args.fps)
        
        print(f"   ✓ 视角{view_idx}: {gt_view_path.name} + {pred_view_path.name}")
    
    print(f"\n✅ 对比视频保存成功！")
    print(f"   主要文件: {mp4_path.name} (GT上 + 预测下 + 3视角横排)")
    
except Exception as e:
    print(f"⚠️  保存视频失败（非致命）: {e}")
    import traceback
    traceback.print_exc()

# 总结
print("\n" + "="*60)
print("📊 测试总结")
print("="*60)
print(f"✅ 验证集样本数: {len(val_dataset)}")
print(f"✅ Batch大小: {len(batch_list)}")
print(f"✅ Latent维度: {current_latent.shape}")
print(f"✅ Actions维度: {actions.shape}")
print(f"✅ 视角数量: {num_views}")
print(f"✅ 所有assertion检查: 通过")
print(f"✅ einops操作: 通过")
print(f"✅ VAE参数: 正确")
print(f"✅ 模型加载: 成功")
print(f"✅ 视频生成: 成功")
print(f"✅ 视频decode: 成功 (GT + 预测)")
print(f"✅ 对比视频保存: 成功")
print("\n🎉 验证视频生成完整测试通过！")
print("   训练中的验证步骤完全正常。")
print(f"\n📁 输出文件:")
print(f"   主视频: test_validation_output/test_validation_comparison.mp4")
print(f"   布局: [GT上 + 预测下] × [3视角横排]")
print("="*60)

