#!/usr/bin/env python3
"""
测试验证视频生成功能
"""

import sys
sys.path.append('.')

import torch
from config_flexiv import wm_args

print("="*60)
print("测试验证视频生成功能")
print("="*60)

args = wm_args()

# 测试1: 加载验证数据集
print("\n1️⃣ 测试加载验证数据集...")
try:
    from dataset.dataset_droid_exp33 import Dataset_mix
    val_dataset = Dataset_mix(args, mode='val')
    print(f"✅ 验证集大小: {len(val_dataset)} samples")
except Exception as e:
    print(f"❌ 加载验证集失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试2: 获取一个样本
print("\n2️⃣ 测试获取样本...")
try:
    sample = val_dataset[0]
    print(f"✅ Sample keys: {list(sample.keys())}")
    print(f"   latent shape: {sample['latent'].shape}")
    print(f"   action shape: {sample['action'].shape}")
    print(f"   text: {sample['text'][:50]}...")
except Exception as e:
    print(f"❌ 获取样本失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 检查latent维度
print("\n3️⃣ 检查latent维度...")
latent = sample['latent']
frames, channels, height, width = latent.shape
print(f"   frames: {frames} (期望: {args.num_frames + args.num_history} = {args.num_frames} + {args.num_history})")
print(f"   channels: {channels} (期望: 4)")
print(f"   height: {height} (期望: 72 for 3 views)")
print(f"   width: {width} (期望: 40)")

if height == 72:
    num_views = 3
    print(f"✅ 检测到 {num_views} 个视角 (72 = 24×3)")
elif height == 48:
    num_views = 2
    print(f"⚠️  检测到 {num_views} 个视角 (48 = 24×2)")
    print(f"   警告: 验证代码中硬编码了3个视角，可能需要调整！")
else:
    print(f"❌ 未知的height: {height}")
    sys.exit(1)

# 测试4: 模拟验证视频生成的batch采样
print("\n4️⃣ 测试验证batch采样...")
try:
    videos_row = args.video_num if not args.debug else 1
    videos_col = 2
    batch_id = list(range(0, len(val_dataset), int(len(val_dataset)/videos_row/videos_col)))
    batch_id = batch_id[0:videos_col]  # 取第一批
    print(f"   videos_row: {videos_row}")
    print(f"   videos_col: {videos_col}")
    print(f"   batch_id: {batch_id}")
    
    batch_list = [val_dataset.__getitem__(id) for id in batch_id]
    print(f"✅ 成功采样 {len(batch_list)} 个样本")
    
    # 测试batch拼接
    video_gt = torch.cat([t['latent'].unsqueeze(0) for t in batch_list], dim=0)
    print(f"   video_gt shape: {video_gt.shape}")
    
except Exception as e:
    print(f"❌ Batch采样失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: 检查einops rearrange
print("\n5️⃣ 测试einops rearrange操作...")
try:
    import einops
    
    # 模拟验证代码中的rearrange
    # 假设pred_latents shape: (B, F, C, H, W) = (2, 5, 4, 72, 40)
    B, F = len(batch_list), args.num_frames
    C, H, W = 4, height, width
    
    dummy_latents = torch.randn(B, F, C, H, W)
    print(f"   输入 shape: {dummy_latents.shape}")
    
    # 验证代码中的rearrange: m=3, n=1
    try:
        rearranged = einops.rearrange(dummy_latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)
        print(f"✅ rearrange(m=3, n=1) 成功: {rearranged.shape}")
        print(f"   输出: (B*3*1, F, C, H/3, W/1) = ({B}*3*1, {F}, {C}, {H//3}, {W})")
    except Exception as e:
        print(f"❌ rearrange(m=3, n=1) 失败: {e}")
        if height == 48:
            print(f"   原因: height=48无法被m=3整除 (48 / 3 = 16, 但24才是单视角高度)")
            print(f"   建议: 改用 m=2 (48 / 2 = 24) 或者保持m=3但height必须是72")
    
except Exception as e:
    print(f"❌ einops测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试6: 检查height参数
print("\n6️⃣ 检查验证生成的height参数...")
expected_height = int(3 * args.height)  # 验证代码第207行
print(f"   args.height: {args.height}")
print(f"   生成时的height: 3 × {args.height} = {expected_height}")
print(f"   实际latent的height: {height}")

if height == 72:
    print(f"✅ 匹配: 3 × {args.height} = {expected_height} ≈ {height}")
else:
    print(f"⚠️  不匹配: 期望 {expected_height}, 实际 {height}")

# 总结
print("\n" + "="*60)
print("📊 测试总结")
print("="*60)
print(f"验证集样本数: {len(val_dataset)}")
print(f"Latent维度: {latent.shape}")
print(f"视角数量: {num_views}")
print(f"验证代码期望视角数: 3")

if num_views == 3 and height == 72:
    print("\n✅ 所有检查通过！验证视频生成应该不会报错。")
elif height == 48:
    print("\n⚠️  警告: 当前latent height=48(2视角), 但验证代码期望72(3视角)")
    print("   需要修改验证代码适配2视角，或者确保数据处理生成72高度的latent")
else:
    print(f"\n❌ 发现问题: latent height={height}, 需要检查")

print("="*60)

