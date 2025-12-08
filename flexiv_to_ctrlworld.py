"""
Flexiv HDF5数据转换为Ctrl-World格式
支持多视角、VAE编码、annotation生成
"""

import h5py
import numpy as np
import json
import cv2
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
import argparse
from diffusers import AutoencoderKLTemporalDecoder


def decode_image(img_data):
    """解码图像数据（支持JPEG压缩和原始像素）"""
    if img_data.ndim == 1:  # JPEG compressed
        jpeg_array = np.frombuffer(img_data, dtype=np.uint8)
        bgr = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif img_data.ndim == 3 and img_data.shape[2] == 3:  # Raw pixels
        img = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    else:
        img = img_data
    return img


def encode_latent_with_vae(images, vae, device='cuda'):
    """
    使用VAE编码图像为latent
    images: (T, H, W, 3) numpy array, uint8, RGB
    返回: (T, 4, h, w) tensor
    """
    vae.eval()
    vae.to(device)
    
    # 预处理图像
    imgs_tensor = []
    for img in images:
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((320, 192), Image.BICUBIC)  # Flexiv分辨率
        img_tensor = torch.from_numpy(np.array(img_pil)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # (3, H, W)
        img_tensor = img_tensor * 2.0 - 1.0  # normalize to [-1, 1]
        imgs_tensor.append(img_tensor)
    
    imgs_tensor = torch.stack(imgs_tensor).to(device)  # (T, 3, H, W)
    
    # VAE编码
    with torch.no_grad():
        latents = []
        batch_size = 8  # 批量处理避免OOM
        for i in range(0, len(imgs_tensor), batch_size):
            batch = imgs_tensor[i:i+batch_size]
            latent = vae.encode(batch).latent_dist.sample()
            latent = latent * vae.config.scaling_factor
            latents.append(latent.cpu())
        latents = torch.cat(latents, dim=0)  # (T, 4, h, w)
    
    return latents


def process_flexiv_episode(hdf5_path, output_dir, vae, episode_id, view_names=['camera_2', 'camera_3']):
        """
    处理单个Flexiv episode
    
        Args:
        hdf5_path: HDF5文件路径
        output_dir: 输出目录
        vae: VAE模型
        episode_id: episode编号
        view_names: 视角名称列表
    """
    output_dir = Path(output_dir)
    
    print(f"\n处理 Episode {episode_id}: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # 读取状态和动作
        obs = f['observations']
        actions = f['action'][()]  # (T, 7): [x, y, z, rx, ry, rz, gripper]
        
        # 使用qpos作为state (关节位置)
        states = obs['qpos'][()]  # (T, 7)
            
        # 读取任务描述
        task_desc = f['metadata/task_description'][()].decode('utf-8') if 'metadata/task_description' in f else "Flexiv manipulation task"
        
        T = len(actions)
        print(f"  总帧数: {T}")
        print(f"  任务描述: {task_desc}")
        
        # 处理每个视角
        for view_idx, view_name in enumerate(view_names):
            print(f"\n  处理视角 {view_idx}: {view_name}")
            
            # 读取RGB图像
            rgb_key = f'images/{view_name}'
            if rgb_key not in obs:
                print(f"    ⚠️  警告: 未找到 {rgb_key}")
                continue
            
            rgb_images = obs[rgb_key]
            print(f"    图像数据类型: {rgb_images.dtype}, shape: {rgb_images.shape}")
            
            # 解码所有图像
            decoded_images = []
            for i in tqdm(range(T), desc=f"    解码{view_name}图像"):
                img = decode_image(rgb_images[i])
                decoded_images.append(img)
            
            decoded_images = np.array(decoded_images)  # (T, H, W, 3)
            print(f"    图像shape: {decoded_images.shape}")
        
        # VAE编码
            print(f"    VAE编码...")
            latents = encode_latent_with_vae(decoded_images, vae)  # (T, 4, h, w)
            
            # 检查latent分辨率，如果是(40, 24)需要转置为(24, 40)
            if latents.shape[2:] == (40, 24):
                print(f"    ⚠️  latent分辨率异常: {latents.shape[2:]}，转置为 (24, 40)")
                latents = latents.transpose(2, 3)
            
            print(f"    Latent shape: {latents.shape}")
            
            # 确保states和latents长度一致
            actual_length = min(len(states), len(latents), len(actions))
            states_aligned = states[:actual_length]
            latents_aligned = latents[:actual_length]
            actions_aligned = actions[:actual_length]
            
            print(f"    对齐后长度: {actual_length} (原始: states={len(states)}, latents={len(latents)}, actions={len(actions)})")
        
            # 创建输出目录结构
            episode_dir = output_dir / str(episode_id)
            latent_dir = episode_dir / 'latent_videos' / str(view_idx)
            anno_dir = episode_dir / 'annotation'
            latent_dir.mkdir(parents=True, exist_ok=True)
            anno_dir.mkdir(parents=True, exist_ok=True)
        
            # 保存latent
            latent_path = latent_dir / '0.pt'
            torch.save(latents_aligned, latent_path)
            print(f"    ✅ 保存latent: {latent_path}")
        
            # 生成annotation
        annotation = {
                'latent': str(latent_path.relative_to(output_dir)),
                'text': task_desc,  # 使用实际的任务描述
                'actions': actions_aligned.tolist(),  # (T, 7)
                'states': states_aligned.tolist(),  # (T, 7) - qpos
                'observation.state.joint_position': states_aligned.tolist(),  # DataLoader需要
                'observation.state.gripper_position': actions_aligned[:, -1].tolist(),  # 使用action的gripper维度
                'dataset_name': 'flexiv_data',
                'episode_id': episode_id,
                'view_id': view_idx,
                'view_name': view_name,
                'num_frames': actual_length,
        }
        
            anno_path = anno_dir / f'{view_idx}.json'
            with open(anno_path, 'w') as f:
            json.dump(annotation, f, indent=2)
            print(f"    ✅ 保存annotation: {anno_path}")


def create_meta_info(output_dir, dataset_name='flexiv_1113'):
    """创建meta info文件（stat.json, train_sample.json, val_sample.json）"""
    output_dir = Path(output_dir)
    meta_dir = output_dir.parent / f'{output_dir.name}_meta_info' / dataset_name
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n生成meta信息: {meta_dir}")
    
    # 收集所有annotation
    all_annotations = list(output_dir.glob('*/annotation/*.json'))
    print(f"  找到 {len(all_annotations)} 个annotation文件")
    
    # 收集所有actions和states计算统计信息
    all_actions = []
    all_states = []
        samples = []
    
    for anno_path in tqdm(all_annotations, desc="  读取annotations"):
        with open(anno_path, 'r') as f:
                anno = json.load(f)
            all_actions.extend(anno['actions'])
            all_states.extend(anno['states'])
            
            # 添加到samples
            rel_path = str(anno_path.relative_to(output_dir))
            samples.append(rel_path)
    
    # 计算统计信息
    actions_array = np.array(all_actions)  # (N, 7)
    states_array = np.array(all_states)    # (N, state_dim)
    
    action_mean = actions_array.mean(axis=0).tolist()
    action_std = actions_array.std(axis=0).tolist()
    action_min = actions_array.min(axis=0).tolist()
    action_max = actions_array.max(axis=0).tolist()
    
    state_mean = states_array.mean(axis=0).tolist()
    state_std = states_array.std(axis=0).tolist()
    state_min = states_array.min(axis=0).tolist()
    state_max = states_array.max(axis=0).tolist()
    
    # 计算第1和第99百分位数（用于归一化）
    state_01 = np.percentile(states_array, 1, axis=0).tolist()
    state_99 = np.percentile(states_array, 99, axis=0).tolist()
    
    stat = {
        'action_mean': action_mean,
        'action_std': action_std,
        'action_min': action_min,
        'action_max': action_max,
        'state_mean': state_mean,
        'state_std': state_std,
        'state_min': state_min,
        'state_max': state_max,
        'state_01': state_01,  # 第1百分位数
        'state_99': state_99,  # 第99百分位数
    }
    
    # 保存stat.json
    with open(meta_dir / 'stat.json', 'w') as f:
        json.dump(stat, f, indent=2)
    print(f"  ✅ 保存stat.json")
    
    # 划分train/val (90%/10%)
    np.random.seed(42)
    np.random.shuffle(samples)
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    # 保存sample lists
    with open(meta_dir / 'train_sample.json', 'w') as f:
        json.dump({'samples': train_samples}, f, indent=2)
    print(f"  ✅ 保存train_sample.json ({len(train_samples)} samples)")
    
    with open(meta_dir / 'val_sample.json', 'w') as f:
        json.dump({'samples': val_samples}, f, indent=2)
    print(f"  ✅ 保存val_sample.json ({len(val_samples)} samples)")
    
    print(f"\n✅ Meta信息生成完成!")
    print(f"   Train samples: {len(train_samples)}")
    print(f"   Val samples: {len(val_samples)}")
    print(f"   Total: {len(samples)}")


def main():
    parser = argparse.ArgumentParser(description='将Flexiv HDF5数据转换为Ctrl-World格式')
    parser.add_argument('--input_dir', type=str, required=True, help='输入HDF5文件目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--vae_path', type=str, 
                        default='/workspace/chenyj36@xiaopeng.com/models/stable-video-diffusion-img2vid',
                        help='VAE模型路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--dataset_name', type=str, default='flexiv_1113', help='数据集名称')
    args = parser.parse_args()
    
    # 加载VAE
    print(f"加载VAE模型: {args.vae_path}")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.vae_path, subfolder="vae")
    vae.eval()
    vae.to(args.device)
    print("✅ VAE加载完成")
    
    # 查找所有HDF5文件
    input_dir = Path(args.input_dir)
    hdf5_files = sorted(list(input_dir.glob('*.hdf5')))
    print(f"\n找到 {len(hdf5_files)} 个HDF5文件")
    
    if len(hdf5_files) == 0:
        print(f"❌ 错误: 在 {input_dir} 中未找到HDF5文件")
        return
    
    # 处理每个episode
    for episode_id, hdf5_path in enumerate(hdf5_files):
        try:
            process_flexiv_episode(
                hdf5_path=hdf5_path,
                output_dir=args.output_dir,
                vae=vae,
                episode_id=episode_id,
                view_names=['camera_2', 'camera_3'],  # 实际的相机名称
            )
        except Exception as e:
            print(f"❌ 处理 {hdf5_path} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成meta信息
    create_meta_info(Path(args.output_dir), args.dataset_name)
    
    print(f"\n{'='*60}")
    print(f"✅ 所有转换完成!")
    print(f"{'='*60}")
    print(f"输出目录: {args.output_dir}")
    print(f"Meta信息: {Path(args.output_dir).parent}/{Path(args.output_dir).name}_meta_info/{args.dataset_name}/")


if __name__ == '__main__':
    main()
