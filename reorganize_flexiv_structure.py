#!/usr/bin/env python3
"""
重新组织Flexiv数据集结构，使其与DROID标准结构一致

从：
flexiv_data/
├── 0/latent_videos/0/
├── 1/latent_videos/0/
└── annotation/train/0.json

到：
flexiv_data/
├── latent_videos/
│   ├── train/0/0/  ← episode/camera
│   └── val/16/0/
└── annotation/
    ├── train/0.json
    └── val/16.json
"""

import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm

def reorganize_flexiv_dataset(base_dir):
    """重新组织Flexiv数据集结构"""
    base_path = Path(base_dir)
    
    # 创建新的目录结构
    new_latent_dir = base_path / "latent_videos_new"
    new_latent_dir.mkdir(exist_ok=True)
    (new_latent_dir / "train").mkdir(exist_ok=True)
    (new_latent_dir / "val").mkdir(exist_ok=True)
    
    print("🔄 重新组织Flexiv数据集结构...")
    print(f"   源目录: {base_path}")
    
    # 读取train_sample.json和val_sample.json来确定哪些是训练集，哪些是验证集
    meta_info_path = Path("/root/workspace/chenyj36@xiaopeng.com/Ctrl-World/dataset_example/flexiv_data_meta_info/flexiv_1113")
    
    train_episodes = set()
    val_episodes = set()
    
    # 读取训练集episodes
    train_sample_file = meta_info_path / "train_sample.json"
    if train_sample_file.exists():
        with open(train_sample_file) as f:
            train_samples = json.load(f)
            for sample in train_samples:
                train_episodes.add(str(sample['episode_id']))
    
    # 读取验证集episodes
    val_sample_file = meta_info_path / "val_sample.json"
    if val_sample_file.exists():
        with open(val_sample_file) as f:
            val_samples = json.load(f)
            for sample in val_samples:
                val_episodes.add(str(sample['episode_id']))
    
    print(f"   训练集episodes: {sorted(train_episodes)}")
    print(f"   验证集episodes: {sorted(val_episodes)}")
    
    # 遍历所有episode目录
    episode_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()]
    
    moved_count = 0
    updated_annotations = {'train': [], 'val': []}
    
    for episode_dir in tqdm(sorted(episode_dirs, key=lambda x: int(x.name)), desc="移动latent文件"):
        episode_id = episode_dir.name
        
        # 确定是train还是val
        if episode_id in train_episodes:
            split = 'train'
        elif episode_id in val_episodes:
            split = 'val'
        else:
            print(f"⚠️  警告: Episode {episode_id} 不在train或val中，跳过")
            continue
        
        # 检查latent_videos目录
        old_latent_dir = episode_dir / "latent_videos"
        if not old_latent_dir.exists():
            print(f"⚠️  警告: {old_latent_dir} 不存在，跳过")
            continue
        
        # 创建新的episode目录
        new_episode_dir = new_latent_dir / split / episode_id
        new_episode_dir.mkdir(parents=True, exist_ok=True)
        
        # 移动所有camera目录
        for camera_dir in old_latent_dir.iterdir():
            if camera_dir.is_dir():
                camera_id = camera_dir.name
                new_camera_dir = new_episode_dir / camera_id
                
                # 复制目录
                if new_camera_dir.exists():
                    shutil.rmtree(new_camera_dir)
                shutil.copytree(camera_dir, new_camera_dir)
                moved_count += 1
    
    print(f"\n✅ 移动了 {moved_count} 个camera目录")
    
    # 更新annotation文件中的路径
    print("\n🔄 更新annotation文件中的路径...")
    annotation_dir = base_path / "annotation"
    
    for split in ['train', 'val']:
        split_dir = annotation_dir / split
        if not split_dir.exists():
            continue
        
        for ann_file in split_dir.glob("*.json"):
            episode_id = ann_file.stem
            
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            # 更新latent_videos路径
            if 'latent_videos' in data:
                for i, latent_info in enumerate(data['latent_videos']):
                    old_path = latent_info['latent_video_path']
                    # 从 "0/latent_videos/0/0.pt" 改为 "latent_videos/train/0/0/0.pt"
                    # 解析路径
                    parts = old_path.split('/')
                    if len(parts) >= 4:
                        ep_id = parts[0]
                        camera_id = parts[2]
                        frame_file = parts[3]
                        new_path = f"latent_videos/{split}/{ep_id}/{camera_id}/{frame_file}"
                        data['latent_videos'][i]['latent_video_path'] = new_path
            
            # 保存更新后的annotation
            with open(ann_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            updated_annotations[split].append(episode_id)
    
    print(f"✅ 更新了 {len(updated_annotations['train'])} 个训练集annotation")
    print(f"✅ 更新了 {len(updated_annotations['val'])} 个验证集annotation")
    
    # 重命名目录
    print("\n🔄 替换旧的latent_videos目录...")
    old_latent_backup = base_path / "latent_videos_old_backup"
    old_latent = base_path / "latent_videos"
    
    # 如果存在旧的latent_videos，先备份
    if old_latent.exists():
        if old_latent_backup.exists():
            shutil.rmtree(old_latent_backup)
        shutil.move(str(old_latent), str(old_latent_backup))
        print(f"   旧latent_videos已备份到: {old_latent_backup}")
    
    # 重命名新目录
    shutil.move(str(new_latent_dir), str(old_latent))
    print(f"✅ 新结构已生效: {old_latent}")
    
    # 清理旧的episode目录（保留annotation子目录）
    print("\n🔄 清理旧的episode目录...")
    cleaned = 0
    for episode_dir in episode_dirs:
        # 删除latent_videos子目录
        latent_subdir = episode_dir / "latent_videos"
        if latent_subdir.exists():
            shutil.rmtree(latent_subdir)
        
        # 删除annotation子目录（已经有统一的annotation了）
        ann_subdir = episode_dir / "annotation"
        if ann_subdir.exists():
            shutil.rmtree(ann_subdir)
        
        # 如果episode目录为空，删除它
        if not any(episode_dir.iterdir()):
            episode_dir.rmdir()
            cleaned += 1
    
    print(f"✅ 清理了 {cleaned} 个空episode目录")
    
    # 验证新结构
    print("\n" + "="*50)
    print("📊 验证新结构:")
    print("="*50)
    
    train_latent_dir = old_latent / "train"
    val_latent_dir = old_latent / "val"
    
    train_episodes_count = len(list(train_latent_dir.iterdir())) if train_latent_dir.exists() else 0
    val_episodes_count = len(list(val_latent_dir.iterdir())) if val_latent_dir.exists() else 0
    
    print(f"✅ latent_videos/train/: {train_episodes_count} episodes")
    print(f"✅ latent_videos/val/: {val_episodes_count} episodes")
    print(f"✅ annotation/train/: {len(updated_annotations['train'])} files")
    print(f"✅ annotation/val/: {len(updated_annotations['val'])} files")
    
    # 显示示例路径
    print("\n📝 示例路径:")
    if train_latent_dir.exists():
        example_files = list(train_latent_dir.glob("*/0/*.pt"))[:2]
        for f in example_files:
            rel_path = f.relative_to(base_path)
            print(f"   {rel_path}")
    
    print("\n🎉 数据集重组完成！")
    print(f"   ⚠️  旧数据备份在: {old_latent_backup}")
    print(f"   如确认无问题，可手动删除: rm -rf {old_latent_backup}")

if __name__ == "__main__":
    base_dir = "/root/workspace/chenyj36@xiaopeng.com/Ctrl-World/dataset_example/flexiv_data"
    reorganize_flexiv_dataset(base_dir)

