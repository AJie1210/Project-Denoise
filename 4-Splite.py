import random
import shutil
import os

def split_dataset(input_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    random.shuffle(files)
    total = len(files)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    splits = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

    for split, split_files in splits.items():
        split_dir = os.path.join(output_base_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        for filename in split_files:
            shutil.copy(os.path.join(input_dir, filename), os.path.join(split_dir, filename))

# 設定路徑
input_gray_dir = 'D:\\Flicker2K\\Noise'
output_base_dir = 'D:\\Flicker2K\\Splits'

# 劃分數據集
split_dataset(input_gray_dir, 
             train_dir=os.path.join(output_base_dir, 'train'),
             val_dir=os.path.join(output_base_dir, 'val'),
             test_dir=os.path.join(output_base_dir, 'test'))
