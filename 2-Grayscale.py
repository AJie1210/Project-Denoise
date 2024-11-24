from PIL import Image
import os

input_dir = 'C:\\Users\\ytes6\\Downloads\\21586188\\Set14\\Set14\\image_SRF_2'
output_dir = 'D:\\Flicker2K\\DenoiseImage\\OriginalImages'

def convert_to_grayscale(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('L')  # 'L' 模式表示灰階
            img.save(os.path.join(output_dir, filename))

# 示例使用
convert_to_grayscale(input_dir, output_dir)
