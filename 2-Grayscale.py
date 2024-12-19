from PIL import Image
import os

input_dir = 'D:\\Flicker2K\\archive\\Flickr2K'
output_dir = 'D:\\Flicker2K\\archive\\Grayscale'

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
