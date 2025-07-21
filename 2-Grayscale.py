from PIL import Image
import os

input_dir = 'D:\Denoise\Flickr2K'
output_dir = 'D:\Denoise\Grayscale'

def convert_to_grayscale(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_files = len([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    processed_count = 0
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('L')  # 'L' 模式表示灰階
            img.save(os.path.join(output_dir, filename))
            processed_count += 1
            print(f"已處理 {processed_count}/{total_files}：{filename}")
            
convert_to_grayscale(input_dir, output_dir)
