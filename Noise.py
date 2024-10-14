import cv2
import numpy as np
import os

# 定義添加雜訊的函數
def add_noise(image):
    # 隨機生成雜訊
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

# 指定資料夾路徑
input_folder = 'D:\\Flicker2K\\Grayscale'
output_folder = 'D:\\Flicker2K\\Noise'

# 創建輸出資料夾（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 瀏覽資料夾中的所有圖像
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # 根據您的圖像格式進行調整
        # 加載圖像
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        # 添加雜訊
        noisy_image = add_noise(image)

        # 保存添加雜訊的圖像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, noisy_image)

        print(f'Processed {filename} and saved to {output_folder}')
