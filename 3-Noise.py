import cv2
import numpy as np
import os
import random

# 定義添加雜訊的函數
def add_noise(image):
    # 隨機選擇添加高斯雜訊或椒鹽雜訊
    noise_type = random.choice(['gaussian', 'salt_pepper'])
    
    if noise_type == 'gaussian':
        # 添加高斯雜訊
        mean = 0
        sigma = 25
        gauss = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, gauss)
    else:
        # 添加椒鹽雜訊
        s_vs_p = 0.5
        amount = 0.004
        noisy_image = np.copy(image)
        
        # 添加鹽雜訊
        num_salt = np.ceil(amount * image.size * s_vs_p).astype(int)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        noisy_image[coords] = 255
        
        # 添加胡椒雜訊
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p)).astype(int)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        noisy_image[coords] = 0
        
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
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 確保圖像成功加載
        if image is None:
            print(f'警告：無法加載圖像 {filename}')
            continue

        # 添加雜訊
        noisy_image = add_noise(image)

        # 保存添加雜訊的圖像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, noisy_image)

        print(f'Processed {filename} and saved to {output_folder}')
