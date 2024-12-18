import cv2
import numpy as np
import os
import random

# 定義添加雜訊的函數
def add_noise(image):
    # 隨機選擇添加高斯雜訊、椒鹽雜訊或均勻分佈雜訊
    noise_type = random.choice(['gaussian', 'salt_pepper', 'uniform'])
    
    if noise_type == 'gaussian':
        # 隨機選擇Gaussian雜訊強度
        mean = 0
        sigma = random.randint(5, 30)  # 可依需求調整範圍
        gauss = np.random.normal(mean, sigma, image.shape).astype(np.int16)
        
        # 將原圖與雜訊相加後裁剪回合理範圍(0-255)
        noisy_image = image.astype(np.int16) + gauss
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    elif noise_type == 'salt_pepper':
        # 隨機選擇鹽胡椒雜訊強度
        s_vs_p = 0.5
        amount = random.uniform(0.001, 0.03)  # 調整鹽胡椒雜訊量的範圍
        noisy_image = np.copy(image)
        
        # 添加鹽雜訊
        num_salt = np.ceil(amount * image.size * s_vs_p).astype(int)
        coords_salt = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        noisy_image[coords_salt] = 255
        
        # 添加胡椒雜訊
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p)).astype(int)
        coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        noisy_image[coords_pepper] = 0

    else: # uniform noise
        # 隨機選擇均勻分佈雜訊強度
        noise_level = random.randint(0, 30)  # 調整此範圍決定雜訊幅度
        # 在 [-noise_level, noise_level] 區間內產生均勻分佈雜訊
        uniform_noise = np.random.randint(-noise_level, noise_level+1, image.shape).astype(np.int16)
        noisy_image = image.astype(np.int16) + uniform_noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
    return noisy_image

# 指定資料夾路徑
input_folder = 'D:\\Flicker2K\\Grayscale'
output_folder = 'D:\\Flicker2K\\Noise'
# input_folder = 'D:\\Flicker2K\\DenoiseImage\\OriginalImages'
# output_folder = 'D:\\Flicker2K\\DenoiseImage\\NoisyImages'

# 創建輸出資料夾（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 瀏覽資料夾中的所有圖像
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):  # 根據您的圖像格式進行調整
        # 加載圖像(以灰階讀取)
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 確保圖像成功加載
        if image is None:
            print(f'警告：無法加載圖像 {filename}')
            continue

        # 只添加雜訊 (高斯、椒鹽、均勻)
        noisy_image = add_noise(image)

        # 保存添加雜訊的圖像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, noisy_image)

        print(f'Processed {filename} and saved to {output_folder}')
