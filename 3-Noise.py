import cv2
import numpy as np
import os
import random
from tqdm import tqdm  # 需要安裝 tqdm：pip install tqdm

# 定義添加雜訊的函數
def add_noise(image):
    """
    向圖像添加隨機噪聲。
    :param image: 輸入的灰階圖像 (numpy array)
    :return: 含噪聲的圖像
    """
    # 隨機選擇雜訊類型
    noise_type = random.choice(['gaussian', 'salt_pepper', 'uniform'])
    
    if noise_type == 'gaussian':
        # 高斯雜訊
        mean = 0
        sigma = random.randint(5, 30)
        gauss = np.random.normal(mean, sigma, image.shape).astype(np.int16)
        noisy_image = image.astype(np.int16) + gauss
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    elif noise_type == 'salt_pepper':
        # 椒鹽雜訊
        s_vs_p = 0.5
        amount = random.uniform(0.001, 0.03)
        noisy_image = np.copy(image)
        
        # 添加鹽雜訊
        num_salt = int(amount * image.size * s_vs_p)
        coords_salt = [np.random.randint(0, i, num_salt) for i in image.shape]
        noisy_image[tuple(coords_salt)] = 255
        
        # 添加胡椒雜訊
        num_pepper = int(amount * image.size * (1 - s_vs_p))
        coords_pepper = [np.random.randint(0, i, num_pepper) for i in image.shape]
        noisy_image[tuple(coords_pepper)] = 0

    else:  # uniform noise
        # 均勻分佈雜訊
        noise_level = random.randint(0, 30)
        uniform_noise = np.random.randint(-noise_level, noise_level + 1, image.shape).astype(np.int16)
        noisy_image = image.astype(np.int16) + uniform_noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

# 資料夾路徑
input_folder = 'D:\\GitHub\\Grayscale'
output_folder = 'D:\\GitHub\\Noise'

# 創建輸出資料夾（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 瀏覽資料夾中的所有圖像並處理
error_files = []  # 用於記錄處理失敗的文件
file_list = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]

for filename in tqdm(file_list, desc="Processing Images", unit="file"):
    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 確保圖像成功加載
    if image is None:
        print(f'警告：無法加載圖像 {filename}')
        error_files.append(filename)
        continue

    try:
        # 添加雜訊
        noisy_image = add_noise(image)

        # 保存添加雜訊的圖像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, noisy_image)

    except Exception as e:
        print(f'錯誤處理圖像 {filename}: {e}')
        error_files.append(filename)

# 輸出處理結果
if error_files:
    print("\n以下圖像處理失敗：")
    for error_file in error_files:
        print(f"- {error_file}")
else:
    print("\n所有圖像處理完成！")
