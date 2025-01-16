import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def multi_scale_conv_block(inputs, filters):
    conv_1x1 = layers.Conv2D(filters, (1,1), activation='relu', padding='same')(inputs)
    conv_3x3 = layers.Conv2D(filters, (3,3), activation='relu', padding='same')(inputs)
    conv_5x5 = layers.Conv2D(filters, (5,5), activation='relu', padding='same')(inputs)
    concat = layers.Concatenate()([conv_1x1, conv_3x3, conv_5x5])
    return concat

def unet_generator(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)

    # 編碼器
    conv1 = multi_scale_conv_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = multi_scale_conv_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = multi_scale_conv_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = multi_scale_conv_block(pool3, 512)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = multi_scale_conv_block(pool4, 1024)

    # 解碼器
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = layers.Concatenate()([up6, conv4])
    conv6 = multi_scale_conv_block(up6, 512)

    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = layers.Concatenate()([up7, conv3])
    conv7 = multi_scale_conv_block(up7, 256)

    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = layers.Concatenate()([up8, conv2])
    conv8 = multi_scale_conv_block(up8, 128)

    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = layers.Concatenate()([up9, conv1])
    conv9 = multi_scale_conv_block(up9, 64)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# 載入已訓練好的生成器權重
generator = unet_generator()
model_weights = 'C:\\Users\\Wayne\\Documents\\GitHub\\Denoise-Project\\training_checkpoints\\generator_epoch_30'
# model_weights = 'best_generator_model.h5'
generator.load_weights(model_weights)  # 請替換為您的實際權重檔路徑

def load_images_from_folder(folder, target_size=(256, 256)):
    images = []
    filenames = sorted(glob.glob(os.path.join(folder, '*.png')))
    if not filenames:
        print(f"警告：在資料夾 {folder} 中未找到任何 PNG 圖像。")
    for filename in filenames:
        img = load_img(filename, color_mode='grayscale', target_size=target_size)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
    return np.array(images), filenames

# 指定雜訊圖像資料夾
noisy_folder = 'D:\\GitHub\\Noise'  # 請根據實際狀況修改
noisy_images, noisy_filenames = load_images_from_folder(noisy_folder)
print(f"共加載 {len(noisy_images)} 張雜訊圖像，開始去雜訊...")

denoised_output_folder = 'C:\\Users\\Wayne\\Documents\\GitHub\\Denoise-Project\\Visualization_result_no'
if not os.path.exists(denoised_output_folder):
    os.makedirs(denoised_output_folder)

# 新增一個資料夾來儲存可視化結果
visualization_output_folder = 'C:\\Users\\Wayne\\Documents\\GitHub\\Denoise-Project\\Visualization_result_no'
if not os.path.exists(visualization_output_folder):
    os.makedirs(visualization_output_folder)

# 對每張雜訊圖像進行去雜訊並顯示及儲存可視化結果
for idx in range(len(noisy_images)):
    noisy_img = noisy_images[idx]
    filename = os.path.basename(noisy_filenames[idx])

    # 增加維度 (batch_size, H, W, C)
    noisy_img_input = np.expand_dims(noisy_img, axis=0)
    denoised_img = generator.predict(noisy_img_input)[0]

    # 將結果轉為uint8並保存
    denoised_img_uint8 = (denoised_img * 255).astype(np.uint8)
    if denoised_img_uint8.ndim == 2:
        denoised_img_uint8 = np.expand_dims(denoised_img_uint8, axis=-1)

    denoised_img_pil = tf.keras.preprocessing.image.array_to_img(denoised_img_uint8, scale=False)
    denoised_img_pil.save(os.path.join(denoised_output_folder, filename))

    # 顯示去雜訊前後的對照
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(noisy_img.squeeze(), cmap='gray')
    plt.title("Noisy")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(denoised_img.squeeze(), cmap='gray')
    plt.title("Denoised")
    plt.axis('off')

    # 儲存可視化結果至指定資料夾
    visualization_path = os.path.join(visualization_output_folder, f"visualization_{idx}.png")
    plt.savefig(visualization_path)
    plt.show()

print("去雜訊處理與可視化完成！")
