import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, Model
from keras.utils import load_img, img_to_array
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_squared_error

# 定義多尺度卷積塊
def multi_scale_conv_block(inputs, filters):
    conv_1x1 = layers.Conv2D(filters, (1,1), activation='relu', padding='same')(inputs)
    conv_3x3 = layers.Conv2D(filters, (3,3), activation='relu', padding='same')(inputs)
    conv_5x5 = layers.Conv2D(filters, (5,5), activation='relu', padding='same')(inputs)
    concat = layers.Concatenate()([conv_1x1, conv_3x3, conv_5x5])
    return concat

# 定義生成器模型
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

print("加載生成器模型...")
generator = unet_generator()
model_weights = 'C:\\Users\\ytes6\\OneDrive\\文件\\GitHub\\Project-Denoise\\training_checkpoints\\generator_epoch_38'
# model_weights = 'best_generator_model.h5'
generator.load_weights(model_weights)
print(f"已加載模型權重：{model_weights}")

def load_images_from_folder(folder, target_size=(256, 256)):
    images = []
    filenames = sorted(glob.glob(os.path.join(folder, '*.png')))
    if not filenames:
        print(f"警告：在資料夾 {folder} 中未找到任何 PNG 圖像。")
    for filename in filenames:
        try:
            img = load_img(filename, color_mode='grayscale', target_size=target_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
        except Exception as e:
            print(f"錯誤：無法加載圖像 {filename}。錯誤信息：{e}")
    return np.array(images), filenames


original_folder = 'D:\\Denoise\\Final_TestImage(Grayscale)'
noisy_folder = 'D:\\Denoise\\Final_TestImage(Noise Image)'
denoised_output_folder = 'C:\\Users\\ytes6\\OneDrive\\文件\\GitHub\\Project-Denoise\\Visualization_result'

if not os.path.exists(denoised_output_folder):
    os.makedirs(denoised_output_folder)

print("開始加載雜訊圖像...")
noisy_images, noisy_filenames = load_images_from_folder(noisy_folder)
print(f"共加載 {len(noisy_images)} 張雜訊圖像。")

have_original = False
if os.path.exists(original_folder):
    print("開始加載原始圖像...")
    original_images, original_filenames = load_images_from_folder(original_folder)
    print(f"共加載 {len(original_images)} 張原始圖像。")
    have_original = True
    assert len(original_images) == len(noisy_images), "原始圖像和雜訊圖像數量不一致。"

print("開始圖像去雜訊處理...")
psnr_list = []
ssim_list = []
mse_list = []
num_images_to_visualize = 5

# 新增可視化存檔路徑
visualization_save_folder = 'C:\\Users\\ytes6\\OneDrive\\文件\\GitHub\\Project-Denoise\\Visualization_result'
if not os.path.exists(visualization_save_folder):
    os.makedirs(visualization_save_folder)

for idx in range(len(noisy_images)):
    noisy_img = noisy_images[idx]
    filename = os.path.basename(noisy_filenames[idx])

    noisy_img_input = np.expand_dims(noisy_img, axis=0)
    denoised_img = generator.predict(noisy_img_input)[0] # 生成器預測原始影像

    denoised_img_uint8 = (denoised_img * 255).astype(np.uint8)
    if denoised_img_uint8.ndim == 2:
        denoised_img_uint8 = np.expand_dims(denoised_img_uint8, axis=-1)

    denoised_img_pil = tf.keras.preprocessing.image.array_to_img(denoised_img_uint8, scale=False)
    denoised_img_pil.save(os.path.join(denoised_output_folder, filename))

    if have_original:
        clean_img = original_images[idx]
        psnr = compare_psnr(clean_img.squeeze(), denoised_img.squeeze(), data_range=1.0)
        ssim = compare_ssim(clean_img.squeeze(), denoised_img.squeeze(), data_range=1.0)
        mse = mean_squared_error(clean_img.flatten(), denoised_img.flatten())

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        mse_list.append(mse)

        if idx < num_images_to_visualize:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(noisy_img.reshape(256,256), cmap='gray')
            plt.title("Noisy")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(denoised_img.reshape(256,256), cmap='gray')
            plt.title("Denoised")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(clean_img.reshape(256,256), cmap='gray')
            plt.title("Original")
            plt.axis('off')

            plt.suptitle(f'PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, MSE: {mse:.4f}')
            plt.tight_layout()

            save_visualization_path = os.path.join(visualization_save_folder, f"visualization_{idx}.png")
            plt.savefig(save_visualization_path)
            plt.show()

print("所有圖像處理完成。")

if have_original and psnr_list:
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_mse = np.mean(mse_list)

    print(f"平均 PSNR: {avg_psnr:.2f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    print(f"平均 MSE: {avg_mse:.4f}")
