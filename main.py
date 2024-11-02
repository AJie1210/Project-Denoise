import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# 設定隨機種子以確保結果可重現
np.random.seed(42)
tf.random.set_seed(42)

# 1. 載入資料
def load_images_from_folder(folder, target_size=(128, 128)):
    images = []
    filenames = sorted(glob.glob(os.path.join(folder, '*.png')))  # 假設圖片格式為 PNG
    if not filenames:
        print(f"警告：在資料夾 {folder} 中未找到任何 PNG 圖像。")
    for filename in filenames:
        try:
            img = load_img(filename, color_mode='grayscale', target_size=target_size)
            img_array = img_to_array(img) / 255.0  # 正規化
            images.append(img_array)
        except Exception as e:
            print(f"錯誤：無法載入圖像 {filename}。錯誤訊息：{e}")
    return np.array(images), [os.path.basename(f) for f in filenames]

# 指定雜訊圖像和清晰圖像的資料夾路徑
clean_images_folder = 'D:\\Flicker2K\\Grayscale'
noisy_images_folder = 'D:\\Flicker2K\\Noise'

print("開始載入清晰圖像...")
clean_images, clean_filenames = load_images_from_folder(clean_images_folder)
print(f"載入 {len(clean_images)} 張清晰圖像。")

print("開始載入雜訊圖像...")
noisy_images, noisy_filenames = load_images_from_folder(noisy_images_folder)
print(f"載入 {len(noisy_images)} 張雜訊圖像。")

# 確保清晰圖像和雜訊圖像數量一致
assert len(clean_images) == len(noisy_images), "清晰圖像和雜訊圖像數量不一致。"

# 2. 建立 U-Net 模型
def unet_model(input_size=(128, 128, 1)):
    inputs = layers.Input(input_size)
    
    # 編碼器
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    # 解碼器
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = layers.Concatenate()([up6, conv4])
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = layers.Concatenate()([up7, conv3])
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = layers.Concatenate()([up8, conv2])
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = layers.Concatenate()([up9, conv1])
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

print("建立 U-Net 模型...")
model = unet_model()
model.summary()

# 3. 資料集劃分
from sklearn.model_selection import train_test_split

print("劃分資料集為訓練集和測試集...")
X_train, X_test, y_train, y_test = train_test_split(noisy_images, clean_images, test_size=0.15, random_state=42)
print(f"訓練集大小：{X_train.shape[0]} 張圖像")
print(f"測試集大小：{X_test.shape[0]} 張圖像")

# 4. 建立 TensorFlow Dataset
batch_size = 16
epochs = 100

def create_dataset(noisy, clean, batch_size=16, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((noisy, clean))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

train_dataset = create_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
test_dataset = create_dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

# 5. 編譯模型
print("編譯模型...")
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 6. 訓練模型
print("開始訓練模型...")
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('best_denoising_unet_model.h5', monitor='val_loss', save_best_only=True, mode='min'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
)

# 7. 評估模型
def evaluate_model(model, test_noisy, test_clean):
    psnr_total = 0.0
    ssim_total = 0.0
    num_images = test_noisy.shape[0]
    
    print("開始評估模型性能...")
    for i in range(num_images):
        noisy_img = np.expand_dims(test_noisy[i], axis=0)  # 增加批次維度
        denoised_img = model.predict(noisy_img)
        denoised_img = denoised_img.squeeze()  # 移除批次維度
        clean_img = test_clean[i].squeeze()
        
        # 計算 PSNR
        psnr = peak_signal_noise_ratio(clean_img, denoised_img, data_range=1.0)
        psnr_total += psnr
        
        # 計算 SSIM
        ssim = structural_similarity(clean_img, denoised_img, data_range=1.0)
        ssim_total += ssim
    
    avg_psnr = psnr_total / num_images
    avg_ssim = ssim_total / num_images
    print(f'測試集平均 PSNR: {avg_psnr:.2f} dB')
    print(f'測試集平均 SSIM: {avg_ssim:.4f}')

print("載入最佳模型進行評估...")
best_model = load_model('best_denoising_unet_model.h5')
evaluate_model(best_model, X_test, y_test)

# 8. 使用模型進行去雜訊
def denoise_and_save(model, input_folder, output_folder, target_size=(128, 128)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    filenames = sorted(glob.glob(os.path.join(input_folder, '*.png')))
    if not filenames:
        print(f"警告：在資料夾 {input_folder} 中未找到任何 PNG 圖像。")
    
    for filename in filenames:
        try:
            img = load_img(filename, color_mode='grayscale', target_size=target_size)
            img_array = img_to_array(img) / 255.0  # 正規化
            img_input = np.expand_dims(img_array, axis=0)  # 增加批次維度
            
            denoised = model.predict(img_input)
            denoised = denoised.squeeze()  # 移除批次維度
            denoised = np.clip(denoised, 0.0, 1.0)  # 保持在 [0,1]
            
            # 轉換回 0-255
            denoised_img = (denoised * 255).astype(np.uint8)
            denoised_pil = Image.fromarray(denoised_img, mode='L')
            
            # 儲存去雜訊後的圖像
            save_path = os.path.join(output_folder, os.path.basename(filename))
            denoised_pil.save(save_path)
            print(f'已儲存去雜訊圖像：{save_path}')
        except Exception as e:
            print(f"錯誤：無法處理圖像 {filename}。錯誤訊息：{e}")

# 指定新的含雜訊圖像資料夾和去雜訊後圖像的儲存資料夾
new_noisy_folder = 'D:\\Flicker2K\\NewNoise'
denoised_output_folder = 'D:\\Flicker2K\\Clean'

print("開始對新圖像進行去雜訊處理...")
denoise_and_save(best_model, new_noisy_folder, denoised_output_folder)
print("去雜訊處理完成。")

# 9. 可視化訓練過程
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# 10. 可視化去雜訊效果（選擇部分圖像）
def visualize_denoising(model, noisy_images, clean_images, num_samples=5):
    indices = np.random.choice(len(noisy_images), num_samples, replace=False)
    for i in indices:
        noisy_img = np.expand_dims(noisy_images[i], axis=0)  # 增加批次維度
        denoised_img = model.predict(noisy_img)
        denoised_img = denoised_img.squeeze()
        clean_img = clean_images[i].squeeze()
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.title('Noise Image')
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title('Denoise Image')
        plt.imshow(denoised_img, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title('Clear Image')
        plt.imshow(clean_img, cmap='gray')
        plt.axis('off')
        
        plt.show()

print("可視化去雜訊效果...")
visualize_denoising(best_model, X_test, y_test, num_samples=5)
print("Complete all step")
