import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG19
from sklearn.model_selection import train_test_split
import time

# 匯入評估指標函數
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_squared_error

# 設定隨機種子確保結果可重現
np.random.seed(42)
tf.random.set_seed(42)

# 1. 載入資料 (改為256x256)
def load_images_from_folder(folder, target_size=(256, 256)):
    images = []
    filenames = sorted(glob.glob(os.path.join(folder, '*.png')))
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

# 資料夾路徑（請根據需要自行修改）
Gray_images_folder = 'D:\\Flicker2K\\archive\\Grayscale'
noisy_images_folder = 'D:\\Flicker2K\\archive\\Noise'

print("開始載入清晰圖像...")
clean_images, clean_filenames = load_images_from_folder(Gray_images_folder)
print(f"載入 {len(clean_images)} 張清晰圖像。")

print("開始載入雜訊圖像...")
noisy_images, noisy_filenames = load_images_from_folder(noisy_images_folder)
print(f"載入 {len(noisy_images)} 張雜訊圖像。")

if len(clean_images) == 0 or len(noisy_images) == 0:
    raise ValueError("資料集中沒有可用的圖像，請檢查資料夾路徑和格式！")
assert len(clean_images) == len(noisy_images), "清晰圖像和雜訊圖像數量不一致。"

# 2. 建立多尺度卷積塊
def multi_scale_conv_block(inputs, filters):
    conv_1x1 = layers.Conv2D(filters, (1,1), activation='relu', padding='same')(inputs)
    conv_3x3 = layers.Conv2D(filters, (3,3), activation='relu', padding='same')(inputs)
    conv_5x5 = layers.Conv2D(filters, (5,5), activation='relu', padding='same')(inputs)
    concat = layers.Concatenate()([conv_1x1, conv_3x3, conv_5x5])
    return concat

# 3. 建立生成器模型（U-Net 結構 + 多尺度卷積）
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

# 4. 建立判別器模型(256x256)
def discriminator_model(input_shape=(256,256,1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3,3), strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(128, (3,3), strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(256, (3,3), strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(512, (3,3), strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs, x)
    return model

print("建立生成器和判別器模型...")
generator = unet_generator()
discriminator = discriminator_model()

# 方法一：初始化模型權重 (Forward Pass 一次)
dummy_input = tf.ones((1,256,256,1))
_ = generator(dummy_input, training=False)

print("劃分資料集為訓練集、驗證集和測試集...")
# 將資料劃分為訓練集、驗證集和測試集 (70% 訓練, 15% 驗證, 15% 測試)
X_train, X_temp, y_train, y_temp = train_test_split(noisy_images, clean_images, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"訓練集大小：{X_train.shape[0]} 張圖像")
print(f"驗證集大小：{X_val.shape[0]} 張圖像")
print(f"測試集大小：{X_test.shape[0]} 張圖像")

# 6. 建立 TensorFlow Dataset
batch_size = 8
epochs = 100

def create_dataset(noisy, clean, batch_size=8, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((noisy, clean))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
val_dataset = create_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)
test_dataset = create_dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

# 7. 定義感知損失（使用 VGG19）
print("定義感知損失...")
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
vgg_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
vgg_model.trainable = False

# 8. 定義損失函數和優化器
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
mse_loss = tf.keras.losses.MeanSquaredError()
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

lambda_value = 10.0

# 9. 定義訓練和驗證步驟
@tf.function
def train_step(noisy_images, clean_images):
    with tf.GradientTape() as disc_tape:
        generated_images = generator(noisy_images, training=True)
        real_output = discriminator(clean_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        disc_loss_real = binary_cross_entropy(tf.ones_like(real_output), real_output)
        disc_loss_fake = binary_cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = disc_loss_real + disc_loss_fake

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noisy_images, training=True)
        fake_output = discriminator(generated_images, training=False)

        adv_loss = binary_cross_entropy(tf.ones_like(fake_output), fake_output)

        y_true_rgb = tf.image.grayscale_to_rgb(clean_images)
        y_pred_rgb = tf.image.grayscale_to_rgb(generated_images)
        y_true_features = vgg_model(y_true_rgb)
        y_pred_features = vgg_model(y_pred_rgb)
        perc_loss = tf.reduce_mean(tf.square(y_true_features - y_pred_features))

        gen_loss = adv_loss + lambda_value * perc_loss
        mse = mse_loss(clean_images, generated_images)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return disc_loss, gen_loss, adv_loss, perc_loss, mse

@tf.function
def val_step(noisy_images, clean_images):
    generated_images = generator(noisy_images, training=False)
    fake_output = discriminator(generated_images, training=False)

    adv_loss = binary_cross_entropy(tf.ones_like(fake_output), fake_output)

    y_true_rgb = tf.image.grayscale_to_rgb(clean_images)
    y_pred_rgb = tf.image.grayscale_to_rgb(generated_images)
    y_true_features = vgg_model(y_true_rgb)
    y_pred_features = vgg_model(y_pred_rgb)
    perc_loss = tf.reduce_mean(tf.square(y_true_features - y_pred_features))

    gen_loss = adv_loss + lambda_value * perc_loss
    mse = mse_loss(clean_images, generated_images)

    return gen_loss, adv_loss, perc_loss, mse

# 10. 開始訓練
print("開始訓練模型...")
EPOCHS = epochs

checkpoint_dir = './training_checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

generator_checkpoint_prefix = os.path.join(checkpoint_dir, "generator_epoch_{epoch}")
discriminator_checkpoint_prefix = os.path.join(checkpoint_dir, "discriminator_epoch_{epoch}")

train_disc_losses = []
train_gen_losses = []
train_mse_losses = []

val_gen_losses = []
val_mse_losses = []

patience = 5
best_val_loss = np.inf
patience_counter = 0

best_generator_weights = None
best_discriminator_weights = None

for epoch in range(EPOCHS):
    print(f"開始第 {epoch+1} 個 Epoch 訓練...")
    start = time.time()

    train_steps = 0
    epoch_train_disc_loss = 0
    epoch_train_gen_loss = 0
    epoch_train_mse_loss = 0

    for step, (noisy_batch, clean_batch) in enumerate(train_dataset):
        disc_loss, gen_loss, adv_loss, perc_loss, mse = train_step(noisy_batch, clean_batch)
        epoch_train_disc_loss += disc_loss.numpy()
        epoch_train_gen_loss += gen_loss.numpy()
        epoch_train_mse_loss += mse.numpy()
        train_steps += 1

        if step % 100 == 0:
            print(f"Epoch {epoch+1}, 訓練步驟 {step}, 判別器損失: {disc_loss.numpy():.4f}, 生成器損失: {gen_loss.numpy():.4f}, MSE: {mse.numpy():.4f}")

    val_steps = 0
    epoch_val_gen_loss = 0
    epoch_val_mse_loss = 0

    for noisy_batch, clean_batch in val_dataset:
        gen_loss, adv_loss, perc_loss, mse = val_step(noisy_batch, clean_batch)
        epoch_val_gen_loss += gen_loss.numpy()
        epoch_val_mse_loss += mse.numpy()
        val_steps += 1

    avg_train_disc_loss = epoch_train_disc_loss / train_steps
    avg_train_gen_loss = epoch_train_gen_loss / train_steps
    avg_train_mse_loss = epoch_train_mse_loss / train_steps
    avg_val_gen_loss = epoch_val_gen_loss / val_steps
    avg_val_mse_loss = epoch_val_mse_loss / val_steps

    train_disc_losses.append(avg_train_disc_loss)
    train_gen_losses.append(avg_train_gen_loss)
    train_mse_losses.append(avg_train_mse_loss)

    val_gen_losses.append(avg_val_gen_loss)
    val_mse_losses.append(avg_val_mse_loss)

    # 儲存權重（此處已經有初始的Forward Pass，避免初始化問題）
    generator_save_path = generator_checkpoint_prefix.format(epoch=epoch+1)
    discriminator_save_path = discriminator_checkpoint_prefix.format(epoch=epoch+1)
    generator.save_weights(generator_save_path)
    discriminator.save_weights(discriminator_save_path)
    print(f"已保存生成器至 {generator_save_path}")
    print(f"已保存判別器至 {discriminator_save_path}")

    print(f"第 {epoch+1} 個 Epoch 花費時間：{time.time()-start:.2f} 秒")
    print(f"訓練生成器損失：{avg_train_gen_loss:.4f}, 訓練 MSE：{avg_train_mse_loss:.4f}")
    print(f"驗證生成器損失：{avg_val_gen_loss:.4f}, 驗證 MSE：{avg_val_mse_loss:.4f}")

    # 檢查早停條件
    if avg_val_gen_loss < best_val_loss:
        best_val_loss = avg_val_gen_loss
        patience_counter = 0
        best_generator_weights = generator.get_weights()
        best_discriminator_weights = discriminator.get_weights()
        print("驗證損失降低，保存當前模型權重為最佳模型。")
    else:
        patience_counter += 1
        print(f"驗證損失未改善，早停計數：{patience_counter}/{patience}")
        if patience_counter >= patience:
            print("早停條件滿足，停止訓練。")
            generator.set_weights(best_generator_weights)
            discriminator.set_weights(best_discriminator_weights)
            break

# 11. 繪製損失曲線
def plot_losses(train_gen_losses, val_gen_losses, train_mse_losses, val_mse_losses):
    epochs_range = range(1, len(train_gen_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Plot Generator Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_gen_losses, label='Training Generator Loss')
    plt.plot(epochs_range, val_gen_losses, label='Validation Generator Loss')
    plt.title('Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Mean Squared Error (MSE)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_mse_losses, label='Training MSE')
    plt.plot(epochs_range, val_mse_losses, label='Validation MSE')
    plt.title('Mean Squared Error (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_validation_losses.png')
    plt.show()

# Call the function to plot the losses
plot_losses(train_gen_losses, val_gen_losses, train_mse_losses, val_mse_losses)

# 12. 可視化訓練結果
def plot_generated_images(generator, dataset, num_images=5, epoch=None):
    for noisy_images, clean_images in dataset.take(1):
        generated_images = generator.predict(noisy_images[:num_images])

        for i in range(num_images):
            noisy_img = noisy_images[i].numpy().squeeze()
            denoised_img = generated_images[i].squeeze()
            clean_img = clean_images[i].numpy().squeeze()

            psnr = compare_psnr(clean_img, denoised_img, data_range=1.0)
            ssim = compare_ssim(clean_img, denoised_img, data_range=1.0)
            mse = mean_squared_error(clean_img.flatten(), denoised_img.flatten())

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(noisy_img, cmap='gray')
            plt.title("Noisy Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(denoised_img, cmap='gray')
            plt.title("Denoised Image")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(clean_img, cmap='gray')
            plt.title("Clean Image")
            plt.axis('off')

            plt.suptitle(f'PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, MSE: {mse:.4f}')
            plt.tight_layout()
            if epoch is not None:
                plt.savefig(f'generated_image_{epoch}_index_{i}.png')
            plt.show()

plot_generated_images(generator, val_dataset, num_images=5)

# 13. 定義評估指標計算函數
def evaluate_metrics(generator, dataset):
    psnr_list = []
    ssim_list = []
    mse_list = []
    for noisy_batch, clean_batch in dataset:
        generated_images = generator(noisy_batch, training=False)
        for i in range(generated_images.shape[0]):
            denoised = generated_images[i].numpy().squeeze()
            clean = clean_batch[i].numpy().squeeze()
            psnr = compare_psnr(clean, denoised, data_range=1.0)
            ssim = compare_ssim(clean, denoised, data_range=1.0)
            mse_val = mean_squared_error(clean.flatten(), denoised.flatten())
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            mse_list.append(mse_val)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_mse = np.mean(mse_list)
    return avg_psnr, avg_ssim, avg_mse

# 14. 計算並顯示驗證集和測試集的平均 PSNR, SSIM, MSE
print("計算驗證集和測試集的平均 PSNR, SSIM, MSE...")
avg_val_psnr, avg_val_ssim, avg_val_mse = evaluate_metrics(generator, val_dataset)
avg_test_psnr, avg_test_ssim, avg_test_mse = evaluate_metrics(generator, test_dataset)

print("\n模型訓練完成！以下是驗證集和測試集的平均評估指標：")
print(f"驗證集 - 平均 PSNR: {avg_val_psnr:.2f} dB")
print(f"驗證集 - 平均 SSIM: {avg_val_ssim:.4f}")
print(f"驗證集 - 平均 MSE: {avg_val_mse:.4f}")

print(f"測試集 - 平均 PSNR: {avg_test_psnr:.2f} dB")
print(f"測試集 - 平均 SSIM: {avg_test_ssim:.4f}")
print(f"測試集 - 平均 MSE: {avg_test_mse:.4f}")
