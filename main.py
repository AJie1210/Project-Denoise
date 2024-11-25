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

# 新增匯入評估指標的函數
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
# 從 sklearn.metrics 匯入 mean_squared_error
from sklearn.metrics import mean_squared_error

# 設定隨機種子以確保結果可重現
np.random.seed(42)
tf.random.set_seed(42)

# 1. 載入資料
def load_images_from_folder(folder, target_size=(256, 256)):
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

# 指定資料夾路徑
Gray_images_folder = 'D:\\Flicker2K\\Grayscale'
noisy_images_folder = 'D:\\Flicker2K\\Noise'

# 載入清晰圖像
print("開始載入清晰圖像...")
clean_images, clean_filenames = load_images_from_folder(Gray_images_folder)
print(f"載入 {len(clean_images)} 張清晰圖像。")

# 載入雜訊圖像
print("開始載入雜訊圖像...")
noisy_images, noisy_filenames = load_images_from_folder(noisy_images_folder)
print(f"載入 {len(noisy_images)} 張雜訊圖像。")

# 確保清晰圖像和雜訊圖像數量一致
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

# 3. 建立生成器模型（U-Net 結構，加入多尺度卷積）
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

# 4. 建立判別器模型
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

# 建立生成器和判別器
print("建立生成器和判別器模型...")
generator = unet_generator()
discriminator = discriminator_model()

# 5. 資料集劃分
print("劃分資料集為訓練集和驗證集...")
X_train, X_val, y_train, y_val = train_test_split(noisy_images, clean_images, test_size=0.15, random_state=42)
print(f"訓練集大小：{X_train.shape[0]} 張圖像")
print(f"驗證集大小：{X_val.shape[0]} 張圖像")

# 6. 建立 TensorFlow Dataset
batch_size = 16
epochs = 100  # 設定一個較大的最大 Epoch 數量

def create_dataset(noisy, clean, batch_size=16, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((noisy, clean))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
val_dataset = create_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)

# 7. 定義感知損失（使用 VGG19）
print("定義感知損失...")
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
vgg_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
vgg_model.trainable = False

# 8. 定義損失函數和優化器
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
# 將 MAE 替換為 MSE
mse_loss = tf.keras.losses.MeanSquaredError()
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

lambda_value = 10.0  # 平衡感知損失和對抗損失的超參數

# 9. 定義訓練和驗證步驟
@tf.function
def train_step(noisy_images, clean_images):
    # 訓練判別器
    with tf.GradientTape() as disc_tape:
        generated_images = generator(noisy_images, training=True)

        real_output = discriminator(clean_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # 判別器損失
        disc_loss_real = binary_cross_entropy(tf.ones_like(real_output), real_output)
        disc_loss_fake = binary_cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = disc_loss_real + disc_loss_fake

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # 訓練生成器
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noisy_images, training=True)

        fake_output = discriminator(generated_images, training=False)

        # 生成器對抗損失
        adv_loss = binary_cross_entropy(tf.ones_like(fake_output), fake_output)

        # 感知損失
        y_true_rgb = tf.image.grayscale_to_rgb(clean_images)
        y_pred_rgb = tf.image.grayscale_to_rgb(generated_images)
        y_true_features = vgg_model(y_true_rgb)
        y_pred_features = vgg_model(y_pred_rgb)
        perc_loss = tf.reduce_mean(tf.square(y_true_features - y_pred_features))

        # 總生成器損失
        gen_loss = adv_loss + lambda_value * perc_loss

        # MSE 損失
        mse = mse_loss(clean_images, generated_images)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return disc_loss, gen_loss, adv_loss, perc_loss, mse

@tf.function
def val_step(noisy_images, clean_images):
    generated_images = generator(noisy_images, training=False)

    fake_output = discriminator(generated_images, training=False)

    # 生成器對抗損失
    adv_loss = binary_cross_entropy(tf.ones_like(fake_output), fake_output)

    # 感知損失
    y_true_rgb = tf.image.grayscale_to_rgb(clean_images)
    y_pred_rgb = tf.image.grayscale_to_rgb(generated_images)
    y_true_features = vgg_model(y_true_rgb)
    y_pred_features = vgg_model(y_pred_rgb)
    perc_loss = tf.reduce_mean(tf.square(y_true_features - y_pred_features))

    # 總生成器損失
    gen_loss = adv_loss + lambda_value * perc_loss

    # MSE 損失
    mse = mse_loss(clean_images, generated_images)

    return gen_loss, adv_loss, perc_loss, mse

# 10. 訓練模型，加入早停回調和可視化訓練過程
print("開始訓練模型...")
EPOCHS = epochs

# 建立保存模型的資料夾
checkpoint_dir = './training_checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# 設定模型保存路徑
generator_checkpoint_prefix = os.path.join(checkpoint_dir, "generator_epoch_{epoch}")
discriminator_checkpoint_prefix = os.path.join(checkpoint_dir, "discriminator_epoch_{epoch}")

# 初始化損失列表
train_disc_losses = []
train_gen_losses = []
train_mse_losses = []

val_gen_losses = []
val_mse_losses = []

# 早停參數
patience = 5  # 容忍驗證損失未改善的次數
best_val_loss = np.inf
patience_counter = 0

for epoch in range(EPOCHS):
    print(f"開始第 {epoch+1} 個 Epoch 訓練...")
    start = time.time()

    # 訓練模式
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

    # 驗證模式
    val_steps = 0
    epoch_val_gen_loss = 0
    epoch_val_mse_loss = 0

    for noisy_batch, clean_batch in val_dataset:
        gen_loss, adv_loss, perc_loss, mse = val_step(noisy_batch, clean_batch)

        epoch_val_gen_loss += gen_loss.numpy()
        epoch_val_mse_loss += mse.numpy()
        val_steps += 1

    # 計算平均損失
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

    # 保存模型權重
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
        # 保存最佳模型權重
        best_generator_weights = generator.get_weights()
        best_discriminator_weights = discriminator.get_weights()
        print("驗證損失降低，保存當前模型權重為最佳模型。")
    else:
        patience_counter += 1
        print(f"驗證損失未改善，早停計數：{patience_counter}/{patience}")
        if patience_counter >= patience:
            print("早停條件滿足，停止訓練。")
            # 恢復最佳模型權重
            generator.set_weights(best_generator_weights)
            discriminator.set_weights(best_discriminator_weights)
            break

    # 可視化部分驗證結果
    # plot_generated_images(generator, val_dataset, num_images=3, epoch=epoch+1)

# 11. 繪製損失曲線
def plot_losses(train_gen_losses, val_gen_losses, train_mse_losses, val_mse_losses):
    epochs_range = range(1, len(train_gen_losses) + 1)
    plt.figure(figsize=(12, 5))

    # 生成器損失
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_gen_losses, label='Training Generator Loss')
    plt.plot(epochs_range, val_gen_losses, label='Validation Generator Loss')
    plt.title('Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # MSE
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

# 在訓練結束後繪製損失曲線
plot_losses(train_gen_losses, val_gen_losses, train_mse_losses, val_mse_losses)

# 12. 可視化訓練結果
def plot_generated_images(generator, dataset, num_images=5, epoch=None):
    import matplotlib.pyplot as plt

    # 從驗證資料集中取一批資料
    for noisy_images, clean_images in dataset.take(1):
        generated_images = generator.predict(noisy_images[:num_images])

        for i in range(num_images):
            noisy_img = noisy_images[i].numpy().squeeze()
            denoised_img = generated_images[i].squeeze()
            clean_img = clean_images[i].numpy().squeeze()

            # 計算評估指標
            psnr = compare_psnr(clean_img, denoised_img, data_range=1.0)
            ssim = compare_ssim(clean_img, denoised_img, data_range=1.0)
            # 使用 mean_squared_error 替換 mean_absolute_error
            mse = mean_squared_error(clean_img.flatten(), denoised_img.flatten())

            # 顯示圖像
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(noisy_img, cmap='gray')
            plt.title("Noisy")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(denoised_img, cmap='gray')
            plt.title("Denoised")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(clean_img, cmap='gray')
            plt.title("Clean")
            plt.axis('off')

            plt.suptitle(f'PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, MSE: {mse:.4f}')
            plt.tight_layout()
            if epoch is not None:
                plt.savefig(f'generated_image_{epoch}_index_{i}.png')
            plt.show()

# 可視化部分驗證結果（在訓練結束後）
plot_generated_images(generator, val_dataset, num_images=5)