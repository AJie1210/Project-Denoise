import os
import tensorflow as tf
from keras import layers, Model
# from tensorflow.keras import layers, Model

# 設置固定權重檔案路徑和輸出路徑
CHECKPOINT_PREFIX = 'D:\\GitHub\\Denoise\\training_checkpoints\\generator_epoch_39'
OUTPUT_FILE = 'D:\\GitHub\\Denoise\\best_generator_model.h5'

# 定義多尺度卷積塊（與訓練時相同）
def multi_scale_conv_block(inputs, filters):
    conv_1x1 = layers.Conv2D(filters, (1, 1), activation='relu', padding='same')(inputs)
    conv_3x3 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv_5x5 = layers.Conv2D(filters, (5, 5), activation='relu', padding='same')(inputs)
    concat = layers.Concatenate()([conv_1x1, conv_3x3, conv_5x5])
    return concat

# 定義生成器模型（與訓練時相同）
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

# 初始化模型
print(f"加載生成器模型...")
generator = unet_generator()

# 加載權重檔案
if os.path.exists(CHECKPOINT_PREFIX + ".index"):
    print(f"加載權重檔案: {CHECKPOINT_PREFIX}")
    generator.load_weights(CHECKPOINT_PREFIX)
else:
    raise FileNotFoundError(f"未找到權重檔案: {CHECKPOINT_PREFIX}")

# 保存為 .h5 文件
print(f"保存為 .h5 文件: {OUTPUT_FILE}")
generator.save(OUTPUT_FILE)
print("轉換完成！")
