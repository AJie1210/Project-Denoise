import torch
import tensorflow as tf
print(torch.cuda.is_available())  # 應返回True
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))