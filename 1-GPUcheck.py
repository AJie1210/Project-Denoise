import torch
import tensorflow as tf
print(torch.cuda.is_available())  # 應返回True
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Tensorflow 版本:", tf.__version__)
print("PyTorch 版本:", torch.__version__)
print("PyTorch 使用的 CUDA 版本:", torch.version.cuda)
print(torch.backends.cudnn.version())
print("CUDA 可用性:", torch.cuda.is_available())