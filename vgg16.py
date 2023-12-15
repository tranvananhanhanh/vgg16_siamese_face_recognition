import numpy as np
import os 
from PIL import Image
import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split



# Tạo mô hình VGG16
vgg16_model = vgg16.VGG16(weights='imagenet', include_top=False,input_shape = (128, 128, 3))
# Đóng băng 4 block đầu
for layer in vgg16_model.layers[:15]:
    layer.trainable = False

for i, layer in enumerate(vgg16_model.layers):
       print(i, layer.name, layer.trainable)
       
vgg16_model.summary()