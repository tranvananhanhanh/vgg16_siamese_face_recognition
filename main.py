from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall, Accuracy
from tensorflow.keras.utils import custom_object_scope
from model import L1Dist
from PIL import Image
import numpy as np
import tensorflow as tf

from data_for_train import preprocess
# Đường dẫn đến tệp h5 của mô hình
model_path = '/Users/jmac/Desktop/siamese/siamesemodel.h5'

custom_objects = {'L1Dist': L1Dist}
try:
    # Tải mô hình từ tệp h5
    
    with custom_object_scope(custom_objects):
        model = load_model('/Users/jmac/Desktop/siamese/siamesemodel.h5',compile=True)
   
    print("Mô hình đã được tải thành công từ tệp h5.")
except OSError:
    print("Không thể tải mô hình từ tệp h5.")



test_input=preprocess('/Users/jmac/Desktop/siamese/testdata/208bf37a592c51ea7d0b40962b568e68.jpeg')
test_val=preprocess('/Users/jmac/Desktop/siamese/testdata/Lisa-Blackpink1.jepg.jpg')
print(test_input.shape)

result = model.predict(list(np.expand_dims([test_input, test_val], axis=1)))
print(result)

y_pred = result*result
print(y_pred)



