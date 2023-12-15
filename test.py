from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall, Accuracy
from tensorflow.keras.utils import custom_object_scope
from model import L1Dist
from dataset import test_data
import numpy as np
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

test_input, test_val, y_true = test_data.as_numpy_iterator().next()
y_hat = model.predict([test_input, test_val])
y_pred = np.sum(y_hat,axis=1)**2
print(y_pred)
print(y_true)
y_pred=[1 if pre >0.5 else 0 for pre in y_hat]
print(y_pred)

m = Recall()
m.update_state(y_true, y_hat)
m.result().numpy()
m = Precision()
m.update_state(y_true, y_hat)
m.result().numpy()
r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = model.predict([test_input, test_val])
    r.update_state(y_true, yhat)
    p.update_state(y_true,yhat) 

print(r.result().numpy(), p.result().numpy())
