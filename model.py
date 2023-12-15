from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
import tensorflow as tf
from vgg16 import vgg16_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.layers import Activation

from tensorflow.keras import backend as K

# Hàm ReLU với ngưỡng 0.5
def custom_relu(x):
    return K.switch(x < 0.5, 1.0, 0.0)


def layer_added(output_based_network):
  x = output_based_network
  x = layers.Flatten()(x)
  x = layers.Dense(512, activation='sigmoid')(x)
  x = layers.Dense(512, activation='sigmoid')(x)
  x = layers.Dense(512, activation='sigmoid')(x)
  return x

output_based_network = vgg16_model.output 
output_layer = layer_added(output_based_network)
model = Model(vgg16_model.input, output_layer,name='embedding')
model.summary()
embedding=model
for i, layer in enumerate(embedding.layers):
       print(i, layer.name, layer.trainable)
       


class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)





def make_siamese_model(): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(128,128,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(128,128,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
   
    
siamese_model = make_siamese_model()
siamese_model.summary()
for i, layer in enumerate(siamese_model.layers):
       print(i, layer.name, layer.trainable)
       
