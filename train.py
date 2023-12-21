from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import os
from model import siamese_model
from dataset import train_data
import numpy as np
from tensorflow.keras.metrics import Precision, Recall

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001


checkpoint_dir = './training_checkpoints'
checkpoint_prefix=os.path.join(checkpoint_dir,'ckpt')
checkpoint= tf.train.Checkpoint(opt=opt,siamese_model=siamese_model)

def train_step(batch):
  
    with tf.GradientTape() as tape:
        X= batch[:2]   
        y=batch[2]
        y = tf.cast(batch[2], dtype=tf.float32)
        yhat =siamese_model(X, training=True)       
        loss = binary_cross_loss(y, yhat)
    gradients = tape.gradient(loss, siamese_model.trainable_variables)
    opt.apply_gradients(zip(gradients, siamese_model.trainable_variables))
    
    return loss

def train(data,EPOCHS):
  
    for epoch in range(1,EPOCHS+1):
        print('\n Epoch{}/{}'.format(epoch,EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        r=Recall()
        p=Precision()
       
        for idx, batch in enumerate(data):
           
            loss=train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 

            progbar.update(idx+1)

        print("Loss: {:.4f}, Recall: {:.4f}, Precision: {:.4f}".format(loss.numpy(), r.result().numpy(), p.result().numpy()))


       
        if epoch%10==0:
            checkpoint.save(file_prefix=checkpoint_prefix)

EPOCHS = 10
train(train_data,EPOCHS) 

siamese_model.save('siamesemodel.h5')
