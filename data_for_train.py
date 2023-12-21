import tensorflow as tf

anchor = tf.data.Dataset.list_files('/Users/jmac/Desktop/siamese/data/anchor/*.jpg', shuffle=False).take(300)
positive = tf.data.Dataset.list_files('/Users/jmac/Desktop/siamese/data/positive/*.jpg', shuffle=False).take(300)
negative = tf.data.Dataset.list_files('/Users/jmac/Desktop/siamese/data/negative/*.jpg', shuffle=False).take(299)

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (128,128))
    img = img / 255.0
    return img

#creat label
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
data = positives.concatenate(negatives)

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)



