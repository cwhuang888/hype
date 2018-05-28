import tensorflow as tf


def parse_fn(example_proto):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
        'image/class/label': tf.FixedLenFeature(
            [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    }
    parsed_features = tf.parse_single_example(serialized=example_proto, features=keys_to_features)
    #===========================================================================
    # img = tf.decode_raw(parsed_features['image/encoded'], tf.uint8)
    # img = tf.reshape(img, [28, 28, 1])
    # img = tf.cast(img, tf.float32)
    # label = tf.cast(parsed_features['image/class/label'], tf.float32)
    #===========================================================================
    
    img = tf.image.decode_png(parsed_features['image/encoded'])
    img = tf.reshape(img, [28, 28, 1])
    img = tf.cast(img, tf.float32)
    label = tf.cast(parsed_features['image/class/label'], tf.float32)
    return img, label


filenames = tf.data.Dataset.list_files("/home/cwhuang/datasets/mnist/*train*.tfrecord", shuffle=True)
train_dataset = tf.data.TFRecordDataset(filenames)
train_dataset = train_dataset.map(map_func=parse_fn, num_parallel_calls=1)
train_dataset = train_dataset.repeat().shuffle(buffer_size=1000000).batch(32)

train_iterator = train_dataset.make_one_shot_iterator()

#===============================================================================
# next_element = train_iterator.get_next()
# 
# with tf.Session() as sess:load_data
#     sess.run(next_element)
#===============================================================================

print("done reading train cifar10 dataset")

import numpy as np
import keras
from keras import layers

batch_size = 32
buffer_size = 10000
steps_per_epoch = int(np.ceil(60000 / float(batch_size)))
epochs = 5
num_classes = 10

def cnn_layers(inputs):
    x = layers.Conv2D(32, (3, 3),
                      activation='relu', padding='valid')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes,
                               activation='softmax',
                               name='x_train_out')(x)
    return predictions

#===============================================================================
# from keras.models import Sequential
# from keras.layers import Dense, Dropout 
# def mlp_layers(inputs):
#     model = Sequential()
#     model.add(Dense(512, activation='relu', input_shape=(3072,)))    # input_shape indicate the shape of one sample, i.e. ignore batch dim
#     model.add(Dropout(0.2))
#     model.add(Dense(512, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(num_classes, activation='softmax')) # logits follows by multi-class classification prediction layer.
#===============================================================================




# Model creation using tensors from the get_next() graph node.
inputs, targets = train_iterator.get_next()
#===============================================================================
# inputs = tf.to_float(inputs)
# targets = tf.to_float(targets)
# inputs = tf.reshape(inputs, shape=[-1, 32, 32, 3])
#===============================================================================
model_input = layers.Input(tensor=inputs)   # batch of tensors streaming from iterator
model_output = cnn_layers(model_input)
train_model = keras.models.Model(inputs=model_input, outputs=model_output)

train_model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    target_tensors=[targets])
train_model.summary()

train_model.fit(epochs=epochs,
                steps_per_epoch=steps_per_epoch)

