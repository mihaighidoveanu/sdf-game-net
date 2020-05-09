import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

import utils
from _fontloader_cffi import ffi, lib

keras = tf.keras

if __name__ == '__main__':
    font_path = "../fonts/times.ttf".encode('ascii')
    lib.load_font(font_path)

    scale = 1

    def get_glyph_sdf(index):
        def glyph_sdf(ps): return np.array([lib.get_glyph_distance(index, scale, *p) for p in ps ])
        return glyph_sdf



    char_set = list("ABC")
    char_set = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    char_ids = [ ord(char) for char in char_set ]
    glyph_indices = [ lib.get_glyph_index(char_id) for char_id in char_ids ]

    embedding_size = 32
    char_embedding = np.random.uniform(-1, 1, size=(len(char_set), embedding_size)) # np.random.choice(1000, size=(len(char_set), replace=False)/1000


    print(char_ids)
    print(glyph_indices)




    ## Model
    
    def scaled_loss(y_true, y_pred):
        return tf.reduce_mean(((y_true - y_pred) / (tf.maximum(tf.abs(y_true), 0.0001)**0.3))**2)

    model = keras.models.Sequential([
        # keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu", input_shape=[2 + embedding_size]),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1, activation=None),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=scaled_loss, metrics=["mse"])
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss="mse")

    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))


    ## Training

    train_count = 10000 # Shared across shapes!

    total_epochs = 50
    for epoch in range(total_epochs):
        print("%s/%s" % (epoch+1, total_epochs))
        train_X = np.empty((train_count, 2 + embedding_size))
        train_X[:,:2] = 0.5*np.random.normal(size=(train_count, 2))
        
        char_dist = np.random.choice(len(char_set), size=train_count, replace=True)


        train_X[:, 2:] = char_embedding[char_dist]

        train_y = np.empty(train_count)
        for i, glyph_index in enumerate(glyph_indices):
            class_mask = np.where(char_dist == i)[0]
            train_y[class_mask] = get_glyph_sdf(glyph_index)(train_X[class_mask][:, :2])

        model.fit(train_X, train_y, batch_size=32, epochs=1)

        res = model.predict(train_X)

        hi = 10


    # display character
    # generate test points in an n x n grid, so we can show the result as an image
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("../out/%s" % timestamp)
    for i, glyph_index in enumerate(glyph_indices):
        def predict(ps):
            _ps = np.empty((len(ps), 2 + embedding_size))
            _ps[:,:2] = ps
            _ps[:,2:] = char_embedding[i]
            # ps = np.insert(ps, 2, char_embedding[i], axis=1)
            return model.predict(_ps)

        fig = plt.figure()

        utils.plot_sdf(predict, scale=scale)
        utils.plot_sdf(get_glyph_sdf(glyph_index), scale=scale, fill=False)
        plt.savefig("../out/%s/%s.png" % (timestamp, char_set[i]))
        plt.close()
