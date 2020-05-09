import os
from datetime import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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


    print(char_ids)
    print(glyph_indices)




    ## Model
    
    def scaled_loss(y_true, y_pred):
        return tf.reduce_mean(((y_true - y_pred) / (tf.maximum(tf.abs(y_true), 0.00001)**0.2))**2)

    model = keras.models.Sequential([
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1, activation=None),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=scaled_loss, metrics=["mse"])


    ## Training

    train_count = 1000000 # Shared across shapes!

    for epochs in range(50):
        train_X = np.empty((train_count, 3))
        train_X[:,:2] = 0.5*np.random.normal(size=(train_count, 2))
        train_X[:, 2] = np.random.choice(char_ids, size=train_count, replace=True)

        train_y = np.empty(train_count)
        for i, glyph_index in enumerate(glyph_indices):
            class_mask = np.where(train_X[:, 2] == char_ids[i])[0]
            train_y[class_mask] = get_glyph_sdf(glyph_index)(train_X[class_mask][:, :2])

        model.fit(train_X, train_y, batch_size=32, epochs=1)


    # display character
    # generate test points in an n x n grid, so we can show the result as an image
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("../out/%s" % timestamp)
    for i, glyph_index in enumerate(glyph_indices):
        def predict(ps):
            ps = np.insert(ps, 2, char_ids[i], axis=1)
            return model.predict(ps)

        fig = plt.figure()

        utils.plot_sdf(predict, scale=scale)
        utils.plot_sdf(get_glyph_sdf(glyph_index), scale=scale, fill=False)
        plt.savefig("../out/%s/%s.png" % (timestamp, char_set[i]))
