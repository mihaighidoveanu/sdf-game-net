from _fontloader_cffi import ffi, lib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import utils

keras = tf.keras

if __name__ == '__main__':
    char = 'a'
    scale = 1

    font_path = "../fonts/times.ttf".encode('ascii')
    lib.load_font(font_path)

    index = lib.get_glyph_index(ord(char))
    def glyph_sdf(ps): return np.array([
        lib.get_glyph_distance(index, scale, *p) for p in ps ])

    # box = lib.get_glyph_box(index, scale)


    ## Model
    
    def scaled_loss(y_true, y_pred):
        return tf.reduce_mean(((y_true - y_pred) / (tf.abs(y_true)**0.2))**2)

    model = keras.models.Sequential([
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(1, activation=None),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=utils.ScaledLoss(), metrics=["mse"])

    ## Training

    train_count = 100000

    for epochs in range(10):
        train_X = 0.5*np.random.normal(size=(train_count, 2))
        train_y = glyph_sdf(train_X)
        # print(pred_y)
        # print(pred_y.shape)
        # print(pred_y.dtype)
        # print(utils.deepsdf_loss(train_y, pred_y))
        # print(utils.deepsdf_loss(train_y, pred_y).dtype)
        # exit()

        model.fit(train_X, train_y, batch_size=32, epochs=1)


    # display character
    # generate test points in an n x n grid, so we can show the result as an image
    utils.plot_sdf(model.predict, scale=scale)
    utils.plot_sdf(glyph_sdf, scale=scale, fill=False)
    plt.show()
