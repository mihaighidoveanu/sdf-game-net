import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import utils

keras = tf.keras

if __name__ == '__main__':


    ## Geometry

    # these SDF functions expect arrays of points, not individual points

    def circle_sdf(v, radius = 1.):
      return np.sqrt((v*v).sum(axis=1)) - radius

    # "fast" because this is not an exact SDF, especially near the corners
    def fast_box_sdf(v, dim = np.array([ 1, 1 ])):
      return np.max(abs(v) - dim, axis=1)


    target_sdf = fast_box_sdf

    ## Model

    model = keras.models.Sequential([
      keras.layers.Dense(512, activation="relu"),
      keras.layers.Dense(512, activation="relu"),
      keras.layers.Dense(512, activation="relu"),
      keras.layers.Dense(1, activation=None),
    ])
    model.compile(optimizer=keras.optimizers.Adam(), loss='mse')


    ## Training

    train_count = 10000

    train_X = 3*np.random.normal(size=(train_count, 2))
    train_y = target_sdf(train_X)

    model.fit(train_X, train_y, batch_size=32, epochs=10)


    ## Evaluation

    utils.plot_sdf(model.predict, scale=scale)
    utils.plot_sdf(glyph_sdf, scale=scale, fill=False)
    plt.show()
