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
      keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')


    ## Training

    train_count = 10000

    train_X = 3*np.random.normal(size=(train_count, 2))
    train_y = target_sdf(train_X) < 0

    model.fit(train_X, train_y, batch_size=32, epochs=10)


    ## Evaluation

train_count = 10000

train_X = 3*np.random.normal(size=(train_count, 2))
train_y = circle_sdf(train_X)

model.fit(train_X, train_y, batch_size=32, epochs=5)

test_side = 100

test_X = np.array([ [ x, y ] for x in range(test_side) for y in range(test_side) ])
test_X = 5*(test_X/100 - 0.5)

test_y = circle_sdf(test_X)


predicted = model.predict(test_X)

print("mean error")
print(np.mean(predicted - test_y))

plt.imshow(predicted.reshape((test_side, test_side)), 
  cmap='hot', interpolation='nearest')
