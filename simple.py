import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


keras = tf.keras


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

test_side = 100 # generate test points in an n x n grid, so we can show the result as an image

test_X = np.array([ [ x, y ] for x in range(test_side) for y in range(test_side) ])
test_X = 5*(test_X/100 - 0.5)

test_y = target_sdf(test_X)

predicted = model.predict(test_X)

print("mean error")
print(np.mean(predicted - test_y))



X, Y = np.meshgrid(range(test_side), range(test_side))
cp = plt.contourf(X, Y, predicted.reshape((test_side, test_side)))
plt.colorbar(cp)
plt.contour(X, Y, test_y.reshape((test_side, test_side)), colors='#ffffff99', linewidths=0.75)
plt.show()
