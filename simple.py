import tensorflow as tf
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from tqdm import tqdm

keras = tf.keras


model = keras.models.Sequential([
  keras.layers.Dense(512, activation="relu"),
  keras.layers.Dense(512, activation="relu"),
  keras.layers.Dense(512, activation="relu"),
  keras.layers.Dense(1, activation=None),
])
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

def circle_sdf(v, radius = 1.):
  return np.sqrt((v*v).sum(axis=1)) - radius

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
plt.show()