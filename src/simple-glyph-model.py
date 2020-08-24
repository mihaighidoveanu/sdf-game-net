import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

import embeddings
import utils
from _fontloader_cffi import ffi, lib

keras = tf.keras

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
out_dir = f"../out/{timestamp}_single-glyph_scaled-loss"
logdir = f"{out_dir}/logs/scalars/"
model_dir = f'{out_dir}/model/'

# Training
train_model = True
tensorboard_logs = train_model and True
batch_size = 64
train_count = batch_size*1000 # Shared across shapes!
total_epochs = 30
switch_thresh = 2
importance_alpha = 0.7

plot_results = True

MeanMetricWrapper = tf.keras.metrics.MeanAbsoluteError.__mro__[1]

class DeepSDFLossMetric(MeanMetricWrapper):
  def __init__(self, delta=0.1, name='DeepSDFLoss', dtype=None):
    super(DeepSDFLossMetric, self).__init__(utils.DeepSDFLoss(delta=delta), name, dtype=dtype)

class ScaledLossMetric(MeanMetricWrapper):
  def __init__(self, c=0.2, name='ScaledLoss', dtype=None):
    super(ScaledLossMetric, self).__init__(utils.ScaledLoss(c=c), name, dtype=dtype)

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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=utils.DeepSDFLoss(), metrics=["mse", DeepSDFLossMetric(), ScaledLossMetric()])


    ## Training

    train_count = 100000

    # for epochs in range(10):
    #     train_X = 0.5*np.random.normal(size=(train_count, 2))
    #     train_y = glyph_sdf(train_X)

    #     model.fit(train_X, train_y, batch_size=32, epochs=1)

    ## Training
    if not train_model:
        model.load_weights(model_path)
    else:
        writer = tf.summary.create_file_writer(logdir)
        # with  as :

        with writer.as_default():
            for epoch in range(total_epochs):
                print(f"{epoch+1}/{total_epochs}")

                train_X = 0.3*np.random.normal(size=(train_count, 2))
                train_y = glyph_sdf(train_X)

                history = model.fit(train_X, train_y, batch_size=batch_size, epochs=1)

                tf.summary.scalar('Loss', history.history['loss'][0], step=epoch)
                tf.summary.scalar('MSE', history.history['mse'][0], step=epoch)
                tf.summary.scalar('DeepSDF Error', history.history['DeepSDFLoss'][0], step=epoch)
                tf.summary.scalar('Scaled Error', history.history['ScaledLoss'][0], step=epoch)
                writer.flush()

            
        # save model
        os.makedirs(model_dir, exist_ok = True)
        model_name = f'model_{char}_{total_epochs}.h5'
        model_path = os.path.join(model_dir, model_name)
        model.save(model_path)




    ## Plotting 

    if plot_results:

        # display character
        # generate test points in an n x n grid, so we can show the result as an image
        utils.plot_sdf(model.predict, scale=scale)
        utils.plot_sdf(glyph_sdf, scale=scale, fill=False)
        
        plt.savefig(f"{out_dir}/{char}.png")
        plt.show()
