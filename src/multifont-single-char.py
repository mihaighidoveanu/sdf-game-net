"""
Train a model to represent multiple SDFs, with an embedding for the shapes. The
shape embeddings are learned at the same time, by using an Embedding layer.

Shape embeddings are 32d.

Model input is (x, y, id), label is sdf_{id}(x, y), where id is an int from
[0, number of shapes]

The dataset consists of a-z, A-Z, 0-9 from Times New Roman, converted to SDFs
using the stb_truetype library.
"""

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


## Configuration

# Paths
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
out_dir = f"../out/{timestamp}"
logdir = f"{out_dir}/logs/scalars/"
model_dir = f'{out_dir}/model/'


# Training
train_model = True
tensorboard_logs = train_model and True
batch_size = 64
train_count = batch_size*1000 # Shared across shapes!
total_epochs = 30

embedding_size = 2

# Dataset
char_set = list("ABC")
char_set = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
char_set = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
char_set = list("G")


font_path_times = "../fonts/times.ttf".encode('ascii')
font_path_baskerville = "../fonts/Libre-Baskerville-master/LibreBaskerville-Regular.ttf".encode('ascii')
font_path_worksans = "../fonts/Work-Sans-master/fonts/static/TTF/WorkSans-Regular.ttf".encode('ascii')
font_path_junicoderegular = "../fonts/junicode-master/fonts/Junicode-Regular.ttf".encode('ascii')
font_path_garamondregular = "../fonts/EBGaramond-0.016/ttf/EBGaramond12-Regular.ttf".encode('ascii')
font_path_oswaldregular = "../fonts/OswaldFont-master/3.0/Roman/400/Oswald-Regular.ttf".encode('ascii')
font_path_archivoregular = "../fonts/Archivo/ttf/Archivo-Regular.ttf".encode('ascii')
font_set = [ font_path_baskerville, font_path_times, font_path_worksans, font_path_junicoderegular, font_path_garamondregular, font_path_oswaldregular, font_path_archivoregular ]
font_names = [ "baskerville", "times", "worksans", "junicoderegular", "garamondregular", "oswaldregular", "archivoregular" ]

scale = 1

# Plots
plot_results = True
plot_embeddings = True



# Provide keras 'metric' wrappers for our loss functions.
# We use the internal MeanMetricWrapper class from https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/metrics.py#L2126-L2149

MeanMetricWrapper = tf.keras.metrics.MeanAbsoluteError.__mro__[1]

class DeepSDFLossMetric(MeanMetricWrapper):
  def __init__(self, delta=0.01, name='DeepSDFLoss', dtype=None):
    super(DeepSDFLossMetric, self).__init__(utils.DeepSDFLoss(delta=delta), name, dtype=dtype)

class ScaledLossMetric(MeanMetricWrapper):
  def __init__(self, c=0.2, name='ScaledLoss', dtype=None):
    super(ScaledLossMetric, self).__init__(utils.ScaledLoss(c=c), name, dtype=dtype)



if __name__ == '__main__':

    ## Initialize

    char_count = len(char_set)
    font_count = len(font_set)
    char_ids = [ ord(char) for char in char_set ]


    print(char_ids)



    ## Model

    model_in_p = keras.Input(shape=[2], batch_size=batch_size)
    model_in_glyph = keras.Input(shape=[1], batch_size=batch_size)

    embedding = keras.layers.Embedding(char_count*font_count, embedding_size, name="embedding")
    embedding = embedding(model_in_glyph)
    embedding = keras.layers.Reshape([embedding_size])(embedding) # go from (batch_size, 1, embedding_size) to (batch_size, embedding_size)

    _model_input = keras.layers.concatenate([model_in_p, embedding], axis=1)

    model_layers = keras.models.Sequential([
        keras.layers.Dense(1024, activation="relu", input_shape=(2 + embedding_size,)),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1, activation=None),
    ])

    model = keras.Model([model_in_p, model_in_glyph], [model_layers(_model_input)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=utils.ScaledLoss(0.2), metrics=["mse", DeepSDFLossMetric(1.), ScaledLossMetric(0.2)])

    model.summary()
    # utils.print_model_variable_counts(model)

    ## Training
    if not train_model:
        model.load_weights(model_path)
    else:
        writer = tf.summary.create_file_writer(logdir)
        # with  as :

        for epoch in range(total_epochs):
            print(f"{epoch+1}/{total_epochs}")

            # Every epoch we create a new data set. This prevents
            # overfitting, because the model should not see the
            # same point more than once.

            # Each point in the training data consists of (x, y, char_id)

            train_X_p = 0.3*np.random.normal(size=(train_count, 2))
            train_X_glyph = np.random.choice(char_count, size=train_count, replace=True)
            train_X_font = np.random.choice(font_count, size=train_count, replace=True)
            # font_p = np.empty((font_count, train_count))

            # Label for point (x, y, id) is sdf_{id}(x, y)
            train_y = np.empty(train_count)
            for font_i in range(font_count):

                lib.load_font(font_set[font_i])
                font_mask = np.where(train_X_font == font_i)[0]

                for char_i in range(char_count):
                    glyph_index = lib.get_glyph_index(char_ids[char_i])
                    # font_p[font_i] = utils.get_glyph_sdf(glyph_index, scale=scale)(train_X_p)

                    glyph_mask = np.where(train_X_glyph[font_mask] == char_i)[0] # These are the data points with (x, y, id==char_i)
                    glyph_mask = font_mask[glyph_mask]
                    train_y[glyph_mask] = utils.get_glyph_sdf(glyph_index, scale=scale)(train_X_p[glyph_mask])


            history = model.fit([ train_X_p, train_X_glyph + train_X_font*char_count ], train_y, batch_size=batch_size, epochs=1)

            tf.summary.scalar('Loss', history.history['loss'][0], step=epoch)
            tf.summary.scalar('MSE', history.history['mse'][0], step=epoch)
            writer.flush()

            # res = model.predict(train_X)

        # save model
        os.makedirs(model_dir, exist_ok = True)
        model_name = f'model_{char_count}_{total_epochs}.h5'
        model_path = os.path.join(model_dir, model_name)
        model.save(model_path)




    ## Plotting

    if plot_results:
        # Render the SDFs as contour plots
        for font_i in range(font_count):

            lib.load_font(font_set[font_i])
            glyph_data = [ (lib.get_glyph_index(char_ids[i]), f"{font_names[font_i]}_{i}_{char_set[i]}", [ i + font_i*char_count ]) for i in range(char_count)]
            utils.plot_glyphs(out_dir, model, glyph_data, scale=scale)

    if plot_embeddings:
        # Plot embeddings in a TSNE projection
        # groups = [ (list(range(char_count*i, char_count*(i + 1))), char_set, font_names[i]) for i in range(font_count) ]
        # embeddings.project_grouped(out_dir, model.get_layer("embedding"), groups)
        embeddings.project(out_dir, model.get_layer("embedding"), list(range(font_count)), font_names)
