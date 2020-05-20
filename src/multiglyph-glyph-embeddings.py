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
font_path = "../fonts/times.ttf".encode('ascii')


# Training
train_model = True
tensorboard_logs = train_model and True
batch_size = 32
train_count = batch_size*1000 # Shared across shapes!
total_epochs = 10


# Dataset
char_set = list("ABC")
char_set = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
char_set = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
embedding_size = 32

scale = 1

# Plots
plot_results = False
plot_embeddings = True





if __name__ == '__main__':

    ## Initialize
    
    lib.load_font(font_path)

    char_count = len(char_set)
    char_ids = [ ord(char) for char in char_set ]
    glyph_indices = [ lib.get_glyph_index(char_id) for char_id in char_ids ]

    print(char_ids)
    print(glyph_indices)



    ## Model

    model_input = keras.Input(shape=[3], batch_size=batch_size)
    
    in_p, in_id = model_input[:, :2], model_input[:, 2]
    model_embedding = keras.layers.Embedding(char_count, embedding_size, input_length=1, name="glyph_embedding")
    model_embedding = model_embedding(in_id)

    _model_input = keras.layers.concatenate([in_p, model_embedding], axis=1)

    model_layers = keras.models.Sequential([
        # keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1, activation=None),
    ])

    model = keras.Model([model_input], [model_layers(_model_input)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=utils.ScaledLoss(), metrics=["mse"])

    utils.print_model_variable_counts(model)

    ## Training
    if not train_model:
        model.load_weights(model_path)
    else:
        with tf.summary.create_file_writer(logdir) as writer:

            for epoch in range(total_epochs):
                print(f"{epoch+1}/{total_epochs}")

                # Every epoch we create a new data set. This prevents
                # overfitting, because the model should not see the
                # same point more than once.

                # Each point in the training data consists of (x, y, char_id)
                char_dist = np.random.choice(char_count, size=train_count, replace=True)

                train_X = np.empty((train_count, 2 + 1))
                train_X[:,:2] = 0.5*np.random.normal(size=(train_count, 2))
                train_X[:, 2] = char_dist

                # Label for point (x, y, id) is sdf_{id}(x, y)
                train_y = np.empty(train_count)
                for i, glyph_index in enumerate(glyph_indices):
                    class_mask = np.where(char_dist == i)[0] # These are the data points with (x, y, id==i)
                    train_y[class_mask] = utils.get_glyph_sdf(glyph_index, scale=scale)(train_X[class_mask][:, :2])


                history = model.fit(train_X, train_y, batch_size=batch_size, epochs=1)

                tf.summary.scalar('Loss', history.history['loss'][0], step=epoch)
                tf.summary.scalar('MSE', history.history['mse'][0], step=epoch)
                writer.flush()

                # res = model.predict(train_X)

            # save model
            os.makedirs(model_dir, exist_ok = True)
            model_name = f'model_{char_count}_{total_epochs}.h5'
            model_path = os.path.join(model_dir, model_name)




    ## Plotting 

    if plot_results:
        # Render the SDFs as contour plots
        glyph_data = [ (glyph_index, f"{i}_{char_set[i]}", i) for i, glyph_indices in enumerate(glyph_indices)]
        utils.plot_glyphs(out_dir, model, glyph_indices, scale=scale)

    if plot_embeddings:
        # Plot embeddings in a TSNE projection
        embeddings.project(out_dir, model.get_layer("glyph_embedding"), list(range(char_count)), char_set)
