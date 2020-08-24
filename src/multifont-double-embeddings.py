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
train_model = False
tensorboard_logs = train_model and True
batch_size = 64
train_count = batch_size*1000 # Shared across shapes!
total_epochs = 50
switch_thresh = 2
importance_alpha = 0.7

# Dataset
char_set = list("ABC")
char_set = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# char_set = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
# char_set = list("abcdefghijklmnopqrstuvwxyz")
glyph_embedding_size = 32



font_path_times = "../fonts/times.ttf".encode('ascii')
font_path_baskerville = "../fonts/Libre-Baskerville-master/LibreBaskerville-Regular.ttf".encode('ascii')
font_path_worksans = "../fonts/Work-Sans-master/fonts/static/TTF/WorkSans-Regular.ttf".encode('ascii')
font_path_junicoderegular = "../fonts/junicode-master/fonts/Junicode-Regular.ttf".encode('ascii')
font_path_garamondregular = "../fonts/EBGaramond-0.016/ttf/EBGaramond12-Regular.ttf".encode('ascii')
font_path_oswaldregular = "../fonts/OswaldFont-master/3.0/Roman/400/Oswald-Regular.ttf".encode('ascii')
font_path_archivoregular = "../fonts/Archivo/ttf/Archivo-Regular.ttf".encode('ascii')
font_set = [ font_path_baskerville, font_path_times, font_path_worksans, font_path_junicoderegular, font_path_garamondregular, font_path_oswaldregular, font_path_archivoregular ]
font_names = [ "baskerville", "times", "worksans", "junicoderegular", "garamondregular", "oswaldregular", "archivoregular" ]
font_embedding_size = 32

scale = 2

# Plots
plot_results = True
plot_embeddings = False



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

    # embedding_model = keras.Sequential([
    #     keras.layers.Embedding(char_count, embedding_size, input_length=1, name="glyph_embedding")
    # ])

    model_in_p = keras.Input(shape=[2], batch_size=batch_size)
    model_in_glyph = keras.Input(shape=[1], batch_size=batch_size)
    model_in_font = keras.Input(shape=[1], batch_size=batch_size)
    
    glyph_embedding = keras.layers.Embedding(char_count, glyph_embedding_size, name="glyph_embedding")
    glyph_embedding = glyph_embedding(model_in_glyph)
    glyph_embedding = keras.layers.Reshape([glyph_embedding_size])(glyph_embedding) # go from (batch_size, 1, glyph_embedding_size) to (batch_size, glyph_embedding_size)
    
    font_embedding = keras.layers.Embedding(font_count, font_embedding_size, name="font_embedding")
    font_embedding = font_embedding(model_in_font)
    font_embedding = keras.layers.Reshape([font_embedding_size])(font_embedding) # go from (batch_size, 1, font_embedding_size) to (batch_size, font_embedding_size)

    _model_input = keras.layers.concatenate([model_in_p, glyph_embedding, font_embedding], axis=1)

    model_layers = keras.models.Sequential([
        keras.layers.Dense(1024, activation="relu", input_shape=(2 + glyph_embedding_size + font_embedding_size,)),
        # keras.layers.Dropout(.3),
        # keras.layers.Dense(1024, activation="relu"),
        # keras.layers.Dropout(.3),
        keras.layers.Dense(1024, activation="relu"),
        # keras.layers.Dropout(.3),
        keras.layers.Dense(1024, activation="relu"),
        # keras.layers.Dropout(.3),
        keras.layers.Dense(1024, activation="relu"),
        # keras.layers.Dropout(.3),
        keras.layers.Dense(1, activation=None),
    ])

    model = keras.Model([model_in_p, model_in_glyph, model_in_font], [model_layers(_model_input)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00025), loss=utils.ScaledLoss(), metrics=["mse", DeepSDFLossMetric(1.), ScaledLossMetric(0.6)])

    model.summary()
    # utils.print_model_variable_counts(model)

    

    font_importance = [ 1 / font_count for i in range(font_count) ]
    glyph_importance = [ 1 / char_count for i in range(char_count) ]
    ## Training
    if not train_model:
        model.load_weights("../out/20200522-175229/model/model_26_50.h5")
    else:
        writer = tf.summary.create_file_writer(logdir)
        # with  as :

        for epoch in range(total_epochs):
            print(f"{epoch+1}/{total_epochs}")

            # if epoch >= switch_thresh:
            #     glyph_embedding.trainable = epoch % 2 == 0
            #     font_embedding.trainable = epoch % 2 == 1

            # Every epoch we create a new data set. This prevents
            # overfitting, because the model should not see the
            # same point more than once.

            # Each point in the training data consists of (x, y, char_id)
            print(f"Font importance {font_importance}")

            train_X_p = 0.6*np.random.normal(size=(train_count, 2))
            train_X_glyph = np.random.choice(char_count, size=train_count, replace=True, p=glyph_importance)
            train_X_font = np.random.choice(font_count, size=train_count, replace=True, p=font_importance)

            # Label for point (x, y, id) is sdf_{id}(x, y)
            train_y = np.empty(train_count)
            for font_i in range(font_count):
                
                lib.load_font(font_set[font_i])
                font_mask = np.where(train_X_font == font_i)[0]

                for char_i in range(char_count):
                    glyph_index = lib.get_glyph_index(char_ids[char_i])
                    
                    glyph_mask = np.where(train_X_glyph[font_mask] == char_i)[0] # These are the data points with (x, y, id==char_i)
                    glyph_mask = font_mask[glyph_mask]
                    train_y[glyph_mask] = utils.get_glyph_sdf(glyph_index, scale=scale)(train_X_p[glyph_mask])


            history = model.fit([ train_X_p, train_X_glyph, train_X_font ], train_y, batch_size=batch_size, epochs=1)

            tf.summary.scalar('Loss', history.history['loss'][0], step=epoch)
            tf.summary.scalar('MSE', history.history['mse'][0], step=epoch)
            writer.flush()

            res = model.predict([ train_X_p, train_X_glyph, train_X_font ])
            error = abs(train_y - res.squeeze())
            
            font_error = np.empty(font_count)
            for i in range(font_count):
                font_mask = np.where(train_X_font == i)[0]
                _err = np.sum(error[font_mask]) / len(font_mask)
                font_error[i] = _err
            font_sum_error = font_error.sum()

            font_importance = np.array([ (err**importance_alpha)/(font_sum_error**importance_alpha) for err in font_error ])
            font_importance = font_importance / font_importance.sum()
            
            glyph_error = np.empty(char_count)
            for i in range(char_count):
                glyph_mask = np.where(train_X_glyph == i)[0]
                _err = np.sum(error[glyph_mask]) / len(glyph_mask)
                glyph_error[i] = _err
            glyph_sum_error = glyph_error.sum()

            glyph_importance = np.array([ (err**importance_alpha)/(glyph_sum_error**importance_alpha) for err in glyph_error ])
            glyph_importance = glyph_importance / glyph_importance.sum()


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
            glyph_data = [ (lib.get_glyph_index(char_ids[i]), f"{font_names[font_i]}_{i}_{char_set[i]}", [ i, font_i ]) for i in range(3,7)]
            utils.plot_glyphs(out_dir, model, glyph_data, scale=scale, single_fig=True, per_row=4, name=f"{font_names[font_i]}.png")

    if plot_embeddings:
        # Plot embeddings in a TSNE projection
        embeddings.project(out_dir, model.get_layer("glyph_embedding"), list(range(char_count)), char_set, name="glyph_tsne.png")
        embeddings.project(out_dir, model.get_layer("font_embedding"), list(range(font_count)), font_names, name="glyph_tsne.png")
