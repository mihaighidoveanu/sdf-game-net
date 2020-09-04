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
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

import embeddings
import utils
from _fontloader_cffi import ffi, lib

keras = tf.keras

import faulthandler
faulthandler.enable()

## Configuration

# Paths
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# Training
train_model = True
tensorboard_logs = train_model and True
batch_size = 64
train_count = batch_size*1000 # Shared across shapes!
total_epochs = 50
switch_thresh = 2
importance_alpha = 0.7

# Dataset
char_set = list("ABC")
char_set = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# char_set = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
# char_set = list("abcdefghijklmnopqrstuvwxyz")
glyph_embedding_size = 32
train_chars_count = len(char_set) - 1
train_chars_count = 5

font_dir = "../fonts/basics".encode('ascii')
font_dir = "../fonts/MCGAN/"
times = "../fonts/times.ttf".encode('ascii')
bahiana = "../fonts/Bahiana-Regular.ttf".encode('ascii')
font_names = ["times", "bahiana"]
test_font_set = ["1/Acme/Acme-Regular.ttf", "1/Cormorant/Cormorant-Italic.ttf", 
    "1/Fascinate_Inline/FascinateInline-Regular.ttf", "1/Alef/Alef-Bold.ttf", 
    "2/Jura/Jura-VariableFont_wght.ttf", "2/Monoton/Monoton-Regular.ttf",
    "2/Roboto/Roboto-Thin.ttf", "2/Source_Sans_Pro/SourceSansPro-SemiBold.ttf",
    "2/Source_Sans_Pro/SourceSansPro-SemiBoldItalic.ttf", "2/Germania_One/GermaniaOne-Regular.ttf"]

test_font_set = [os.path.join(font_dir, font).encode('ascii') for font in test_font_set]

font_embedding_size = 32

all_embedding_size = 32

scale = 2

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

    # define possible characters

    ## Initialize

    char_count = len(char_set)
    char_ids = [ ord(char) for char in char_set ]

    fonts_set = list(Path(font_dir).glob('**/*.ttf'))
    fonts_set = [ str(font).encode('ascii') for font in fonts_set ]


    if len(sys.argv) == 3:
        train_fonts_count = int(sys.argv[1])
        train_chars_count = int(sys.argv[2])
        fonts_set = np.random.choice(fonts_set, size = train_fonts_count, replace = False)
    else:
        train_chars_count = int(sys.argv[1])

    # font_names = [ os.path.basename(font) for font in fonts_set ]

    # fonts_set = []
    # for font in os.listdir(font_dir):
    #     if os.path.splitext(font)[1] == b'.ttf':
    #         font_path = os.path.join(font_dir, font)
    #         fonts_set.append(font_path)

    font_count = len(fonts_set)
    print('Fonts ', fonts_set)
    print('Chars ', char_ids)

    print('Font count', font_count)
    print('Char count', char_count)
    print('Train char count', train_chars_count)

    out_dir = f"../out/multi-font/final/{font_count}fonts_{train_chars_count}chars_each_{total_epochs}"
    logdir = f"{out_dir}/logs/scalars/"
    logfile = f"{out_dir}/logs/log.txt"
    model_dir = f'{out_dir}/model/'
    os.makedirs(out_dir, exist_ok = True)
    os.makedirs(logdir, exist_ok = True)

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

    all_embedding = keras.layers.Embedding(font_count * char_count, all_embedding_size, name="all_embedding")
    idx = model_in_glyph + char_count * model_in_font
    all_embedding = all_embedding(idx)
    all_embedding = keras.layers.Reshape([all_embedding_size])(all_embedding) # go from (batch_size, 1, font_embedding_size) to (batch_size, font_embedding_size)

    _model_input = keras.layers.concatenate([model_in_p, glyph_embedding, font_embedding], axis=1)
    # _model_input = keras.layers.concatenate([model_in_p, all_embedding], axis=1)

    input_size = 2 + glyph_embedding_size + font_embedding_size
    # input_size = 2 + all_embedding_size

    model_layers = keras.models.Sequential([
        keras.layers.Dense(1024, activation="relu", input_shape=(input_size ,)),
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

    # font_importance = [ 1 / font_count for i in range(font_count) ]
    # glyph_importance = [ 1 / char_count for i in range(char_count) ]

    font_glyphs = {}
    for font_i in range(font_count):
        chars_to_train = np.random.choice(char_count, size = train_chars_count, replace = False)
        chars_to_test = [ char for char in range(char_count) if char not in chars_to_train ]

        font_glyphs[font_i] = {
            'train' : chars_to_train,
            'test' : chars_to_test
            }

    # font_glyphs = {}

    # chars_to_train = np.random.choice(char_count, size = train_chars_count, replace = False)
    # chars_to_test = [ char for char in range(char_count) if char not in chars_to_train ]

    # font_glyphs[0] = {
    #     'train' : chars_to_train,
    #     'test' : chars_to_test
    #     }

    # font_glyphs[1] = {
    #     'train' : chars_to_test,
    #     'test' : chars_to_train
    #     }

    ## Training
    if not train_model:
        model.load_weights("../out/20200522-175229/model/model_26_50.h5")
    else:
        writer = tf.summary.create_file_writer(logdir)
        # with  as :
        # with writer.as_default():
        logwriter = open(logfile, 'w')
        for epoch in range(total_epochs):

            print(f" {epoch+1} / {total_epochs} ")

            # if epoch >= switch_thresh:
            #     glyph_embedding.trainable = epoch % 2 == 0
            #     font_embedding.trainable = epoch % 2 == 1

            # Every epoch we create a new data set. This prevents
            # overfitting, because the model should not see the
            # same point more than once.

            # Each point in the training data consists of (x, y, char_id)

            # print(f"Font importance {font_importance}")

            train_X_p = 0.6 * np.random.normal(size = (train_count, 2))
            train_X_font = np.random.choice(font_count, size=train_count, replace=True)
            train_X_glyph = np.empty(train_count)

            for font_i in range(font_count):
                mask = (train_X_font == font_i)
                glyphs  = np.random.choice(font_glyphs[font_i]['train'], size=sum(mask == True), replace=True)
                train_X_glyph[mask] = glyphs

            # Label for point (x, y, id) is sdf_{id}(x, y)
            train_y = np.empty(train_count)
            for font_i in range(font_count):
                # print('Font ', fonts_set[font_i])
                if lib.load_font(fonts_set[font_i]) == 0:
                    print(f'ERROR : Font {fonts_set[font_i]} not loaded!')
                    continue
                font_mask = np.where(train_X_font == font_i)[0]

                for char_i in range(char_count):
                    glyph_index = lib.get_glyph_index(char_ids[char_i])
                    glyph_mask = np.where(train_X_glyph[font_mask] == char_i)[0] # These are the data points with (x, y, id==char_i)
                    glyph_mask = font_mask[glyph_mask]
                    train_y[glyph_mask] = utils.get_glyph_sdf(glyph_index, scale=scale)(train_X_p[glyph_mask])

            history = model.fit([ train_X_p, train_X_glyph, train_X_font ], train_y, batch_size=batch_size, epochs=1)

            loss = history.history['loss'][0]
            mse = history.history['mse'][0]
            logwriter.write(f'{epoch}, {loss}, {mse}\n')

            tf.summary.scalar('Loss', history.history['loss'][0], step=epoch)
            tf.summary.scalar('MSE', history.history['mse'][0], step=epoch)
            writer.flush()

            res = model.predict([ train_X_p, train_X_glyph, train_X_font ])

        logwriter.close()

        # save model
        os.makedirs(model_dir, exist_ok = True)
        model_name = f'model_{font_count}_{char_count}_{total_epochs}.h5'
        model_path = os.path.join(model_dir, model_name)
        model.save_weights(model_path)
        # model.save(model_path)

    ## Plotting 

    if plot_results:
        # Render the SDFs as contour plots

        test_fount_count = len(test_font_set)
        for font_i in range(test_fount_count):
            font_path = test_font_set[font_i]
            print('Font ', font_path)
            if lib.load_font(font_path) == 0:
                continue

            font_name = os.path.basename(font_path)
            font_name, _ = os.path.splitext(font_name)
            font_name = str(font_name)[2:-1]
            plot_dir = os.path.join(out_dir, font_name, "plot")
            draw_dir = os.path.join(out_dir, font_name, "draw")

            glyph_data = [ (lib.get_glyph_index(char_ids[i]), f"test_{i}_{char_set[i]}", [ i, font_i ]) for i in font_glyphs[font_i]['test'] ]
            utils.plot_glyphs(plot_dir, model, glyph_data, scale=scale)
            utils.draw_glyphs(draw_dir, model, glyph_data, scale=scale)

            # training_glyphs_plot = np.random.choice(font_glyphs[font_i]['train'], size = 5, replace = False)
            glyph_data = [ (lib.get_glyph_index(char_ids[i]), f"train_{i}_{char_set[i]}", [ i, font_i ]) for i in font_glyphs[font_i]['train'] ]
            utils.plot_glyphs(plot_dir, model, glyph_data, scale=scale)
            utils.draw_glyphs(draw_dir, model, glyph_data, scale=scale)

    if plot_embeddings:
        # Plot embeddings in a TSNE projection
        embeddings.project(out_dir, model.get_layer("glyph_embedding"), list(range(char_count)), char_set, name="glyph_tsne.png")
        embeddings.project(out_dir, model.get_layer("font_embedding"), list(range(font_count)), font_names, name="glyph_tsne.png")
        # embeddings.project(out_dir, model.get_layer("all_embedding"), list(range(char_count)), char_set, name="glyph_tsne.png")
