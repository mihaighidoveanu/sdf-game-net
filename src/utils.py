import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from _fontloader_cffi import ffi, lib


def plot_sdf(sdf, scale=1, size=100, offset=0.5, fill=True):
    canvas = np.array([ [ x, y ] for y in reversed(range(size)) for x in range(size) ])
    canvas = scale*(canvas/size - offset)

    canvas = sdf(canvas)
    canvas = np.array(canvas)

    X, Y = np.meshgrid(range(size), range(size))

    if fill:
        cp = plt.contourf(X, Y, canvas.reshape((size, size)), 16)
        plt.colorbar(cp)
    else:
        cp = plt.contour(X, Y, canvas.reshape((size, size)), 16, colors='#ffffff99', linewidths=0.75)

    return cp

def clamp(x, delta):
    return tf.minimum(delta, tf.maximum(-delta, x))

def deepsdf_loss(y_true, y_pred):
    delta = 0
    return tf.abs(clamp(y_true, delta) - clamp(y_pred, delta))


def ScaledLoss(c = 0.2):
    def scaled_loss(y_true, y_pred):
        return tf.reduce_mean(((y_true - y_pred) / (tf.maximum(tf.abs(y_true), 0.0001)**c))**2)
    return scaled_loss

def print_model_variable_counts(model):
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

    print(f'        Total params: {trainable_count + non_trainable_count}')
    print(f'    Trainable params: {trainable_count}')
    print(f'Non-trainable params: {non_trainable_count}')


def get_glyph_sdf(index, scale=1):
    def glyph_sdf(ps): return np.array([lib.get_glyph_distance(index, scale, *p) for p in ps ])
    return glyph_sdf

def plot_glyphs(out_dir, model, glyph_data, scale=1):
    os.makedirs(out_dir, exist_ok=True)

    for glyph_index, name, data in enumerate(glyph_indices):
        def predict(ps):
            _ps = np.empty((len(ps), 2 + len(data)))
            _ps[:,:2] = ps
            _ps[:,2] = data
            return model.predict(_ps)

        fig = plt.figure()

        utils.plot_sdf(predict, scale=scale)
        utils.plot_sdf(get_glyph_sdf(glyph_index, scale=scale), scale=scale, fill=False)
        plt.savefig(f"{out_dir}/{name}.png")
        plt.close()