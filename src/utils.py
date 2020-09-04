from PIL import Image
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import sys

from _fontloader_cffi import ffi, lib

def plot_sdf(sdf, scale=1, size=100, offset=0.5, fill=True, levels=16):
    canvas = np.array([ [ x, y ] for y in reversed(range(size)) for x in range(size) ])
    canvas = scale*(canvas/size - offset)

    canvas = sdf(canvas)
    canvas = np.array(canvas)

    X, Y = np.meshgrid(range(size), range(size))

    if fill:
        cp = plt.contourf(X, Y, canvas.reshape((size, size)), levels)
        plt.colorbar(cp)
    else:
        cp = plt.contour(X, Y, canvas.reshape((size, size)), levels, colors='#ffffff99', linewidths=0.75)
    return cp

def DeepSDFLoss(delta = 0.1):
    def clamp(x, delta):
        return tf.minimum(delta, tf.maximum(-delta, x))
    
    def deepsdf_loss(y_true, y_pred):
        result = tf.reduce_mean(tf.abs(clamp(y_true, delta) - clamp(y_pred, delta)))
        # result = tf.cast(result, 'float32') # pretty sure we don't need to do this
        return result
    return deepsdf_loss

def ScaledLoss(c = 0.2, e=0.0001):
    def scaled_loss(y_true, y_pred):
        return tf.reduce_mean(((y_true - y_pred) / (tf.maximum(tf.abs(y_true), e)**c))**2)
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

    for glyph_index, name, data in glyph_data:
        def predict(ps):
            # _ps = np.empty((len(ps), 2 + len(data)))
            # _ps[:,:2] = ps
            # _ps[:,2:] = data

            # data = np.empty( (len(ps), 2 + 2) )
            # data[:, :2] = ps
            # data[:, 2] = glyph_id
            # data[:, 3] = font_id
            # return model.predict(data)

            datas = [ np.array([ datum ]*len(ps)) for datum in data ]
            datas = [ ps, *datas ]
            return model.predict(datas)

        fig = plt.figure()

        plot_sdf(predict, scale=scale, size=64)
        plot_sdf(get_glyph_sdf(glyph_index, scale=scale), scale=scale, size=64, fill=False)
        plt.savefig(f"{out_dir}/{name}.png")
        plt.close()

def sdf_to_bitmap(sdf, scale = 1, offset = .5, size = 256, target_size = 64):

    canvas = np.array([ [ x, y ] for y in range(size) for x in range(size) ])
    canvas = scale*(canvas/size - offset)

    canvas = sdf(canvas)
    canvas = np.array(canvas)

    canvas[canvas <= 0] = 0
    canvas[canvas > 0] = 255

    img = canvas.reshape((size, size))
    img = Image.fromarray(img)

    target_size = (target_size, target_size)
    img = img.resize(target_size, resample = Image.LANCZOS)

    return img

def draw_glyphs(out_dir, model, glyph_data, scale  = 1):

    truedir = os.path.join(out_dir, 'gt')
    predictdir = os.path.join(out_dir, 'pred')
    
    os.makedirs(truedir, exist_ok=True)
    os.makedirs(predictdir, exist_ok=True)

    for glyph_index, name, data in glyph_data:

        def predict(ps):
            datas = [ np.array([ datum ]*len(ps)) for datum in data ]
            datas = [ ps, *datas ]
            return model.predict(datas)

        gt = sdf_to_bitmap(get_glyph_sdf(glyph_index, scale=scale), scale=scale)
        pred = sdf_to_bitmap(predict, scale=scale)

        gt = gt.convert("RGB")
        pred = pred.convert("RGB")

        gt_path = os.path.join(truedir, name)
        gt.save(gt_path, "JPEG")

        pred_path = os.path.join(predictdir, name)
        pred.save(pred_path, "JPEG")

if __name__ == '__main__':

    def circle_sdf(v, radius = 1.):
        return np.sqrt((v * v).sum(axis=1)) - radius

    font = "../fonts/basics/times.ttf".encode('ascii')
    scale = 1
    char = sys.argv[1]

    if lib.load_font(font) == 0:
        print(f'ERROR : Font {font} not loaded!')
        exit()
    
    char_id = ord(char)
    glyph_id = lib.get_glyph_index(char_id)
    sdf = get_glyph_sdf(glyph_id, scale = scale)

    img = sdf_to_bitmap(sdf, size = 150, target_size = 100)
    img.show()

    plot_sdf(sdf)
    plt.show()

