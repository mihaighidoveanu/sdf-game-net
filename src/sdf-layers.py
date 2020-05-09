import os
from datetime import datetime


os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import tensorflow as tb
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K




tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

import utils
from _fontloader_cffi import ffi, lib

keras = tf.keras

def circle_sdf(v, radius = 1.):
    return tf.reshape(tf.sqrt(tf.reduce_sum(v*v, axis=1)) - radius, (-1, 1))

# "fast" because this is not an exact SDF, especially near the corners
def fast_box_sdf(v, dim = np.array([ 1, 1 ])):
    return tf.reshape(tf.reduce_max(abs(v) - dim, axis=1), (-1, 1))

def exact_box_sdf(v, dim = 1.):
    d = abs(v)-dim
    d1 = tf.norm(tf.maximum(d, 0.0), axis=1)
    d2 = tf.minimum(tf.reduce_max(d, axis=1), 0.)
    return tf.reshape(d1 + d2, (-1, 1))
    # return length(max(d,0.0)) + min(max(d.x,d.y),0.0)


def dot(a, b):
    return tf.reduce_sum(a*b, axis=1)

def segment_sdf(p, a, b):
    pa, ba = p-a, b-a
    paba = dot(pa, ba)
    baba = dot(ba, ba)
    h = tf.clip_by_value(paba/baba, 0., 1.)
    h = tf.reshape(h, (-1, 1))
    # h = np.clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 )
    return tf.reshape(tf.norm( pa - ba*h, axis=1), (-1, 1))


class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ScaleLayer, self).__init__()
        self.scale = tf.Variable(1.)

    def call(self, inputs):
        return inputs * self.scale


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = "../out/%s" % timestamp
    os.makedirs(out_dir)

    logdir = "%s/logs/scalars/" % out_dir
    writer = tf.summary.create_file_writer(logdir)


    font_path = "../fonts/times.ttf".encode('ascii')
    lib.load_font(font_path)

    scale = 1

    def get_glyph_sdf(index):
        def glyph_sdf(ps): return np.array([lib.get_glyph_distance(index, scale, *p) for p in ps ])
        return glyph_sdf



    char_set = list("ABC")
    char_set = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    char_ids = [ ord(char) for char in char_set ]
    glyph_indices = [ lib.get_glyph_index(char_id) for char_id in char_ids ]


    print(char_ids)
    print(glyph_indices)




    ## Model
    
    def scaled_loss(y_true, y_pred):
        return tf.reduce_mean(((y_true - y_pred) / (tf.maximum(tf.abs(y_true), 0.001)**0.2))**2)

    model_in = keras.Input(shape=(3), batch_size=32)
    sdf_units = [model_in]

    sdf_filters = 64

    if sdf_filters > 0:
        # for i in range(sdf_filters):
        #     d = keras.layers.Dense(32, activation="relu")(model_in)
        #     d = keras.layers.Dense(2, activation=None)(d)
        #     d = circle_sdf(d)
        #     d = keras.layers.Dense(1, activation=None)(d)
        #     sdf_units.append(d)

        for i in range(sdf_filters):
            d = keras.layers.Dense(64, activation="relu")(model_in)
            d = keras.layers.Dense(64, activation="relu")(model_in)
            d = keras.layers.Dense(4, activation=None)(d)
            d = exact_box_sdf(d[:,:2], d[:,2:])
            d = keras.layers.Dense(1, activation=None)(d)
            sdf_units.append(d)

        # for i in range(sdf_filters):
        #     d = keras.layers.Dense(32, activation="relu")(model_in)
        #     d = keras.layers.Dense(32, activation="relu")(model_in)
        #     d = keras.layers.Dense(6, activation=None)(d)
        #     d = segment_sdf(d[:, :2], d[:, 2:4], d[:, 4:])
        #     d = keras.layers.Dense(1, activation=None)(d)
        #     sdf_units.append(d)


    

        #     sdf_units.append(keras.Sequential([
        #         model_in,
        #         keras.layers.Dense(2, activation=None),
        #         keras.layers.Lambda(circle_sdf),
        #         # keras.layers.Dense(1, activation=None)
        #     ]))
        # for i in range(64):
        #     sdf_units.append(keras.Sequential([
        #         model_in,
        #         keras.layers.Dense(2, activation=None),
        #         keras.layers.Lambda(fast_box_sdf),
        #         # keras.layers.Dense(1, activation=None)
        #     ]))

        print(len(sdf_units))

        bla = keras.layers.concatenate(sdf_units, axis=1)
        print(bla)
    else:
        bla = model_in

    model = keras.Sequential([
        # keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        # keras.layers.Dense(256, activation="relu"),
        # keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(1, activation=None),
    ])(bla)
    model = keras.Model(inputs=[model_in], outputs=[model])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=scaled_loss, metrics=["mse"])


    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))


    ## Training

    train_count = 32*1000 # Shared across shapes!

    total_epochs = 200
    for epoch in range(total_epochs):
        print("%s/%s" % (epoch+1, total_epochs))
        train_X = np.empty((train_count, 3))
        train_X[:,:2] = 0.5*np.random.normal(size=(train_count, 2))
        train_X[:, 2] = np.random.choice(char_ids, size=train_count, replace=True)

        train_y = np.empty(train_count)
        for i, glyph_index in enumerate(glyph_indices):
            class_mask = np.where(train_X[:, 2] == char_ids[i])[0]
            train_y[class_mask] = get_glyph_sdf(glyph_index)(train_X[class_mask][:, :2])

        history = model.fit(train_X, train_y, batch_size=32, epochs=1)

        tf.summary.scalar('Loss', history.history['loss'][0], step=epoch)
        tf.summary.scalar('MSE', history.history['mse'][0], step=epoch)
        writer.flush()


    # display character
    # generate test points in an n x n grid, so we can show the result as an image
    
    for i, glyph_index in enumerate(glyph_indices):
        def predict(ps):
            ps = np.insert(ps, 2, char_ids[i], axis=1)
            return model.predict(ps)

        fig = plt.figure()

        utils.plot_sdf(predict, scale=scale)
        utils.plot_sdf(get_glyph_sdf(glyph_index), scale=scale, fill=False)
        plt.savefig("%s/%s.png" % (out_dir, char_set[i]))
