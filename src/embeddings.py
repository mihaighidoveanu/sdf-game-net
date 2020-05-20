from datetime import datetime
import os
from tensorflow import keras
import tensorflow as tf
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def getdata(char_classes):
    point_count = char_classes * 500
    X = np.empty((point_count, 2 + 1))
    X[:,:2] = 0.5*np.random.normal(size=(point_count, 2))
    char_dist = np.random.choice(char_classes, size=point_count, replace=True)
    X[:, 2] = char_dist
    return X, char_dist

def getmodel(model):
    embedding = model.get_layer('embedding')
    input_dim = embedding.input_dim
    embedding = embedding.output
    transformer = keras.models.Model(inputs = model.input, outputs = embedding)
    return transformer, input_dim

def buildmodel(embeddings_input_dim, embedding_size):
    def scaled_loss(y_true, y_pred):
        return tf.reduce_mean(((y_true - y_pred) / (tf.maximum(tf.abs(y_true), 0.0001)**0.3))**2)

    model_input = keras.Input(shape=[3], batch_size=32)
    
    in_p, in_id = model_input[:, :2], model_input[:, 2]
    model_embedding = keras.layers.Embedding(embeddings_input_dim, embedding_size, input_length=1)
    model_embedding = model_embedding(in_id)

    _model_input = keras.layers.concatenate([in_p, model_embedding], axis=1)

    model_layers = keras.models.Sequential([
        # keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu", input_shape=[2 + embedding_size]),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dense(1, activation=None),
    ])

    model = keras.Model([model_input], [model_layers(_model_input)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=scaled_loss, metrics=["mse"])

    return model

def visualise(model, out_dir, char_classes, show_plots = True):
    model, input_dim = getmodel(model)
    x, y = getdata(char_classes = char_classes)
    embeddings = model.predict(x)
    compressor = TSNE(metric = 'euclidean')
    e = compressor.fit_transform(embeddings)
    # plt.scatter(e[:, 0], e[:, 1], c = y, cmap = 'Reds')
    plt.scatter(e[:, 0], e[:, 1], c = y, cmap = 'viridis')
    plt.savefig(f'{out_dir}/tsne.png')
    if show_plots:
        plt.show()
    plt.close()

def save(model):
    pass

if __name__ == '__main__':
    char_set = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    char_count = len(char_set)
    total_epochs = 50
    embedding_size = 32

    model_dir = '../models'
    model_name = f'model_{char_count}_{total_epochs}.h5'
    model_path = os.path.join(model_dir, model_name)

    embeddings_input_dim = char_count
    model = buildmodel(embeddings_input_dim, embedding_size)
    model.load_weights(model_path)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = "../out/%s" % timestamp
    os.makedirs(out_dir)

    for max_char in range(1, char_count):
        print(f'Embedding points from {max_char} character classes')
        save_dir = os.path.join(out_dir, f'{max_char}')
        os.makedirs(save_dir)
        visualise(model, save_dir, char_classes = max_char, show_plots = False)
