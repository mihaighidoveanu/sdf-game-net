from tensorflow import keras
import tensorflow as tf
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def getdata(input_dim):
    point_count = 100
    X = np.empty((point_count, 2 + 1))
    X[:,:2] = 0.5*np.random.normal(size=(point_count, 2))
    char_dist = np.random.choice(input_dim, size=point_count, replace=True)
    X[:, 2] = char_dist
    return X, char_dist

def getmodel(model):
    embedding = model.get_layer('embedding')
    input_dim = embedding.input_dim
    embedding = embedding.output
    transformer = keras.models.Model(inputs = model.input, outputs = embedding)
    return transformer, input_dim


def visualise(model, out_dir):
    model, input_dim = getmodel(model)
    x, y = getdata(input_dim)
    embeddings = model.predict(x)
    compressor = TSNE(metric = 'euclidean')
    e = compressor.fit_transform(embeddings)
    plt.scatter(e[:, 0], e[:, 1], c = y, cmap = 'Reds')
    plt.savefig(f'{out_dir}/tsne.png')
    plt.show()

def save(model):
    pass

if __name__ == '__main__':
    pass
