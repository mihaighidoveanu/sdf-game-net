import os

from tensorflow import keras
import tensorflow as tf
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def project(out_dir, embedding_layer, ids, labels, name="tsne.png"):
    os.makedirs(out_dir, exist_ok=True)
    model = keras.Sequential([ embedding_layer ])
    shape_embeddings = model.predict(ids).reshape((len(ids), -1))

    compressor = TSNE(metric = 'euclidean')
    tsne_embeddings = compressor.fit_transform(shape_embeddings)

    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1])
    for label, coord in zip(labels, tsne_embeddings):
        plt.annotate(label, coord)

    plt.savefig(f'{out_dir}/tsne.png')
    plt.show()
    plt.close()

if __name__ == '__main__':
    pass
