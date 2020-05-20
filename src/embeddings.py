from datetime import datetime
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
