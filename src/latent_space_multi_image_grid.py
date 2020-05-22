"""
Train a model to represent multiple SDFs, with an embedding for the shapes. The
shape embeddings are learned at the same time, by using an Embedding layer.

Shape embeddings are 32d.

Model input is (x, y, id), label is sdf_{id}(x, y), where id is an int from
[0, number of shapes]

The dataset consists of a-z, A-Z, 0-9 from Times New Roman, converted to SDFs
using the stb_truetype library.

Also allows for the creation of a pictures of different characters and configurations
"""

import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

import embeddings
import utils
from _fontloader_cffi import ffi, lib

import shutil

keras = tf.keras


## Configuration

# Paths
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
out_dir = f"../out/{timestamp}"
logdir = f"{out_dir}/logs/scalars/"
model_dir = f'{out_dir}/model/'
font_path = "../fonts/times.ttf".encode('ascii')


# Training
train_model = False
tensorboard_logs = train_model and True
batch_size = 32
train_count = batch_size*1000 # Shared across shapes!
total_epochs = 10


# Dataset
#char_set = list("ABC")
char_set = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
#char_set = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
embedding_size = 32

scale = 1


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
    ],name="sequential_part")

    model = keras.Model([model_input], [model_layers(_model_input)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=utils.ScaledLoss(), metrics=["mse"])

    utils.print_model_variable_counts(model)



    ## Training
    os.makedirs(model_dir, exist_ok = True)
    model_name = f'model_{char_count}_{total_epochs}.h5'
    model_path = os.path.join(model_dir, model_name)
    
    if not train_model:
        model.load_weights(model_path)
    else:
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
            #writer.flush()

            # res = model.predict(train_X)

        # save model
        model.save_weights(model_path)
        print(model_path)

    



    # Plotting function
    def latent_plot_sdf(sdf, a, scale=1, size=100, offset=0.5):
        canvas = np.array([ [ x, y ] for y in reversed(range(size)) for x in range(size) ])
        canvas = scale*(canvas/size - offset)
    
        canvas = sdf(canvas)
        canvas = np.array(canvas)

        a.imshow(canvas.reshape((size, size)), interpolation='bilinear', origin='lower',
                        cmap=cm.flag, vmin = -1, vmax = 1, extent=(0, 1, 0, 1))
        return a


    # Get embeddings
    ids = list(range(char_count))
    embedding_layer = model.get_layer("glyph_embedding")
    model_temp = keras.Sequential([ embedding_layer ])
    shape_embeddings = model_temp.predict(ids).reshape((len(ids), -1))
    
    # Build output model
    temp_seq = model.get_layer("sequential_part")
    temp_input = keras.Input(shape=[embedding_size+2])
    model_latent = keras.Sequential([temp_input,temp_seq])
    #model_latent.summary()

    
    # Plot settings
    
    # Grid of the glyphs to be plotted and their positions
    # Numbers are indeces from the char_set list
    grid = [[ 0, 1, 2, 3, 4],
            [ 5, 6, 7, 8, 9],
            [10,11,12,13,14],
            [15,16,17,18,19],
            [20,21,22,23,24]]
    
    xres = 1 # Amount of plots between each glyph on the x axis
    yres = 1 # Amount of plots between each glyph on the y axis
        
    # Output file name
    outputfile = "ABCDE.png"
    

    # Initialize
    y,x = np.shape(grid)    
    new_shape = (yres*(y-1)+y,xres*(x-1)+x)
    
    emb_grid = [[np.zeros(embedding_size) for j in range(x)] for i in range(y)]
    emb_array = [[np.zeros(embedding_size) for j in range(new_shape[1])] for i in range(new_shape[0])]
    
    # Get all embeddings
    for i,row in enumerate(grid):
        for j,d in enumerate(row):
            emb_grid[i][j] = shape_embeddings[d]
    
    # Calculate latent vectors from embeddings
    if new_shape[0] == 1:
        for j,val in enumerate(grid[0]):
            if (j < x - 1):
                H = np.linspace(emb_grid[0][j],emb_grid[0][j+1],xres+2)
                for m,h in enumerate(H):
                    emb_array[0][j*(xres+1) + m] = h
    elif new_shape[1] == 1:
        for i,val in enumerate(grid):
            if (i < y - 1):
                V = np.linspace(emb_grid[i][0],emb_grid[i+1][0],yres+2)
                for n,v in enumerate(V):
                    emb_array[i*(yres+1) + n][0] = v
    else:
        for i,row in enumerate(grid):    
            for j,val in enumerate(row):
                if (i < y - 1) and (j < x - 1):
                    H0 = np.linspace(emb_grid[i][j],emb_grid[i][j+1],xres+2)
                    H1 = np.linspace(emb_grid[i+1][j],emb_grid[i+1][j+1],xres+2)
                    for m,h in enumerate(zip(H0,H1)):
                        emb_array[i*(yres+1)][j*(xres+1) + m] = h[0]
                        emb_array[(i+1)*(yres+1)][j*(xres+1) + m] = h[1]
                        V0 = np.linspace(h[0],h[1],yres+2)
                        for n,v in enumerate(V0):
                            emb_array[i*(yres+1) + n][j*(xres+1) + m] = v


    # Plotting
    temp_col = 0.5/(new_shape[1]+1)
    temp_row = 0.5/(new_shape[0]+1)
    fig = plt.figure(figsize=(new_shape[1]+1, new_shape[0]+1))
    grid_spec = gridspec.GridSpec(new_shape[0],new_shape[1], wspace=0.0, hspace=0.0,
                                  left=temp_col,right=1-temp_col,top=1-temp_row,bottom=temp_row)
    
    for i,row in enumerate(emb_array):
        for j,lat in enumerate(row):
            def predict(ps):
                _ps = np.empty((len(ps), 2 + embedding_size))
                _ps[:,:2] = ps
                _ps[:,2:] = lat
                return model_latent.predict(_ps)
            ax = plt.subplot(grid_spec[i,j])
            ax = latent_plot_sdf(predict,ax, scale=1)
            ax.axis('off')
            print(i,j)

    plt.savefig(outputfile,pad_inches=0.0,bbox_inches='tight',dpi=200)
