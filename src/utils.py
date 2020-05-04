    
import numpy as np
import matplotlib.pyplot as plt



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