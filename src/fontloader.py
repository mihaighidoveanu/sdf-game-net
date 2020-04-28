from _fontloader_cffi import ffi, lib
import numpy as np
import matplotlib.pyplot as plt

font_path = "../fonts/times.ttf".encode('ascii')
char = 'a'
scale = 5

lib.load_font(font_path)
print(ffi.string(lib.fontpath))

index = lib.get_glyph_index(ord(char))

box = lib.get_glyph_box(index, scale)
print(box)
print(box.x0, box.y0, box.x1, box.y1)

# display character
# generate test points in an n x n grid, so we can show the result as an image
size = 100 
canvas = np.array([ [ x, y ] for x in range(size) for y in range(size) ])
canvas = 5*(canvas/100 - 0.5)

canvas = [lib.get_glyph_distance(index, scale, p[0], p[1]) for p in canvas]
canvas = np.array(canvas)

X, Y = np.meshgrid(range(size), range(size))
cp = plt.contourf(X, Y, canvas.reshape((size, size)))
plt.colorbar(cp)
plt.show()