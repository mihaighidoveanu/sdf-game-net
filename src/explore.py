import fire
import os
import utils
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

def explore(input_dir):
    loss = utils.ScaledLoss()

    for experiment in os.listdir(input_dir):
        if experiment == '.DS_Store':
            continue
        font_dir = os.path.join(input_dir, experiment, 'fonts')
        if not os.path.isdir(font_dir):
            continue
        glyphs = [] 
        for font in os.listdir(font_dir):
            if font == '.DS_Store':
                continue
            pred_dir = os.path.join(font_dir, font, 'draw', 'pred')
            gt_dir = os.path.join(font_dir, font, 'draw', 'gt')
            for glyph in os.listdir(pred_dir):
                if glyph.split('_')[0] == 'train':
                    continue
                gname = glyph.split('_')[2]



if __name__ == '__main__':
    fire.Fire(explore)
