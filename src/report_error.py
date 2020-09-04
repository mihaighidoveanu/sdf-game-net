import fire
import os
import utils
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

def report(input_dir):
    loss = utils.ScaledLoss()

    logfile = os.path.join(input_dir, 'error.txt')
    f = open(logfile, 'w')

    for experiment in os.listdir(input_dir):
        if experiment == '.DS_Store':
            continue
        errors = []
        font_dir = os.path.join(input_dir, experiment, 'fonts')
        if not os.path.isdir(font_dir):
            continue
        for font in os.listdir(font_dir):
            if font == '.DS_Store':
                continue
            pred_dir = os.path.join(font_dir, font, 'draw', 'pred')
            gt_dir = os.path.join(font_dir, font, 'draw', 'gt')
            for glyph in os.listdir(pred_dir):
                if glyph.split('_')[0] == 'train':
                    continue
                pred_path = os.path.join(pred_dir, glyph)
                gt_path = os.path.join(gt_dir, glyph)

                pred = Image.open(pred_path)
                pred = np.asarray(pred) / 255

                gt = Image.open(gt_path)
                gt = np.asarray(gt) / 255

                error = loss(gt, pred)
                errors.append(error)

        final_error = np.mean(errors)

        print(f"Report for {experiment} - {final_error}")
        f.write(f"{experiment}, {final_error}\n")

    f.close()

if __name__ == '__main__':
    fire.Fire(report)
