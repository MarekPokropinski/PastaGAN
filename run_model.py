import matplotlib.pyplot as plt
from pasta_gan import PastaGAN
from imageio import imread
import numpy as np
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

BATCH_SIZE = 1

image = imread(args.path)/127.5-1.0
image = tf.image.resize(image, [256, 256], method='bicubic')[:, :, :3]
image = image[np.newaxis, ...]

fig, ax = plt.subplots(2)

pastaGAN = PastaGAN([256, 256], 8, 8)
pastaGAN.load('model')

predictions = {}

pred = pastaGAN.generator.predict(image, batch_size=1)

ax[0].imshow((image[0]+1)/2)
ax[1].imshow((pred[0]+1)/2)
ax[0].axis('off')
ax[1].axis('off')

plt.show()