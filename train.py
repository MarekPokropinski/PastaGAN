from pasta_gan import PastaGAN
from data_preparation import make_human_dataset, make_pasta_dataset
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

BATCH_SIZE = 4

human_train_dataset, human_test_dataset = make_human_dataset(BATCH_SIZE, 8)
pasta_dataset = make_pasta_dataset(BATCH_SIZE)

gan = PastaGAN([256, 256], BATCH_SIZE, 100)
gan.load('model')

for x in human_test_dataset:
    vis_imgs = x
    break

print(vis_imgs.shape)

fig, ax = plt.subplots(2, 8)
for i in range(2):
    for j in range(8):
        ax[i,j].axis('off')


train_ds = tf.data.Dataset.zip((human_train_dataset, pasta_dataset))

step = 0
while True:
    for human, pasta in train_ds:
        if human.shape != pasta.shape:
            print(human.shape, pasta.shape)
            continue
        losses = gan.train_step(human, pasta)
        print(losses)

        if step % 20 == 0:
            gan.save('model')
            pred = gan.generator.predict(vis_imgs)

            for j in range(8):
                ax[0, j].imshow((vis_imgs[j]+1.0)/2.0)
                ax[1, j].imshow(np.clip((pred[j]+1.0)/2.0, 0.0, 1.0))

            plt.savefig(f'out/{step}.png')

        step += 1
