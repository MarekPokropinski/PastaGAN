import matplotlib.pyplot as plt
from pasta_gan import PastaGAN
from data_preparation import make_human_dataset

BATCH_SIZE = 1

human_train_dataset, human_test_dataset = make_human_dataset(BATCH_SIZE, 8)

fig, ax = plt.subplots(4, 4)

pastaGAN = PastaGAN([256, 256], 8, 8)
pastaGAN.load('model')

predictions = {}

for x in human_test_dataset:
    vis_imgs = x
    break

pred = pastaGAN.generator.predict(vis_imgs, batch_size=1)

for i in range(0, 16, 2):
    k=i//2
    ax[i%4, i//4].imshow((vis_imgs[i//2]+1)/2)
    ax[(i+1)%4, (i+1)//4].imshow((pred[i//2]+1)/2)
    ax[i%4, i//4].axis('off')
    ax[(i+1)%4, (i+1)//4].axis('off')

plt.show()