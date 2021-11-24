import tensorflow as tf
import os
from os.path import join
import json

human_dataset_path = r'E:\image datasets\thumbnails128x128'
pasta_dataset_path = r'E:\image datasets\spaghetti'

def scale_image(image):
    return tf.cast(image, tf.float32)/127.5-1.0

def read_png(path, size=None):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=3)
    if size:
        img = tf.image.resize(img, size)
    img = scale_image(img)
    return img


def read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=3)
    # img = tf.image.resize(img, [256, 256])
    img = scale_image(img)
    return img


def make_human_dataset(batch_size, test_batch_size=None):
    human_train_paths = []
    human_test_paths = []
    
    if test_batch_size is None:
        test_batch_size = batch_size

    for d in os.listdir(human_dataset_path):
        if not os.path.isdir(join(human_dataset_path, d)):
            continue
        if d == '69000':
            print('leaving \'69000\' directory for validation')
            for f in os.listdir(join(human_dataset_path, d)):
                human_test_paths.append(join(human_dataset_path, d, f))
            continue
        for f in os.listdir(join(human_dataset_path, d)):
            human_train_paths.append(join(human_dataset_path, d, f))

    read_file = lambda x: read_png(x, size=[256, 256])
    train_ds = tf.data.Dataset.from_tensor_slices(human_train_paths)
    train_ds = train_ds.map(read_file, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=1000).batch(
        batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(human_test_paths)
    test_ds = test_ds.map(read_file, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(test_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, test_ds


def make_pasta_dataset(batch_size):
    pasta_paths = []
    for d in os.listdir(pasta_dataset_path):
        if not join(pasta_dataset_path, d).endswith('jpg'):
            continue

        pasta_paths.append(join(pasta_dataset_path, d))
    
    
    random_flip = tf.keras.layers.RandomFlip()
    random_translate = tf.keras.layers.RandomTranslation([0, 0.2], [0, 0.5])
    random_rot = tf.keras.layers.RandomRotation([0, 1])
    random_zoom = tf.keras.layers.RandomZoom([-0.1, 0.1])
    random_crop = tf.keras.layers.RandomCrop(256, 256)
    
    def preprocess(x):
        x = random_flip(x)
        x = random_translate(x)
        x = random_rot(x)
        x = random_zoom(x)
        x = random_crop(x)
        return x
        
    print(pasta_paths)
    train_ds = tf.data.Dataset.from_tensor_slices(pasta_paths)
    train_ds = train_ds.map(read_jpg, num_parallel_calls=tf.data.AUTOTUNE).cache().repeat()
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=batch_size).batch(
        batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)


    return train_ds


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    pasta = make_pasta_dataset(64)
    fig, ax = plt.subplots(8, 8)
    for x in pasta:
        for i in range(64):
            row = i//8
            col = i%8
            ax[row, col].imshow((x[i]+1)/2)
            ax[row, col].axis('off')
        break

    plt.show()