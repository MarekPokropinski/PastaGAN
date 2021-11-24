import tensorflow as tf
from image_loss import PerceptualLoss
from models import ae_model, patch_gan_discriminator
import numpy as np
import os
import random


class ImagePool:
    def __init__(self, size) -> None:
        self.buf = []
        self.size = size

    def add(self, image):
        self.buf.append(image)

    def sample_images(self, images):
        n = images.shape[0]
        self.buf.extend(images)

        ret = []
        new_buf = []
        idx = random.sample(range(len(self.buf)), k=n)

        for i in range(len(self.buf)):
            if i in idx:
                ret.append(self.buf[i])
            else:
                new_buf.append(self.buf[i])

        self.buf = new_buf

        for element in ret:
            if len(self.buf) < self.size:
                self.buf.append(element)

        return np.stack(ret)

    def clear(self):
        self.buf.clear()


class PastaGAN:
    def __init__(self, image_size, batch_size, buf_size) -> None:
        self.generator = ae_model((*image_size, 3))
        self.discriminator = patch_gan_discriminator()

        self.generator_learning_rate = 0.00008
        self.generator_pretrain_learning_rate = 0.0001

        self.image_size = image_size
        self.buf_size = buf_size
        self.batch_size = batch_size

        self.fake_pool = ImagePool(buf_size)

        self.gen_optim = tf.keras.optimizers.Adam(learning_rate=0.00008)
        self.disc_optim = tf.keras.optimizers.Adam(learning_rate=0.00016)
        self.vgg = PerceptualLoss((*image_size, 3))

        self.lambda_adv = 300
        self.lambda_con = 1.5
        self.lambda_style = 3

    def adversarial_loss(self, prediction, is_real):
        if is_real:
            return tf.reduce_mean(tf.keras.losses.mean_squared_error(1.0, prediction))
        else:
            return tf.reduce_mean(tf.keras.losses.mean_squared_error(0.0, prediction))


    @tf.function
    def train_generator(self, real_human, style_img):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.generator.trainable_weights)
            fake = self.generator(real_human)

            adversarial_loss = self.lambda_adv*self.adversarial_loss(
                self.discriminator(fake), True)
            
            # adversarial_loss = 0

            vgg_human = self.vgg(real_human)
            vgg_anime = self.vgg(style_img)
            vgg_fake = self.vgg(fake)

            content_loss = self.lambda_con * \
                self.vgg.content_loss(vgg_fake, vgg_human)
            style_loss = self.lambda_style * \
                self.vgg.style_loss(vgg_fake, vgg_anime)

            total_loss = adversarial_loss + content_loss + style_loss

        vars = tape.watched_variables()
        grads = tape.gradient(total_loss, vars)
        self.gen_optim.apply_gradients(zip(grads, vars))
        return fake, adversarial_loss, content_loss, style_loss

    @tf.function
    def train_discriminator(self, real, fake):
        with tf.GradientTape() as tape:
            loss_D_real = self.adversarial_loss(
                self.discriminator(real), True)
            loss_D_fake = self.adversarial_loss(
                self.discriminator(fake), False)
            loss_D = (loss_D_real+loss_D_fake) * self.lambda_adv

        vars = tape.watched_variables()
        grads = tape.gradient(loss_D, vars)

        self.disc_optim.apply_gradients(zip(grads, vars))
        return loss_D

    def train_step(self, real_A, real_B):
        fake, adversarial_loss, content_loss, style_loss = self.train_generator(
            real_A, real_B)

        fake = self.fake_pool.sample_images(fake.numpy())
        D_loss = self.train_discriminator(real_B, fake)
        # return [x.numpy() for x in (content_loss, style_loss)]
        return [x.numpy() for x in (adversarial_loss, D_loss, content_loss, style_loss)]

    def save(self, filename):
        self.discriminator.save_weights(
            os.path.join(filename, 'discriminator' + '.h5'))

        self.generator.save_weights(
            os.path.join(filename, 'generator' + '.h5'))

        # with open(os.path.join(filename, 'optimizer.json'), 'w') as f:
        #     json.dump(self.optim.get_config(), f)

    def load(self, filename):
        self.discriminator.build((1, 256, 256, 3))

        self.discriminator.load_weights(
            os.path.join(filename, 'discriminator' + '.h5'))

        self.generator.load_weights(
            os.path.join(filename, 'generator' + '.h5'))
