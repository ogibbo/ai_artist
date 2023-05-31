import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback

# source: https://github.com/nicknochnack/GANBasics
class ModelMonitor(Callback):
    def __init__(self, latent_dim, save_imgs_fp, num_img=1):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.save_imgs_fp = save_imgs_fp

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal((self.num_img, self.latent_dim))
        generated_images = self.model.generator_model(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join(self.save_imgs_fp, f'img_{epoch}_{i}.png'))