import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from models import GAN
from model_monitor import ModelMonitor

class Artist():

    def process_data(self, data_dir: str, img_height: int, img_width: int, batch_size: int):
        
        # Turning image dataset containing jpeg into a batch dataset
        self.raw_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            labels=None,
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        
        # Normalizing the dataset
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        normalized_ds = self.raw_ds.map(lambda x: normalization_layer(x))

        self.normalized_ds = normalized_ds
            
                
    def learn(self, gan: GAN, num_epochs: int, gen_learning_rate: float, dis_learning_rate: float, model_monitor: ModelMonitor):
        
        # Need to train the generator and the discriminator at the same time
        # Balancing act between of speed of generator and discriminator learning

        # Defining the optimisers
        gen_optimiser = Adam(learning_rate = gen_learning_rate)
        dis_optimiser = Adam(learning_rate = dis_learning_rate)

        # Defining the losses
        gen_loss = BinaryCrossentropy()
        dis_loss = BinaryCrossentropy()

        gan.compile(gen_optimiser, dis_optimiser, gen_loss, dis_loss)

        # Now training the gan as a whole
        self.hist = gan.fit(self.normalized_ds, epochs = num_epochs, callbacks = [model_monitor])

        # Saving the final gan and generator model as object attributes
        self.final_model = gan.generator_model
        self.final_gan = gan

