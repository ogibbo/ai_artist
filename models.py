import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, Dropout, Dense, Reshape, UpSampling2D, Conv2D, Flatten


# Implementation for the generator model
class Generator():

    def __init__(self, latent_dim: int, image_size: int):

        self.input_size = latent_dim
        
        # Defining the model
        self.model = Sequential()
        self.model.add(Dense((image_size // 4) * (image_size // 4) * 256, input_dim = latent_dim))
        self.model.add(LeakyReLU(0.2))
        self.model.add(Reshape((image_size // 4, image_size // 4, 256)))

        self.model.add(UpSampling2D())
        self.model.add(Conv2D(128, 5, padding='same'))
        self.model.add(LeakyReLU(0.2))
        
        self.model.add(UpSampling2D())
        self.model.add(Conv2D(64, 5, padding='same'))
        self.model.add(LeakyReLU(0.2))
        
        # Conv layer to get to one channel
        self.model.add(Conv2D(3, 5, padding='same', activation='sigmoid'))
        
        

# Implementation of the discriminator model
class Discriminator():

    def __init__(self, image_height: int, image_width: int):

        self.model = Sequential()

        # First Conv Block
        self.model.add(Conv2D(64, 5, input_shape = (image_height, image_width,3)))
        self.model.add(LeakyReLU(0.2))
        self.model.add(Dropout(0.4))
        
        self.model.add(Conv2D(128, 5))
        self.model.add(LeakyReLU(0.2))
        self.model.add(Dropout(0.4))
        
        self.model.add(Flatten())
        self.model.add(Dropout(0.4))
        self.model.add(Dense(1, activation='sigmoid'))

        

class GAN(Model):
    
    def __init__(self, generator, discriminator):
        super().__init__()

        self.generator_input_size = generator.input_size
        self.generator_model = generator.model
        self.discriminator_model = discriminator.model

    def compile(self, gen_optimiser, dis_optimiser, gen_loss, dis_loss):
        
        # Compile with the base class
        super().compile(run_eagerly=True)

        # Storing the optimisers and losses as part of object attributes
        self.gen_optimiser = gen_optimiser
        self.dis_optimiser = dis_optimiser
        self.gen_loss = gen_loss
        self.dis_loss  = dis_loss

    def train_step(self, batch):

        # Storing the actual data (from the real dataset)
        real_data = batch
        # Creating fake data via the generator model and storing this as fake data
        fake_data = self.generator_model(tf.random.normal((batch.shape[0],self.generator_input_size)), training=False)

        # Need to train the discriminator
        with tf.GradientTape(persistent=True) as d_tape:
            # Passing the real and fake data through discriminator
            dis_pred_real = self.discriminator_model(real_data, training = True)
            dis_pred_fake = self.discriminator_model(fake_data, training = True)
            # Combining the outputs
            dis_pred_realplusfake = tf.concat([dis_pred_fake, dis_pred_real], axis = 0)

            # Need to create labels for this real and fake data (=1 if fake data)
            true_labels = tf.concat([tf.ones_like(dis_pred_fake), tf.zeros_like(dis_pred_real)], axis = 0)

            # Adding output noise to make it a bit harder for the discriminator
            noise_real = 0.15*tf.random.uniform(tf.shape(dis_pred_real))
            noise_fake = -0.15*tf.random.uniform(tf.shape(dis_pred_fake))
            dis_pred_realplusfake += tf.concat([noise_fake, noise_real], axis = 0)

            # Calculating the overall loss for the discriminative model
            total_dis_loss = self.dis_loss(true_labels, dis_pred_realplusfake)

        # Then we want to backpropogate to calcualte gradients
        dgrad = d_tape.gradient(total_dis_loss, self.discriminator_model.trainable_variables)
        # Then use this to update the weights for our discriminator model
        self.dis_optimiser.apply_gradients(zip(dgrad, self.discriminator_model.trainable_variables))

        # Now we train the generator
        with tf.GradientTape(persistent=True) as g_tape:
            
            # Generate new data
            gen_data = self.generator_model(tf.random.normal((batch.shape[0],self.generator_input_size)), training = True)

            # Creating the predicted labels (Running generated data through the discriminator model)
            # Dont want discrimnator to learn whilst training generator
            # Discriminator outputs 1 if it believes the data is fake
            predicted_labels = self.discriminator_model(gen_data, training = False)

            # Calculate the loss
            # Here we reward the generator if the discriminator outputs 0 (ie believes its real)
            total_gen_loss = self.gen_loss(tf.zeros_like(predicted_labels), predicted_labels)

        # Calculate the gradients
        ggrad = g_tape.gradient(total_gen_loss, self.generator_model.trainable_variables)
        # Use the gradients to update the parameters
        self.gen_optimiser.apply_gradients(zip(ggrad, self.generator_model.trainable_variables))

        return {"discriminator_loss":total_dis_loss, "generator_loss":total_gen_loss}