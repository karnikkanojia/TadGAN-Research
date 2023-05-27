import logging

import pandas as pd
import tensorflow as tf
from mlprimitives.utils import import_object
from tensorflow.keras import Model
from orion.primitives.tadgan import TadGAN

from orion.loader.loader import TadGANDataLoader

LOGGER = logging.getLogger(__name__)

WORKERS=4
BATCH_SIZE=32
EPOCHS=20
LR=0.0002
SEED=2
NOISE_DIM=100

class LSTMGenerator(Model):
    """An LSTM based generator. It expects a sequence of noise vectors as input.

    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim

        self.lstm0 = tf.keras.layers.LSTM(units=32, return_sequences=True, input_shape=(None, in_dim))
        self.lstm1 = tf.keras.layers.LSTM(units=64, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(units=128, return_sequences=True)
        
        self.linear = tf.keras.Sequential([
            tf.keras.layers.Dense(units=out_dim),
            tf.keras.layers.Activation('tanh')
        ])

    def call(self, input):
        batch_size, seq_len = input.shape[0], input.shape[1]
        h_0 = tf.zeros(shape=(batch_size, 32))
        c_0 = tf.zeros(shape=(batch_size, 32))

        recurrent_features = self.lstm0(input, initial_state=[h_0, c_0])
        recurrent_features = self.lstm1(recurrent_features)
        recurrent_features = self.lstm2(recurrent_features)
        
        outputs = self.linear(tf.reshape(recurrent_features, shape=(batch_size*seq_len, 128)))
        outputs = tf.reshape(outputs, shape=(batch_size, seq_len, self.out_dim))
        return outputs, recurrent_features

class LSTMDiscriminator(tf.keras.Model):
    """An LSTM based discriminator. It expects a sequence as input and outputs a probability for each element.

    Args:
        in_dim: Input noise dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms

    Inputs: sequence of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, 1)
    """

    def __init__(self, in_dim):
        super(LSTMDiscriminator, self).__init__()
        self.lstm = tf.keras.layers.LSTM(units=100, return_sequences=True, input_shape=(None, in_dim))
        self.linear = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])

    def call(self, input):
        batch_size, seq_len = input.shape[0], input.shape[1]
        h_0 = tf.zeros((batch_size, 100))
        c_0 = tf.zeros((batch_size, 100))

        recurrent_features = self.lstm(input, initial_state=[h_0, c_0])
        outputs = self.linear(tf.reshape(recurrent_features, [batch_size*seq_len, 100]))
        outputs = tf.reshape(outputs, [batch_size, seq_len, 1])
        return outputs, recurrent_features

generator = LSTMGenerator(in_dim=NOISE_DIM, out_dim=1)
discriminator = LSTMDiscriminator(in_dim=1)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizerD = tf.keras.optimizers.Adam(learning_rate=LR)
optimizerG = tf.keras.optimizers.Adam(learning_rate=LR)

def discriminator_loss(real_output, fake_output):
    real_loss = loss_fn(tf.ones_like(real_output), real_output)
    fake_loss = loss_fn(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return loss_fn(tf.ones_like(fake_output), fake_output)

# @tf.function
def train_step(batch):
    noise = tf.random.normal([BATCH_SIZE, seq_dim, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_batch, _ = generator(noise)

        real_output, _ = discriminator(batch)
        fake_output, _ = discriminator(generated_batch)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    D_x = tf.reduce_mean(real_output).numpy()
    D_G_z1 = tf.reduce_mean(fake_output).numpy()
    D_G_z2 = tf.reduce_mean(real_output).numpy()
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    optimizerG.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    optimizerD.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return D_x, D_G_z1, D_G_z2, disc_loss, gen_loss

def train(dataset, epochs):
  for epoch in range(epochs):
    for image_batch in dataset:
        D_x, D_G_z1, D_G_z2, errD, errG = train_step(image_batch)
        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' 
            % (epoch, epochs, errD, errG, D_x, D_G_z1, D_G_z2), end='')
        print()



if __name__ == "__main__":
    tf.random.set_seed(SEED)
    batch_size = 16
    seq_len = 32
    noise_dim = 100
    seq_dim = 4

    # generator = LSTMGenerator(noise_dim, seq_dim)
    # discriminator = LSTMDiscriminator(seq_dim)
    # noise = tf.random.normal(shape=(8, 16, noise_dim))
    # gen_out, _ = generator(noise)
    # dis_out, _ = discriminator(gen_out)

    # print("Noise: ", noise.shape)
    # print("Generator output: ", gen_out.shape)
    # print("Discriminator output: ", dis_out.shape)
    data = pd.read_csv('540821.csv', usecols=['Date', 'No. of Trades'], parse_dates=['Date'])
    data.rename(columns={'No. of Trades': 'value', 'Date': 'timestamp'}, inplace=True)
    data.timestamp = (data.timestamp - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    data.head()
    data_loader = TadGANDataLoader(data, shuffle=True)
    dataset = data_loader.get_tfdataset()
    train(dataset, 10)