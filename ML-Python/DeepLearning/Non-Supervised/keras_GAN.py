from keras.api.layers import LeakyReLU, Dense, Input
from keras.api.models import Sequential, Model
import numpy as np

# Define the generator model
def build_generator():
    model = Sequential()
    model.add(Dense(128, input_dim=100))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(784, activation='tanh'))
    return model

# Define the discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Dense(128, input_shape=(784,)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

# Build the generator
generator = build_generator()

# Create the GAN by combining the generator and discriminator
discriminator.trainable = False
gan_input = Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = Model(gan_input, gan_output)

# Compile the GAN
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training the GAN
def train_gan(gan, generator, discriminator, x_train, epochs=400,
              batch_size=128):
    for epoch in range(epochs):
        # Generate random noise as input for the generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        generated_image = generator.predict(noise)

        # Get a random set of real images
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]

        # Labels for real and fake images
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # Train the discriminator on real and fake images separately
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_image, fake_labels)

        # Calculate the avergae loss for the discriminator
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Generate new noise and train the generator through the GAN model (note:
        # We train the generator via the GAN model, where the discriminator's weights are frozen)

        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, real_labels, return_dict=True)

        # Print the progress every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch} - Discriminator Loss: {d_loss[0]},
                   Generator Loss: {g_loss['loss']}')
        
        return d_loss, g_loss