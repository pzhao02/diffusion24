import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

# Load MNIST data
(X_train, y_train), (_, _) = mnist.load_data()
X_train = X_train[y_train == 9]
# X_train = X_train[(y_train == 9) | (y_train == 6)]

# Normalize data to [-1, 1]
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=-1)

# Define dimensions
img_shape = (28, 28, 1)
latent_dim = 100


# Build Generator
def build_generator():
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(np.prod(img_shape), activation="tanh"))
    model.add(Reshape(img_shape))
    return model


# Build Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation="sigmoid"))
    return model


# Compile GAN components
optimizer = Adam(0.0002, 0.5)
discriminator = build_discriminator()
discriminator.compile(
    loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)

generator = build_generator()
z = tf.keras.Input(shape=(latent_dim,))
img = generator(z)

discriminator.trainable = False
validity = discriminator(img)

gan = tf.keras.Model(z, validity)
gan.compile(loss="binary_crossentropy", optimizer=optimizer)

# Training
batch_size = 64
epochs = 4000
sample_interval = 500

for epoch in range(epochs):
    # Train Discriminator
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_imgs = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(imgs, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % sample_interval == 0:
        print(
            f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]"
        )

# # Generate and visualize images
# def generate_images(generator, latent_dim, examples=100):
#     noise = np.random.normal(0, 1, (examples, latent_dim))
#     gen_imgs = generator.predict(noise)

#     gen_imgs = 0.5 * gen_imgs + 0.5
#     fig, axs = plt.subplots(int(np.sqrt(examples)), int(np.sqrt(examples)), figsize=(examples * 2, 2))
#     for i in range(examples):
#         axs[i].imshow(gen_imgs[i, :, :, 0], cmap='gray')
#         axs[i].axis('off')
#     plt.show()

# # Generate images after training
# generate_images(generator, latent_dim)


# Helper function to plot support points (generated images)
def plot_support_points(support_points, title):
    grid_size = int(np.sqrt(len(support_points)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        # If support_points have a channel dimension, remove it for plotting
        image = (
            support_points[i].reshape(28, 28)
            if support_points[i].shape[-1] == 1
            else support_points[i]
        )
        ax.imshow(image, cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    plt.show()


# Generate images after training using the generator
def generate_and_plot(generator, latent_dim, num_images=100, title="Generated Images"):
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    gen_imgs = generator.predict(noise)
    # Rescale images from [-1, 1] to [0, 1]
    # print(f"min={gen_imgs.min():.4f}, max={gen_imgs.max():.4f}")
    gen_imgs = 0.5 * gen_imgs + 0.5
    # print(gen_imgs[:10])
    plot_support_points(gen_imgs, title)

# Generate and visualize images using the new plotting function
generate_and_plot(
    generator, latent_dim, num_images=100, title="GANs: Generated MNIST Digit 6_9s"
)
