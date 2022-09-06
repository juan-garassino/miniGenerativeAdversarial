from torchvision import transforms
from torchvision import datasets
from torch.autograd import Variable
import torch

from miniGan.logger import Logger
from miniGan.utils import noise, images_to_vectors, vectors_to_images
from miniGan.models import GeneratorNet, DiscriminatorNet
from miniGan.trainers import train_discriminator, train_generator
from miniGan.optimizers import discriminator_optimizer, generator_optimizer
from miniGan.losses import loss
from miniGan.data import mnist_data

# Load data
data = mnist_data()
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)

num_test_samples = 16

test_noise = noise(num_test_samples)

discriminator = DiscriminatorNet()

generator = GeneratorNet()

# Create logger instance
logger = Logger(model_name="vgan", data_name="mnist")
# Total number of epochs to train
num_epochs = 50

for epoch in range(num_epochs):
    for n_batch, (real_batch, _) in enumerate(data_loader):
        N = real_batch.size(0)
        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch))
        # Generate fake data and detach (so gradients are not calculated for generator)
        fake_data = generator(noise(N)).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(
            discriminator,
            loss,
            discriminator_optimizer(discriminator)[1],
            real_data,
            fake_data,
        )

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(N))
        # Train G
        g_error = train_generator(
            discriminator, loss, generator_optimizer(generator)[1], fake_data
        )
        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        # Display Progress every few batches
        if (n_batch) % 100 == 0:
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data
            logger.log_images(
                test_images, num_test_samples, epoch, n_batch, num_batches
            )
            # Display status Logs
            logger.display_status(
                epoch,
                num_epochs,
                n_batch,
                num_batches,
                d_error,
                g_error,
                d_pred_real,
                d_pred_fake,
            )

            logger.save_models(generator, discriminator, epoch)
