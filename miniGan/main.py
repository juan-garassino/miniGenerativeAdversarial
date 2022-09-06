from torch.autograd import Variable
import torch

from miniGan.manager import Manager
from miniGan.utils import noise, images_to_vectors, vectors_to_images
from miniGan.models import GeneratorNet, DiscriminatorNet
from miniGan.trainers import train_discriminator, train_generator
from miniGan.optimizers import discriminator_optimizer, generator_optimizer
from miniGan.losses import loss
from miniGan.data import mnist_data


def main(
    num_epochs=100,
    num_test_samples=16,
    batch_size=64,
    model_name="vgan",
    data_name="mnist",
    save_step=25,
):
    # Load data
    data = mnist_data()
    # Create loader with data, so that we can iterate over it
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    # Num batches
    num_batches = len(data_loader)

    test_noise = noise(num_test_samples)

    discriminator = DiscriminatorNet()

    generator = GeneratorNet()

    # Create manager instance
    manager = Manager(model_name=model_name, data_name=data_name)
    # Total number of epochs to train

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
            manager.log(d_error, g_error, epoch, n_batch, num_batches)
            # Display Progress every few batches
            if (n_batch) % 100 == 0:

                test_images = vectors_to_images(generator(test_noise))
                test_images = test_images.data

                # saves images
                manager.log_images(
                    test_images, num_test_samples, epoch, n_batch, num_batches
                )
                # Display status Logs
                manager.display_status(
                    epoch,
                    num_epochs,
                    n_batch,
                    num_batches,
                    d_error,
                    g_error,
                    d_pred_real,
                    d_pred_fake,
                )

        if (epoch + 1) % save_step == 0:
            manager.save_models(generator, discriminator, epoch)


if __name__ == "__main__":
    main(
        num_epochs=200,
        num_test_samples=32,
        batch_size=128,
        model_name="vgan",
        data_name="mnist",
        save_step=2,
    )
