from miniGan.utils import vectors_to_images, noise
from miniGan.models import GeneratorNet
from miniGan.manager import Manager

import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def predict(status="generator-25", num_images=16, last=True, plot=False):

    manager = Manager(model_name="vgan", data_name="mnist")

    generator = GeneratorNet()

    generator.load_state_dict(manager.load_models(status), strict=True)

    test_noise = noise(num_images)

    predicted_images = vectors_to_images(generator(test_noise)).data

    if plot:

        manager.make_snapshot(
            predicted_images.detach().numpy(),
            16,
            epoch=None,
            n_batch=None,
            num_batches=None,
            predict=True,
        )

        nrows = int(np.sqrt(num_images))

        grid = vutils.make_grid(
            predicted_images, nrow=nrows, normalize=True, scale_each=True
        )

        return plt.imshow(np.moveaxis(grid.numpy(), 0, -1))

    return predicted_images


def interpolate():
    pass


if __name__ == "__main__":
    pass
