from miniGan.logger import Logger
from miniGan.utils import vectors_to_images, noise
from miniGan.models import GeneratorNet
from miniGan.logger import Logger

import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def predict(status, last=True, plot=False):
    logger = Logger(model_name="vgan", data_name="mnist")

    generator = GeneratorNet()
    generator.load_state_dict(logger.load_models(status), strict=True)

    num_images = 16
    test_noise = noise(num_images)
    predicted_images = vectors_to_images(generator(test_noise)).data

    if plot:

        logger.log_images(predicted_images.detach().numpy(), 16, 1, 1, 1)

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
