from miniGenerativeAdversarial.utils.utils import vectors_to_images, noise
from miniGenerativeAdversarial.models.model import GeneratorNet
from miniGenerativeAdversarial.managers.manager import Manager

from datetime import datetime
from colorama import Style, Fore
import os
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
            num_images,
            epoch=None,
            n_batch=None,
            num_batches=None,
            plot_horizontal=True,
            plot_square=True,
        )

        nrows = int(np.sqrt(num_images))

        grid = vutils.make_grid(
            predicted_images, nrow=nrows, normalize=True, scale_each=True
        )

        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))

        out_dir = os.path.join(
            os.environ.get("HOME"), "Results", "miniGan", "predictions"
        )

        if int(os.environ.get("COLAB")) == 1:

            out_dir = os.path.join(
                os.environ.get("HOME"),
                "..",
                "content",
                "results",
                "miniGan",
                "predictions",
            )

        Manager.make_directory(out_dir)

        now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        picture_name = "{}/image[{}].png".format(out_dir, now)

        plt.savefig(picture_name)

        print(
            "\n🔽 "
            + Fore.BLUE
            + f"Generated picture {picture_name} at {out_dir}"
            + Style.RESET_ALL
        )

    return predicted_images


def interpolate():
    pass


if __name__ == "__main__":
    pass
