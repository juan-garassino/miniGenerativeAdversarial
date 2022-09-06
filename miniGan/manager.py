import os
import numpy as np
import errno
import torchvision.utils as vutils

from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch

from colorama import Fore, Style
from datetime import datetime


"""
    TensorBoard Data will be stored in './runs' path
"""


class Manager:  # make manager work with and with out epochs
    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = "{}_{}".format(model_name, data_name)
        self.data_subdir = "{}/{}".format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment)

    def log(self, d_error, g_error, epoch, n_batch, num_batches):

        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()

        step = Manager.manager_step(epoch, n_batch, num_batches)
        self.writer.add_scalar("{}/D_error".format(self.comment), d_error, step)
        self.writer.add_scalar("{}/G_error".format(self.comment), g_error, step)

    def single_snapshot(
        self, fig, epoch=None, n_batch=None, comment=""
    ):  # here option to save without epoch!!!

        if epoch and n_batch:
            out_dir = "./results/images/{}".format(self.data_subdir)
            Manager.make_directory(out_dir)
            fig.savefig(
                "{}/{}_epoch_{}_batch_{}.png".format(out_dir, comment, epoch, n_batch)
            )

        if not epoch and n_batch:
            out_dir = "./results/images/{}".format(self.data_subdir)
            Manager.make_directory(out_dir)

            now = datetime.now().strftime("%d-%m-%Y-%H-%M")

            fig.savefig("image[{}].png".format(now))

    def save_torch_images(
        self, horizontal_grid, grid, epoch=None, n_batch=None, plot_horizontal=True, predict=False,
    ):

        if predict:
            out_dir = "./results/generated/{}".format(self.data_subdir)

        if not predict:
            out_dir = "./results/images/{}".format(self.data_subdir)


        Manager.make_directory(out_dir)

        # Plot and save horizontal
        fig = plt.figure(figsize=(25, 25))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis("off")

        if plot_horizontal:
            display.display(plt.gcf())
            self.single_snapshot(fig, epoch, n_batch, "horizontal")
            plt.close()

        # Save squared
        fig = plt.figure(figsize=(25, 25))
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis("off")

        if not plot_horizontal:
            self.single_snapshot(fig, epoch, n_batch, "square")
            plt.close()

    def make_snapshot(
        self,
        images,
        num_images,
        epoch=None,
        n_batch=None,
        num_batches=None,
        format="NCHW",
        normalize=True,
        predict=False
    ):
        """
        input images are expected in format (NCHW)
        """
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)

        if format == "NHWC":
            images = images.transpose(1, 3)

        step = Manager.manager_step(epoch, n_batch, num_batches)
        img_name = "{}/images{}".format(self.comment, "")

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(images, normalize=normalize, scale_each=True)
        # Make vertical grid from image tensor
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(images, nrow=nrows, normalize=True, scale_each=True)

        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, horizontal_grid, step)

        # Save plots
        self.save_torch_images(horizontal_grid,
                               grid,
                               epoch,
                               n_batch,
                               predict=predict)

    def display_status(
        self,
        epoch,
        num_epochs,
        n_batch,
        num_batches,
        d_error,
        g_error,
        d_pred_real,
        d_pred_fake,
    ):

        # var_class = torch.autograd.variable.Variable
        if isinstance(d_error, torch.autograd.Variable):
            d_error = d_error.data.cpu().numpy()
        if isinstance(g_error, torch.autograd.Variable):
            g_error = g_error.data.cpu().numpy()
        if isinstance(d_pred_real, torch.autograd.Variable):
            d_pred_real = d_pred_real.data
        if isinstance(d_pred_fake, torch.autograd.Variable):
            d_pred_fake = d_pred_fake.data

        print(
            "\n✅ "
            + Fore.MAGENTA
            + "Epoch: [{}/{}], Batch Num: [{}/{}]".format(
                (epoch + 1), num_epochs, n_batch, num_batches
            )
            + Style.RESET_ALL
        )
        print(
            "\n✅ "
            + Fore.GREEN
            + "Discriminator Loss: {:.4f}, Generator Loss: {:.4f}".format(
                d_error, g_error
            )
            + Style.RESET_ALL
        )
        print(
            "\n✅ "
            + Fore.CYAN
            + "D(x): {:.4f}, D(G(z)): {:.4f}".format(
                d_pred_real.mean(), d_pred_fake.mean()
            )
            + Style.RESET_ALL
        )

    def save_models(self, generator, discriminator, epoch):
        out_dir = "./results/models/{}".format(self.data_subdir)
        Manager.make_directory(out_dir)
        torch.save(generator.state_dict(), "{}/checkpoint-generator@{}".format(out_dir, (epoch + 1)))
        torch.save(
            discriminator.state_dict(), "{}/checkpoint-critic@{}".format(out_dir, (epoch + 1))
        )

        print(
            "\n✅"
            + Fore.YELLOW
            + "Saved model for epoch {}".format((epoch + 1))
            + Style.RESET_ALL
        )

    def load_models(self, *args, last=True):
        parent = os.path.join(os.path.dirname(__file__), "..")
        input_dir = f"{parent}/results/models/{self.data_subdir}/{args[0]}"
        generator = torch.load(input_dir, map_location=lambda storage, loc: storage)

        print(
            "\n✅"
            + Fore.YELLOW
            + "Loaded model from {}...".format(input_dir[:59])
            + Style.RESET_ALL
        )

        return generator

    def close(self, generator, discriminator, out_dir, epoch):
        self.writer.close()
        torch.save(generator.state_dict(), "{}/G_epoch_{}".format(out_dir, epoch))
        torch.save(discriminator.state_dict(), "{}/D_epoch_{}".format(out_dir, epoch))

    # Private Functionality

    @staticmethod
    def manager_step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def make_directory(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


if __name__ == "__main__":
    pass
