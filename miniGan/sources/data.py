from torchvision import transforms
from torchvision import datasets


def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )  # ((.5, .5, .5), (.5, .5, .5))
    out_dir = "./miniGan/data"
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)


def get_custom_data():
    pass
