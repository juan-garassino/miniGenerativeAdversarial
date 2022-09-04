import torch


def discriminator_optimizer(discriminator):
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    return discriminator, d_optimizer


def generator_optimizer(generator):
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    return generator, g_optimizer
