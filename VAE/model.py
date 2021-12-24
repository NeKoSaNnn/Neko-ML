import torch
from torch import sigmoid, exp, randn_like, sum
from torch.nn import Module, Linear
from torch.nn.functional import relu, binary_cross_entropy


class VAE(Module):
    def __init__(self, input_size, encoder_dim, decoder_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.encoder_fc1 = Linear(input_size, encoder_dim)
        self.mean_layer = Linear(encoder_dim, latent_dim)
        self.log_variance_layer = Linear(encoder_dim, latent_dim)
        self.z_fc = Linear(latent_dim, decoder_dim)
        self.decoder_fc1 = Linear(decoder_dim, input_size)

    def get_z_fc(self, mean, log_variance):
        stdv = exp(0.5 * log_variance)
        eps = randn_like(stdv)
        return mean + eps * stdv

    def decoder(self, z):
        res = relu(self.z_fc(z))
        res = sigmoid(self.decoder_fc1(res))
        return res

    def forward(self, x):
        res = relu(self.encoder_fc1(x.reshape(-1, self.input_size)))
        # 均值和方差层不需要激活
        mean = self.mean_layer(res)
        log_variance = self.log_variance_layer(res)
        z = self.get_z_fc(mean, log_variance)
        res = self.decoder(z)
        return res, mean, log_variance

    def kl_loss(self, mean, log_variance):
        return -0.5 * sum(1 + log_variance - mean.pow(2) - exp(log_variance))

    def reconstruct_loss(self, decoder_images, ori_images):
        return binary_cross_entropy(decoder_images, ori_images, reduction="sum")
