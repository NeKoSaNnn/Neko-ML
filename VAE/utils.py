import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import norm


def plot_latent_image(model, latent_dim, patch_count, patch_side_size):
    # 2σ原则
    xs = norm.ppf(np.linspace(0.05, 0.95, patch_count))
    ys = norm.ppf(np.linspace(0.05, 0.95, patch_count))
    image_size = [patch_count * patch_side_size, patch_count * patch_side_size]
    image = np.zeros(image_size)

    for x_index, x in enumerate(xs):
        for y_index, y in enumerate(ys):
            z = np.tile(np.array([[x, y]]), latent_dim).reshape(-1, latent_dim)
            z = torch.Tensor(z).cuda()
            decoder_image = model.decoder(z)
            decoder_image = decoder_image.reshape(-1, patch_side_size, patch_side_size)
            image[x_index * patch_side_size:(x_index + 1) * patch_side_size,
            y_index * patch_side_size:(y_index + 1) * patch_side_size] = decoder_image[0].cpu().detach().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="gray")
    plt.show()
