import torch

import train
import utils

if __name__ == "__main__":
    train.train()

    # model = torch.load("./save_model/latent_dim-2/VAE_epoch_100_2021-12-26-12-11-14.pth")
    # if hasattr(model, "latent_dim"):
    #     utils.plot_latent_image(model, model.latent_dim, patch_count=20, patch_side_size=28)
    # else:
    #     utils.plot_latent_image(model, int(input("latent_dim=")), patch_count=20, patch_side_size=28)
