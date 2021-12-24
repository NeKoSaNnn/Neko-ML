import utils

if __name__ == "__main__":
    model = torch.load("...")
    if hasattr(model, "latent_dim"):
        utils.plot_latent_image(model, model.latent_dim, patch_count=30, patch_side_size=28)
    else:
        utils.plot_latent_image(model, int(input("latent_dim=")), patch_count=30, patch_side_size=28)
