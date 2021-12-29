import os.path as osp
import sys

import torch
import torch.optim as optim
import torchvision

if hasattr(sys.modules["__main__"], "get_ipython"):
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import dataset
from model import VAE
from utils import utils

utils = utils()


def train():
    encoder_dim = 400
    decoder_dim = 400
    latent_dim = 2
    dataset_type = "mnist"
    dataset_path = "/dataset"
    save_val_path = "./val_result"
    save_model_path = "./save_model"
    batch_size = 512
    epoch = 100
    lr = 1e-3

    log_interval = 10  # iter
    checkpoint_interval = 20  # epoch

    utils.mkdir_f(save_val_path)
    utils.mkdir_nf(dataset_path)
    utils.mkdir_nf(save_model_path)

    # dataset

    train_loader, test_loader, input_size = dataset.get(dataset_type, dataset_path, batch_size, num_workers=1)

    # model

    vae = VAE(input_size, encoder_dim, decoder_dim, latent_dim)
    if torch.cuda.is_available:
        vae = vae.cuda()

    # optimizer
    optimizer = optim.Adam(vae.parameters(), lr)
    utils.divide_line("{}--latent-{}--batch_size-{}".format(dataset_type, latent_dim, batch_size), is_log=True)
    for now_epoch in tqdm(range(1, epoch + 1), desc="Epoch", unit="epoch"):
        # train
        utils.divide_line("train")
        vae.train()
        train_rec_loss = []
        train_kl_loss = []
        train_total_loss = []
        train_mean_loss = []
        for train_iter, (train_img, train_label) in enumerate(
                tqdm(train_loader, desc="Train", unit="batch")):
            train_img = train_img.cuda()

            decoder_images, mean, log_variance = vae(train_img)
            # reconstruct_loss
            train_rec_loss.append(vae.reconstruct_loss(decoder_images, train_img.reshape(-1, input_size)))
            # kl_loss
            train_kl_loss.append(vae.kl_loss(mean, log_variance))
            train_total_loss.append(train_rec_loss[-1] + train_kl_loss[-1])
            train_mean_loss.append(train_total_loss[-1] / batch_size)

            optimizer.zero_grad()
            train_mean_loss[-1].backward()
            optimizer.step()

            if train_iter % log_interval == 0:
                utils.log("train", {"epoch": now_epoch, "iter": train_iter,
                                    "kl_loss": format(train_kl_loss[-1], ".4f"),
                                    "rec_loss": format(train_rec_loss[-1], ".4f"),
                                    "totoal_loss": format(train_total_loss[-1], ".4f"),
                                    "mean_loss": format(train_mean_loss[-1], ".4f")}, is_dynamic=True)
        utils.log("train", {"epoch": now_epoch,
                            "avg_kl_loss": format(torch.stack(train_kl_loss).mean(), ".4f"),
                            "avg_rec_loss": format(torch.stack(train_rec_loss).mean(), ".4f"),
                            "avg_totoal_loss": format(torch.stack(train_total_loss).mean(), ".4f"),
                            "avg_mean_loss": format(torch.stack(train_mean_loss).mean(), ".4f")}, is_dynamic=False)
        # validation
        utils.divide_line("val")
        test_rec_loss = []
        test_kl_loss = []
        test_total_loss = []
        test_mean_loss = []
        vae.eval()
        with torch.no_grad():
            for test_iter, (test_img, test_label) in enumerate(
                    tqdm(test_loader, desc="Validate", unit="batch")):
                test_img = test_img.cuda()
                decoder_images, mean, log_variance = vae(test_img)

                # reconstruct_loss
                test_rec_loss.append(vae.reconstruct_loss(decoder_images, test_img.reshape(-1, input_size)))
                # kl_loss
                test_kl_loss.append(vae.kl_loss(mean, log_variance))
                test_total_loss.append(test_rec_loss[-1] + test_kl_loss[-1])
                test_mean_loss.append(test_total_loss[-1] / batch_size)

                test_img = test_img.reshape(-1, 1, 28, 28)
                decoder_images = decoder_images.reshape(-1, 1, 28, 28)
                sava_data = torch.cat((test_img, decoder_images), dim=3)

                if test_iter % log_interval == 0:
                    utils.log("val", {"epoch": now_epoch, "iter": test_iter,
                                      "kl_loss": format(test_kl_loss[-1], ".4f"),
                                      "rec_loss": format(test_rec_loss[-1], ".4f"),
                                      "totoal_loss": format(test_total_loss[-1], ".4f"),
                                      "mean_loss": format(test_mean_loss[-1], ".4f")}, is_dynamic=True)
                if test_iter == 0:
                    torchvision.utils.save_image(sava_data, osp.join(save_val_path,
                                                                     "epoch_{}_test_iter_{}_val.jpg".format(now_epoch,
                                                                                                            test_iter)))
            utils.log("val", {"epoch": now_epoch,
                              "avg_kl_loss": format(torch.stack(test_kl_loss).mean(), ".4f"),
                              "avg_rec_loss": format(torch.stack(test_rec_loss).mean(), ".4f"),
                              "avg_totoal_loss": format(torch.stack(test_total_loss).mean(), ".4f"),
                              "avg_mean_loss": format(torch.stack(test_mean_loss).mean(), ".4f")}, is_dynamic=False)
        # checkpoint
        if now_epoch % checkpoint_interval == 0:
            utils.divide_line("save")
            now_save_model_path = osp.join(save_model_path, "latent_dim-{}".format(latent_dim))
            utils.mkdir_nf(now_save_model_path)
            save_model_name = "VAE_epoch_{}_{}.pth".format(now_epoch, utils.get_now_time())
            torch.save(vae, osp.join(now_save_model_path, save_model_name))
            utils.divide_line("save {} success !".format(save_model_name))
