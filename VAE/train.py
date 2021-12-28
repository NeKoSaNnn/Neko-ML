import os.path as osp

import torch
import torch.optim as optim
import torchvision
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

    train_loader, test_loader, input_size = dataset.get(dataset_type, dataset_path, batch_size)

    # model

    vae = VAE(input_size, encoder_dim, decoder_dim, latent_dim)
    if torch.cuda.is_available:
        vae = vae.cuda()

    # optimizer
    optimizer = optim.Adam(vae.parameters(), lr)

    for now_epoch in tqdm(range(1, epoch + 1), desc="Epoch", unit="epoch", mininterval=1):
        # train
        utils.divide_line("train")
        vae.train()
        for train_iter, (train_img, train_label) in enumerate(
                tqdm(train_loader, desc="Train", unit="batch", mininterval=1)):
            train_img = train_img.cuda()

            decoder_images, mean, log_variance = vae(train_img)
            # reconstruct_loss
            rec_loss = vae.reconstruct_loss(decoder_images, train_img.reshape(-1, input_size))
            # kl_loss
            kl_loss = vae.kl_loss(mean, log_variance)
            total_loss = rec_loss + kl_loss
            mean_loss = total_loss / batch_size

            optimizer.zero_grad()
            mean_loss.backward()
            optimizer.step()

            if train_iter % log_interval == 0:
                utils.log("train", now_epoch, train_iter,
                          {"kl_loss": kl_loss, "rec_loss": rec_loss, "totoal_loss": total_loss,
                           "mean_loss": mean_loss})
                # print(
                #     "train -- epoch {} iter {} : kl_loss = {:.6f},rec_loss = {:.6f},total_loss = {:.6f},mean_loss = {:.6f}".format(
                #         i, train_iter, kl_loss, rec_loss, total_loss, mean_loss))
        # validation
        utils.divide_line("val")
        vae.eval()
        with torch.no_grad():
            for test_iter, (test_img, test_label) in enumerate(
                    tqdm(test_loader, desc="Validate", unit="batch", mininterval=1)):
                test_img = test_img.cuda()
                decoder_images, mean, log_variance = vae(test_img)
                # reconstruct_loss
                rec_loss = vae.reconstruct_loss(decoder_images, test_img.reshape(-1, input_size))
                # kl_loss
                kl_loss = vae.kl_loss(mean, log_variance)
                total_loss = rec_loss + kl_loss
                mean_loss = total_loss / batch_size

                test_img = test_img.reshape(-1, 1, 28, 28)
                decoder_images = decoder_images.reshape(-1, 1, 28, 28)
                sava_data = torch.cat((test_img, decoder_images), dim=3)

                if test_iter % log_interval == 0:
                    utils.log("val", now_epoch, test_iter,
                              {"kl_loss": kl_loss, "rec_loss": rec_loss, "totoal_loss": total_loss,
                               "mean_loss": mean_loss})
                    # print(
                    #     "val -- epoch {} iter {} : kl_loss = {:.6f},rec_loss = {:.6f},total_loss = {:.6f},mean_loss = {:.6f}".format(
                    #         i, test_iter, kl_loss, rec_loss, total_loss, mean_loss))
                if test_iter == 0:
                    torchvision.utils.save_image(sava_data, osp.join(save_val_path,
                                                                     "epoch_{}_test_iter_{}_val.jpg".format(now_epoch,
                                                                                                            test_iter)))
        # checkpoint
        if now_epoch % checkpoint_interval == 0:
            utils.divide_line("save")
            now_save_model_path = osp.join(save_model_path, "latent_dim-{}".format(latent_dim))
            utils.mkdir_nf(now_save_model_path)
            save_model_name = "VAE_epoch_{}_{}.pth".format(now_epoch, utils.get_now_time())
            torch.save(vae, osp.join(now_save_model_path, save_model_name))
            utils.divide_line("save {} success !".format(save_model_name))
