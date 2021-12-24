import os.path as osp

import torch
import torch.optim as optim
import torchvision

from tqdm.notebook import tqdm

import dataset
from neko import neko_utils
from model import VAE

encoder_dim = 400
decoder_dim = 400
latent_dim = 20
dataset_path = "/dataset"
save_val_path = "./val_result"
save_model_path = "./save_model"
batch_size = 256
epoch = 100
lr = 1e-3

log_interval = 10  # iter
checkpoint_interval = 20  # epoch

neko_utils.mkdir_f(save_val_path)
neko_utils.mkdir_nf(dataset_path)
neko_utils.mkdir_nf(save_model_path)

# dataset

train_loader, test_loader, input_size = dataset.get()

# model

vae = VAE(input_size, encoder_dim, decoder_dim, latent_dim)
if torch.cuda.is_available:
    vae = vae.cuda()

# optimizer
optimizer = optim.Adam(vae.parameters(), lr)

for i in tqdm(range(1, epoch + 1), desc="Epoch", unit="epoch"):
    # train
    neko_utils.divid_line("train")
    vae.train()
    for train_iter, (train_img, train_label) in enumerate(tqdm(train_loader, desc="Train", unit="batch")):
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
            neko_utils.log("train", epoch, train_iter,
                           {"kl_loss": kl_loss, "rec_loss": rec_loss, "totoal_loss": total_loss,
                            "mean_loss": mean_loss})
            # print(
            #     "train -- epoch {} iter {} : kl_loss = {:.6f},rec_loss = {:.6f},total_loss = {:.6f},mean_loss = {:.6f}".format(
            #         i, train_iter, kl_loss, rec_loss, total_loss, mean_loss))
    # validation
    neko_utils.divid_line("val")
    vae.eval()
    with torch.no_grad():
        for test_iter, (test_img, test_label) in enumerate(tqdm(test_loader, desc="Validate", unit="batch")):
            test_img = test_img.cuda()
            decoder_images, mean, log_variance = vae(test_img)
            # reconstruct_loss
            rec_loss = vae.reconstruct_loss(decoder_images, test_img.reshape(-1, input_size))
            # kl_loss
            kl_loss = vae.kl_loss(mean, log_variance)
            total_loss = rec_loss + kl_loss
            mean_loss = total_loss / batch_size

            test_img = test_img.reshape(-1, 1, 28, 28)
            test_res = test_res.reshape(-1, 1, 28, 28)
            sava_data = torch.cat((test_img, test_res), dim=3)

            if test_iter % log_interval == 0:
                neko_utils.log("val", epoch, test_iter,
                               {"kl_loss": kl_loss, "rec_loss": rec_loss, "totoal_loss": total_loss,
                                "mean_loss": mean_loss})
                # print(
                #     "val -- epoch {} iter {} : kl_loss = {:.6f},rec_loss = {:.6f},total_loss = {:.6f},mean_loss = {:.6f}".format(
                #         i, test_iter, kl_loss, rec_loss, total_loss, mean_loss))
            if test_iter == 0:
                torchvision.utils.save_image(sava_data, osp.join(save_val_path,
                                                                 "epoch_{}_test_iter_{}_val.jpg".format(i,
                                                                                                        test_iter)))
    # checkpoint
    if i % checkpoint_interval == 0:
        neko_utils.divid_line("save")
        now_save_model_path = osp.join(save_model_path, "latent_dim-{}".format(latent_dim))
        neko_utils.mkdir_nf(now_save_model_path)
        save_model_name = "VAE_epoch_{}_{}.pth".format(i, neko_utils.get_now_time())
        torch.save(model, osp.join(now_save_model_path, save_model_name))
        neko_utils.divid_line("save {} success !".format(save_model_name))
