import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import tqdm

from utils.config import config_parse
from utils.dataset import load_mnist
from models.vae import Encoder, Decoder
from utils.noise import create_noise
from utils.draw import save_gif, draw_line
from utils.seed import setup_seed

def criterion(x:torch.Tensor, x_hat:torch.Tensor, mean:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
    # reconstruction loss
    # recons_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')     # standard implementation, but assume original distribution is Bernoulli or Multinomial
    recons_loss = F.mse_loss(x_hat, x, reduction='sum')                 # alternative implementation, but assume original distribution is Gaussian
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return recons_loss + kl_loss

def reparameterization(mean:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(std.device)
    z = mean + eps * std
    return z


if __name__ == "__main__":
    args = config_parse()
    print(args.expname)
    assert args.model_type == "VAE"
    if not os.path.exists(os.path.join(args.log_path, args.expname)):
        os.mkdir(os.path.join(args.log_path, args.expname))
    else:
        if not os.path.exists(os.path.join(args.log_path, args.expname, 'train')):
            os.mkdir(os.path.join(args.log_path, args.expname, 'train'))
        else:
            print("experiment has already been trained")
            exit(-1)
    with open(os.path.join(args.log_path, args.expname, 'train', 'args.txt'), 'w') as f:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            f.write('{} = {}\n'.format(arg, attr))
    log_file = open(os.path.join(args.log_path, args.expname, 'train', 'log.txt'), 'w')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(47)
    print(device)

    # initialize dataset
    print("start initializing dataset")
    if args.dataset_type == 'MNIST':
        num_channel = 1
        num_height = 28
        num_width = 28
        train_dataloader = load_mnist(args.data_path, args.batch_size, True)
    else:
        raise NotImplementedError
    print("finish initializing dataset")

    # initialize models
    print("start initializing models")
    encoder = Encoder(args.z_size, num_channel, num_height, num_width).to(device)
    decoder = Decoder(args.z_size, num_channel, num_height, num_width).to(device)
    optim_e = optim.Adam(encoder.parameters(), lr=args.enc_learning_rate)
    optim_d = optim.Adam(decoder.parameters(), lr=args.dec_learning_rate)
    scheduler_e = optim.lr_scheduler.ExponentialLR(optim_e, gamma=args.enc_gamma)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=args.dec_gamma)
    print("finish initializing models")

    # loop
    print("start training")
    loss_list = []          # training loss along epochs
    image_list = []         # generated images along epochs interval
    eval_noise = create_noise(args.z_size, args.eval_num).to(device)    # fixed noise for evaluation
    for epoch in tqdm.trange(args.epochs):
        # train
        encoder.train()
        decoder.train()
        loss = 0
        num_batch = 0
        for batch_idx, data in enumerate(train_dataloader):
            image, _ = data
            image = image.to(device)
            batch_size = image.shape[0]

            optim_e.zero_grad()
            optim_d.zero_grad()

            mean, logvar = encoder(image)
            z = reparameterization(mean, logvar)
            image_hat = decoder(z)

            batch_loss = criterion(image, image_hat, mean, logvar)
            loss += batch_loss.item() / batch_size
            batch_loss.backward()
            optim_e.step()
            optim_d.step()
            num_batch += 1
        epoch_loss = loss / num_batch
        loss_list.append(epoch_loss)

        # eval
        if epoch % args.eval_interval == 0:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                generated_img = decoder(eval_noise)
                image_list.append(generated_img.cpu().detach())
        else:
            pass

        print(f"epoch loss: {epoch_loss:.8f}, encoder learning rate: {scheduler_e.get_last_lr()[0]:.6f}, decoder learning rate: {scheduler_d.get_last_lr()[0]:.6f}")
        print(f"epoch loss: {epoch_loss:.8f}, encoder learning rate: {scheduler_e.get_last_lr()[0]:.6f}, decoder learning rate: {scheduler_d.get_last_lr()[0]:.6f}", file=log_file)
        scheduler_e.step()
        scheduler_d.step()
    print("finish training")

    # save results
    print("start saving results")
    torch.save(encoder.state_dict(), os.path.join(args.log_path, args.expname, 'train', 'encoder.pth'))
    torch.save(decoder.state_dict(), os.path.join(args.log_path, args.expname, 'train', 'decoder.pth'))
    save_gif(image_list, os.path.join(args.log_path, args.expname, 'train', 'train_eval_process.gif'), args.dataset_type, True)
    draw_line(loss_list, 1, title='train_loss', xlabel='epoch', ylabel='loss', 
                path=os.path.join(args.log_path, args.expname, 'train', 'train_loss.png'), ylimit=False)
    log_file.close()
    print("finish saving results")
