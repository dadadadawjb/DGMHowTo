import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import tqdm

from utils.config import config_parse
from utils.dataset import load_mnist
from models.gan import Generator, Discriminator
from utils.noise import create_noise
from utils.draw import save_gif, draw_line, draw_lines
from utils.seed import setup_seed

def train_generator(discriminator:Discriminator, g_optimizer:optim.Optimizer, criterion:nn.Module, 
    data_fake:torch.Tensor) -> float:
    # hope discriminator cannot distinguish correctly
    label_real = torch.ones(data_fake.shape[0], 1).to(data_fake.device)

    g_optimizer.zero_grad()

    output_fake = discriminator(data_fake)
    loss = criterion(output_fake, label_real)

    loss.backward()
    g_optimizer.step()

    return loss.item()

def train_discriminator(discriminator:Discriminator, d_optimizer:optim.Optimizer, criterion:nn.Module, 
    data_real:torch.Tensor, data_fake:torch.Tensor) -> float:
    # hope discriminator can distinguish correctly
    label_real = torch.ones(data_real.shape[0], 1).to(data_real.device)
    label_fake = torch.zeros(data_fake.shape[0], 1).to(data_fake.device)

    d_optimizer.zero_grad()

    output_real = discriminator(data_real)
    loss_real = criterion(output_real, label_real)

    output_fake = discriminator(data_fake)
    loss_fake = criterion(output_fake, label_fake)

    loss = loss_real + loss_fake
    loss.backward()
    d_optimizer.step()

    return loss.item()


if __name__ == "__main__":
    args = config_parse()
    print(args.expname)
    assert args.model_type == "GAN"
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
        train_dataloader = load_mnist(args.data_path, args.batch_size)
    else:
        raise NotImplementedError
    print("finish initializing dataset")

    # initialize models
    print("start initializing models")
    generator = Generator(args.z_size, num_channel, num_height, num_width).to(device)
    discriminator = Discriminator(args.z_size, num_channel, num_height, num_width).to(device)
    optim_g = optim.Adam(generator.parameters(), lr=args.gen_learning_rate)
    optim_d = optim.Adam(discriminator.parameters(), lr=args.dis_learning_rate)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_g, gamma=args.gen_gamma)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=args.dis_gamma)
    criterion = nn.BCELoss()
    print("finish initializing models")

    # loop
    print("start training")
    g_loss_list = []        # generator training loss along epochs
    d_loss_list = []        # discriminator training loss along epochs
    image_list = []         # generator generated images along epochs interval
    acc_list = []           # discriminator accuracy along epochs interval
    eval_noise = create_noise(args.z_size, args.eval_num).to(device)    # fixed noise for evaluation
    for epoch in tqdm.trange(args.epochs):
        # train
        generator.train()
        discriminator.train()
        loss_g = 0.0
        loss_d = 0.0
        num_batch = 0
        for batch_idx, data in enumerate(train_dataloader):
            image, _ = data
            image = image.to(device)
            batch_size = image.shape[0]
            # train discriminator for k steps
            for step in range(args.k):
                noise = create_noise(args.z_size, batch_size).to(device)
                data_fake = generator(noise).detach()       # detach generator
                data_real = image
                loss_d += train_discriminator(discriminator, optim_d, criterion, data_real, data_fake)
            # train generator for 1 step
            noise = create_noise(args.z_size, batch_size).to(device)
            data_fake = generator(noise)
            loss_g += train_generator(discriminator, optim_g, criterion, data_fake)
            num_batch += 1
        epoch_loss_g = loss_g / num_batch
        epoch_loss_d = loss_d / num_batch
        g_loss_list.append(epoch_loss_g)
        d_loss_list.append(epoch_loss_d)

        # eval
        if epoch % args.eval_interval == 0:
            generator.eval()
            discriminator.eval()
            with torch.no_grad():
                generated_img = generator(eval_noise)
                image_list.append(generated_img.cpu().detach())

                pred_label = discriminator(generated_img)
                acc = (pred_label < 0.5).sum().item() / args.eval_num
                acc_list.append(acc)

                print(f"discriminator accuracy: {acc}")
                print(f"discriminator accuracy: {acc}", file=log_file)
        else:
            pass
        
        print(f"generator epoch loss: {epoch_loss_g:.8f}, generator learning rate: {scheduler_g.get_last_lr()[0]:.6f}, discriminator epoch loss: {epoch_loss_d:.8f}, discriminator learning rate: {scheduler_d.get_last_lr()[0]:.6f}")
        print(f"generator epoch loss: {epoch_loss_g:.8f}, generator learning rate: {scheduler_g.get_last_lr()[0]:.6f}, discriminator epoch loss: {epoch_loss_d:.8f}, discriminator learning rate: {scheduler_d.get_last_lr()[0]:.6f}", file=log_file)
        scheduler_g.step()
        scheduler_d.step()
    print("finish training")

    # save results
    print("start saving results")
    torch.save(generator.state_dict(), os.path.join(args.log_path, args.expname, 'train', 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(args.log_path, args.expname, 'train', 'discriminator.pth'))     # actually no need to save discriminator
    save_gif(image_list, os.path.join(args.log_path, args.expname, 'train', 'generator_eval_process.gif'), args.dataset_type)
    draw_line(acc_list, args.eval_interval, title='discriminator_accuracy', xlabel='epoch', ylabel='accuracy', 
                path=os.path.join(args.log_path, args.expname, 'train', 'discriminator_eval_process.png'), ylimit=False)
    draw_lines(g_loss_list, d_loss_list, 'generator_loss', 'discriminator_loss', 1, 
                title='train_loss', xlabel='epoch', ylabel='loss', 
                path=os.path.join(args.log_path, args.expname, 'train', 'train_process.png'), ylimit=False)
    log_file.close()
    print("finish saving results")
