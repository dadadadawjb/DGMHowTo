import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm

from utils.config import config_parse
from utils.dataset import load_mnist
from models.nade import NADE
from utils.draw import save_gif, draw_line
from utils.seed import setup_seed

def sequence_generate(model:NADE, init_image:torch.Tensor) -> torch.Tensor:
    generate_num, generate_channel, generate_height, generate_width = init_image.shape
    generated_image = init_image.detach()
    for i in range(generate_channel * generate_height * generate_width):
        i_output = model(generated_image)
        i_output = torch.bernoulli(i_output)    # distribution sample
        i_output = i_output.view(generate_num, generate_channel * generate_height * generate_width)
        generated_image = generated_image.view(generate_num, generate_channel * generate_height * generate_width)
        generated_image[:, i] = i_output[:, i]
        generated_image = generated_image.view(generate_num, generate_channel, generate_height, generate_width)
    return generated_image

if __name__ == "__main__":
    args = config_parse()
    print(args.expname)
    assert args.model_type == "NADE"
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
        train_dataloader = load_mnist(args.data_path, args.batch_size, False)
    else:
        raise NotImplementedError
    print("finish initializing dataset")

    # initialize models
    print("start initializing models")
    model = NADE(args.h_size, num_channel, num_height, num_width).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    criterion = nn.BCELoss()
    print("finish initializing models")

    # loop
    print("start training")
    loss_list = []          # training loss along epochs
    image_list = []         # generated images along epochs interval
    zero_eval = torch.zeros(args.eval_num, num_channel, num_height, num_width).to(device)
    for epoch in tqdm.trange(args.epochs):
        # train
        model.train()
        loss = 0.0
        num_batch = 0
        for batch_idx, data in enumerate(train_dataloader):
            image, _ = data
            image = image.to(device)
            batch_size = image.shape[0]

            optimizer.zero_grad()

            image_hat = model(image)

            batch_loss = criterion(image_hat, image)
            loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            num_batch += 1
        epoch_loss = loss / num_batch
        loss_list.append(epoch_loss)

        # eval
        if epoch % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                generated_img = sequence_generate(model, zero_eval)
                image_list.append(generated_img.cpu().detach())
        else:
            pass

        print(f"epoch loss: {epoch_loss:.8f}, learning rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"epoch loss: {epoch_loss:.8f}, learning rate: {scheduler.get_last_lr()[0]:.6f}", file=log_file)
        scheduler.step()
    print("finish training")

    # save results
    print("start saving results")
    torch.save(model.state_dict(), os.path.join(args.log_path, args.expname, 'train', 'nade.pth'))
    save_gif(image_list, os.path.join(args.log_path, args.expname, 'train', 'train_eval_process.gif'), args.dataset_type, False)
    draw_line(loss_list, 1, title='train_loss', xlabel='epoch', ylabel='loss', 
                path=os.path.join(args.log_path, args.expname, 'train', 'train_loss.png'), ylimit=False)
    log_file.close()
    print("finish saving results")
