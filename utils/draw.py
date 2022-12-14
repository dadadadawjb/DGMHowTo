import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import imageio
import matplotlib.pyplot as plt
import numpy as np

def save_gif(image_list:list, path:str, dataset_type:str, normalize:bool) -> None:
    to_pil_image = transforms.ToPILImage()
    if dataset_type == "MNIST":
        if normalize:
            frames = [np.array(to_pil_image(make_grid(torch.min(torch.max(image * 0.3081 + 0.1307, torch.zeros_like(image)), torch.ones_like(image))))) for image in image_list]
        else:
            frames = [np.array(to_pil_image(make_grid(torch.min(torch.max(image, torch.zeros_like(image)), torch.ones_like(image))))) for image in image_list]
    else:
        raise NotImplementedError
    imageio.mimsave(path, frames)

def draw_line(data:list, interval:int, title:str, xlabel:str, ylabel:str, path:str, ylimit:bool) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = list(range(len(data)))
    x = [i * interval for i in x]
    if ylimit:
        plt.ylim(ymax=5 * sum(data) / len(data))
    ax.plot(x, data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(path)


def draw_lines(data1:list, data2:list, name1:str, name2:str, interval:int, title:str, xlabel:str, ylabel:str, path:str, ylimit:bool) -> None:
    assert len(data1) == len(data2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = list(range(len(data1)))
    x = [i * interval for i in x]
    if ylimit:
        plt.ylim(ymax=min(5 * sum(data1) / len(data1), 5 * sum(data2) / len(data2)))
    ax.plot(x, data1, color='red', linestyle='-', label=name1)
    ax.plot(x, data2, color='blue', linestyle='-', label=name2)
    plt.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(path)
