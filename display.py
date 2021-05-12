import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def draw_sample_image(x, prefix, postfix, output_path, write = True):

    fig = plt.figure(figsize=(8,8))
    fig.patch.set_facecolor('white')
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    img_grid = make_grid(x, padding=2, normalize=True)
    plt.imshow(np.transpose(img_grid, (1, 2, 0)))
    if write:
        writer=SummaryWriter(output_path)
        writer.add_image('{}_{}.png'.format(prefix, postfix), img_grid)
    else:
        fig.savefig(output_path + '{}_{}.png'.format(prefix, postfix), facecolor=fig.get_facecolor(),  edgecolor='none')

