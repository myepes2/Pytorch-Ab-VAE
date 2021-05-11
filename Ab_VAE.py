# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torch.utils.data import sampler
import matplotlib.pyplot as plt

import os
from os import path

from torch.optim import Adam


# %%
print(torch.__version__)
print(torch.cuda.is_available())


# %%
# New Model Hyperparameters

dataset_path = './datasets'

#cuda = False
#DEVICE = torch.device("cuda" if cuda else "cpu")
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

batch_size = 128
img_shape = (3, 32, 32) # (channels, width, height)


#color channels
#input_dim = 3
#latent_dim = 128


#output_dim = 3

lr = 2e-4

#DH3: 
#lr = 1e-2


# %%


# %% [markdown]
# ###    Step 1. Load (or download) Dataset

# %%
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


mnist_transform = transforms.Compose([
        transforms.ToTensor(),
])

kwargs = {'num_workers': 1, 'pin_memory': True} 

train_dataset = CIFAR10(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset  = CIFAR10(dataset_path, transform=mnist_transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False,  **kwargs)

# %% [markdown]
# ###    Step 1. Define Frankencoder

# %%
"""
    Frankencoder
"""

class VAE(nn.Module):
    
    def __init__(self, input_shape, latent_dim = 128, kernel_size = 3, stride = 1):
    
        super(VAE, self).__init__()
        
        c, h, w = input_shape
        self.input_shape = input_shape
        
        max_shape = (64, h, w)
        
        max_dim = 64*h*w
        
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 8, kernel_size, stride = stride, padding = kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size, stride = stride, padding = kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size, stride = stride, padding = kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size, stride = stride, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        #self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean  = nn.Linear(max_dim, latent_dim)
        self.FC_var   = nn.Linear(max_dim, latent_dim)
        self.FC_hidden = nn.Linear(latent_dim, max_dim)
        self.training = True
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, max_shape),
            nn.ConvTranspose2d(64, 32, kernel_size, stride = stride, padding=kernel_size // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size, stride = stride, padding=kernel_size // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size, stride = stride, padding=kernel_size // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, c, kernel_size, stride = stride, padding=kernel_size // 2),
            nn.Sigmoid(),
        )
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if std.is_cuda:
            esp = esp.cuda()
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.FC_mean(h), self.FC_var(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar    
    
    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar
    
    def decode(self, z):
        z = self.FC_hidden(z)
        z = self.decoder(z)
        return z        
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

# %% [markdown]
# ###    Step 3. Define Loss function and optimizer

# %%
#F.kl_div()


# %%
#BCE_loss = nn.BCELoss()
#mse_loss = nn.MSELoss()
def loss_function(x, x_hat, mean, log_var):
    #recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    recon_loss = F.mse_loss(x_hat, x, reduction = 'sum')
    KLD  = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return recon_loss, KLD

#optimizer = Adam(model.parameters(), lr=lr)


# %%
def test (model, test_loader, DEVICE, loss_function, batch_size, show = True):
    with torch.no_grad():
        
        model.eval()
        
        #[recon, kld]
        #test_loss = torch.zeros(1,2)
        test_rc_loss = 0
        test_kl_loss = 0

        for batch_idx, (x, _) in enumerate(tqdm(test_loader)):

            x = x.to(DEVICE)
            x_hat, mean, log_var = model(x)
            #loss = loss_function(x, x_hat, mean, log_var)
            
            recon_loss, KLD = loss_function(x, x_hat, mean, log_var)
            
            test_rc_loss += recon_loss.item()
            test_kl_loss += KLD.item()
            
            #batch_loss = torch.Tensor([recon_loss,KLD])
            
            #test_loss += batch_loss

            #print("loss: ", loss.item(),"recon_loss: ", recon_loss.item(), "  KLD: ", KLD.item())
            #if batch_idx==0:
            #   break
    
    if show:
        draw_sample_image(x[:batch_size//2], "Ground-truth images")
        draw_sample_image(x_hat[:batch_size//2], "Reconstructed images")
    
    return test_rc_loss / len(test_loader), test_kl_loss / len(test_loader)

# %% [markdown]
# ###    Step 3. Train VAE

# %%
save_path = './'

train_session = "tst-vae-dim"

lr = 2e-4

overwrite = False

#lr *= 2

print(lr)

model_path = save_path + train_session + "_" + "model.p"
fig_path = save_path + train_session + "_" + "loss.png"

print(model_path)
print(fig_path)


# %%
lr *= 2
print(lr)


# %%
model = VAE(img_shape, latent_dim = 256).to(DEVICE)

optimizer = Adam(model.parameters(), lr=lr) 

if path.isfile(model_path) and not overwrite:
    #model.load_state_dict(torch.load(model_path))
    print("Loading existing model: {}".format(model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']
          
else:
    print("Making new model: {}".format(model_path))
    #optimizer = Adam(model.parameters(), lr=lr)   
    
#properties = {} 
    
epochs = 1

#print_step = 50

print("Start training VAE...")
model.train()

info_list = []

for epoch in range(epochs):
    #overall_loss = 0
    train_rc_loss = 0
    train_kl_loss = 0
    for batch_idx, (x, _) in enumerate(tqdm(train_loader)):
        x = x.to(DEVICE)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        
        #recon_loss = mse_loss(x_hat, x)
        #KLD  = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        
        #loss = recon_loss + KLD
        
        recon_loss, KLD = loss_function(x, x_hat, mean, log_var)
        
        loss = recon_loss + KLD
        
        #overall_loss += loss.item()
        train_rc_loss  += recon_loss.item()
        train_kl_loss += KLD.item()
                
        loss.backward()
        optimizer.step()
        
    
    avg_rec = train_rc_loss / len(train_loader)
    avg_KLD = train_kl_loss / len(train_loader)
    average_loss = avg_rec + avg_KLD
    
    tst_rc, tst_kl = test(model, test_loader, DEVICE, loss_function, batch_size, show = False)
    tst_loss = tst_rc + tst_kl
    
    info_list.append([epoch + 1, epochs, avg_rec, avg_KLD, average_loss, tst_rc, tst_kl, tst_loss]) 
    
    print("""
    Epoch: [{}/{}]  
    Training:
        recon_loss: {:.2f}  KLD: {:.2f} total_loss: {:.2f}
    Test: 
        recon_loss: {:.2f}  KLD: {:.2f} total_loss: {:.2f}
    """.format(*info_list[-1]))
    
    
    #print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", average_loss)
    
print("Finish!!")

epoch_idx, _, avg_rec, avg_KLD, average_loss, tst_rc, tst_kl, tst_loss = zip(*info_list)

fig = plt.figure(0)
plt.plot(epoch_idx, average_loss, 'b.', label = "Training")
plt.plot(epoch_idx, tst_loss, 'g.', label = "Testing")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

fig = plt.figure(1)
plt.plot(epoch_idx, avg_rec, 'y.', label = "Reconstruction")
plt.plot(epoch_idx, avg_KLD, 'm.', label = "KLD")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Training Loss')

print("lr: ",lr)
#print("dLoss / num epochs: ", (average_loss[-1] - average_loss[0])/(epoch_idx[-1] - epoch_idx[0]))

torch.save({
            #'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'loss': average_loss[-1],
            }, model_path)

#fig = plt.figure()
#plt.plot(range(epoch+1), loss_list, 'b.')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#fig.patch.set_facecolor('white')
#fig.savefig(fig_path, facecolor=fig.get_facecolor(), edgecolor='none')

#properties.update({'model_state_dict': model.state_dict()})
#properties.update({
#    'optimizer_state_dict': optimizer.state_dict(),
#    'train_loss': train_loss_dict,
#    'val_loss': val_loss_dict,
#    'epoch': epoch
#})

#torch.save(model.state_dict(), model_path)


# %%
print("dLoss / num epochs: ", (average_loss[-1] - average_loss[1])/(epoch_idx[-1] - epoch_idx[1]))


# %%
test(model, train_loader, DEVICE, loss_function, batch_size)


# %%
test(model, test_loader, DEVICE, loss_function, batch_size)


# %%
fig = plt.figure(0)
plt.plot(epoch_idx, average_loss, 'b.', label = "Training")
plt.plot(epoch_idx, tst_loss, 'g.', label = "Testing")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

fig = plt.figure(1)
plt.plot(epoch_idx, avg_rec, 'y.', label = "Reconstruction")
plt.plot(epoch_idx, avg_KLD, 'm.', label = "KLD")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Training Loss')


# %%
def draw_sample_image(x, postfix):
  
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))


# %%



# %%
model.eval()

with torch.no_grad():

    for batch_idx, (x, _) in enumerate(tqdm(train_loader)):

        x = x.to(DEVICE)
        x_hat, mean, log_var = model(x)
        #loss = loss_function(x, x_hat, mean, log_var)
        loss, recon_loss, KLD = loss_function(x, x_hat, mean, log_var)
 
        print("loss: ", loss.item(),"recon_loss: ", recon_loss.item(), "  KLD: ", KLD.item())
        if batch_idx==0:
            break
            
draw_sample_image(x[:batch_size//2], "Ground-truth images")
draw_sample_image(x_hat[:batch_size//2], "Reconstructed images")


# %%
model.eval()

with torch.no_grad():

    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):

        x = x.to(DEVICE)
        x_hat, mean, log_var = model(x)
        #loss = loss_function(x, x_hat, mean, log_var)
        loss, recon_loss, KLD = loss_function(x, x_hat, mean, log_var)
 
        print("loss: ", loss.item(),"recon_loss: ", recon_loss.item(), "  KLD: ", KLD.item())
        if batch_idx==0:
            break
            
draw_sample_image(x[:batch_size//2], "Ground-truth images")
draw_sample_image(x_hat[:batch_size//2], "Reconstructed images")


# %%
draw_sample_image(x[:batch_size//2], "Ground-truth images")


# %%
draw_sample_image(x_hat[:batch_size//2], "Reconstructed images")

# %% [markdown]
# input_dim = 3
# 
# img_shape = (32, 32)
# 
# h_dim = 65536 
# 
# latent_dim = 128
# 
# kernel_size = 3
# 
# stride = 1
# 
# inp = torch.randn(1,input_dim,32,32)
# 
# m = nn.Sequential(
#     nn.Conv2d(input_dim, 8, kernel_size, stride = stride, padding=kernel_size // 2),
#     nn.ReLU(),
#     nn.Conv2d(8, 16, kernel_size, stride = stride, padding=kernel_size // 2),
#     nn.ReLU(),
#     nn.Conv2d(16, 32, kernel_size, stride = stride, padding=kernel_size // 2),
#     nn.ReLU(),
#     nn.Conv2d(32, 64, kernel_size, stride = stride, padding=kernel_size // 2),
#     nn.ReLU(),
#     nn.Flatten(),
#     nn.Linear(h_dim, latent_dim),
#     nn.Linear(latent_dim, h_dim),
#     nn.Unflatten(1, (64,32,32)),
#     nn.ConvTranspose2d(64, 32, kernel_size, stride = stride, padding=kernel_size // 2),
#     nn.ConvTranspose2d(32, 16, kernel_size, stride = stride, padding=kernel_size // 2),
#     nn.ConvTranspose2d(16, 8, kernel_size, stride = stride, padding=kernel_size // 2),
#     nn.ConvTranspose2d(8, input_dim, kernel_size, stride = stride, padding=kernel_size // 2),
#     nn.Sigmoid(),
#         )
# 
# out = m(inp)
# out.size()

# %%
criterion = nn.CrossEntropyLoss(ignore_index=-999)


# %%
def train_epoch(model, train_loader, optimizer, device, criterion, loss_size):
    """Trains a model for one epoch"""
    model.train()
    running_losses = torch.zeros(loss_size)
    for inputs, labels in tqdm(train_loader, total=len(train_loader)):
        inputs = inputs.to(device)
        labels = [label.to(device) for label in labels]

        optimizer.zero_grad()

        def handle_batch():
            """Function done to ensure variables immediately get dealloced"""
            outputs = model(inputs)
            losses = [
                criterion(output, label)
                for output, label in zip(outputs, labels)
            ]
            total_loss = sum(losses)
            losses.append(total_loss)

            total_loss.backward()
            optimizer.step()
            return outputs, torch.Tensor([float(l.item()) for l in losses])

        outputs, batch_loss = handle_batch()
        running_losses += batch_loss

    return running_losses


# %%
def train(model,
          train_loader,
          validation_loader,
          optimizer,
          epochs,
          current_epoch,
          device,
          criterion,
          lr_modifier,
          writer,
          save_file,
          save_every,
          properties=None):
    """"""
    #properties = {} if properties is None else properties
    #print('Using {} as device'.format(str(device).upper()))
    model = model.to(device)
    #loss_size = len(_output_names) + 1

    for epoch in range(current_epoch, epochs):
        train_losses = train_epoch(model, train_loader, optimizer, device,
                                   criterion, loss_size)
        avg_train_losses = train_losses / len(train_loader)
        train_loss_dict = dict(
            zip(_output_names + ['total'], avg_train_losses.tolist()))
        writer.add_scalars('train_loss', train_loss_dict, global_step=epoch)
        print('\nAverage training loss (epoch {}): {}'.format(
            epoch, train_loss_dict))

        val_losses = validate(model, validation_loader, device, criterion,
                              loss_size)
        avg_val_losses = val_losses / len(validation_loader)
        val_loss_dict = dict(
            zip(_output_names + ['total'], avg_val_losses.tolist()))
        writer.add_scalars('validation_loss', val_loss_dict, global_step=epoch)
        print('\nAverage validation loss (epoch {}): {}'.format(
            epoch, avg_val_losses.tolist()))

        total_val_loss = val_losses[-1]
        lr_modifier.step(total_val_loss)

        if (epoch + 1) % save_every == 0:
            properties.update({'model_state_dict': model.state_dict()})
            properties.update({
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_dict,
                'val_loss': val_loss_dict,
                'epoch': epoch
            })
            torch.save(properties, save_file + ".e{}".format(epoch + 1))

    properties.update({'model_state_dict': model.state_dict()})
    properties.update({
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss_dict,
        'val_loss': val_loss_dict,
        'epoch': epoch
    })
    torch.save(properties, save_file)


# %%
class ResNet2D(nn.Module):
    def __init__(self,
                 in_channels,
                 block,
                 num_blocks,
                 init_planes=64,
                 kernel_size=3,
                 dilation_cycle=5,
                 norm=nn.BatchNorm2d):
        """
        :param in_channels: The number of channels coming from the input.
        :type in_channels: int
        :param block: The type of residual block to use.
        :type block: torch.nn.Module
        :param num_blocks:
            A list of the number of blocks per layer. Each layer increases the
            number of channels by a factor of 2.
        :type num_blocks: List[int]
        :param init_planes: The number of planes the first 1D CNN should output.
                            Must be a power of 2.
        :type init_planes: int
        :param kernel_size: Size of the convolving kernel used in the Conv2D
                            convolution.
        :type kernel_size: int
        """
        super(ResNet2D, self).__init__()
        # Check if the number of initial planes is a power of 2, done for faster computation on GPU
        if not (init_planes != 0 and ((init_planes & (init_planes - 1)) == 0)):
            raise ValueError(
                'The initial number of planes must be a power of 2')

        self.activation = F.relu
        self.norm = norm
        self.kernel_size = kernel_size
        self.init_planes = init_planes
        self.in_planes = self.init_planes  # Number of input planes to the final layer
        self.num_layers = len(num_blocks)

        self.conv1 = nn.Conv2d(in_channels,
                               self.in_planes,
                               kernel_size=kernel_size,
                               stride=(1, 1),
                               padding=(kernel_size // 2, kernel_size // 2),
                               bias=False)
        self.bn1 = norm(self.in_planes)

        self.layers = []
        # Raise the number of planes by a power of two for each layer
        for i in range(0, self.num_layers):
            new_layer = self._make_layer(block,
                                         int(self.init_planes *
                                             math.pow(2, i)),
                                         num_blocks[i],
                                         stride=1,
                                         kernel_size=kernel_size,
                                         dilation_cycle=dilation_cycle)
            self.layers.append(new_layer)

            # Done to ensure layer information prints out when print() is called
            setattr(self, 'layer{}'.format(i), new_layer)

    def _make_layer(self, block, planes, num_blocks, stride, kernel_size,
                    dilation_cycle):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            dilation = int(math.pow(
                2, i % dilation_cycle)) if dilation_cycle > 0 else 1
            layers.append(
                block(self.in_planes,
                      planes,
                      stride=stride,
                      kernel_size=kernel_size,
                      dilation=(dilation, dilation),
                      norm=self.norm))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        return out


