import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

class F_ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=5) -> None:
        super().__init__()

        self.kernel_size = kernel_size

        self.activation = F.relu

        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.bn2 = torch.nn.BatchNorm2d(in_channels * 2)

        self.padding = (self.kernel_size - 1) // 2

        self.conv1 = torch.nn.Conv2d(
            in_channels,
            in_channels * 2,
            kernel_size=self.kernel_size,
            stride=2,
            padding=self.padding,
            bias=False,
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels * 2,
            in_channels * 2,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=False,
        )

        self.conv_skip = torch.nn.Conv2d(
            in_channels,
            in_channels * 2,
            kernel_size=self.kernel_size,
            stride=2,
            padding=self.padding
        )

    def forward(self, x):
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(out)))
        out += self.conv_skip(x)
        return out

class R_ResBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size=5) -> None:
        super().__init__()

        self.kernel_size = kernel_size

        self.activation = F.relu

        self.bn1 = torch.nn.BatchNorm2d(in_channels)

        self.bn2 = torch.nn.BatchNorm2d(in_channels // 2)

        self.padding = (kernel_size - 1) // 2
        
        #1 if odd, 0 if even
        self.output_padding = (lambda x : (x % 2))(kernel_size)

        self.deconv1 = torch.nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=self.kernel_size,
            stride=2,
            padding=self.padding,
            output_padding = self.output_padding
        )


        self.deconv2 = torch.nn.ConvTranspose2d(
            in_channels // 2,
            in_channels // 2,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
        )

        self.deconv_skip = torch.nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=self.kernel_size,
            stride=2,
            padding=self.padding,
            output_padding = self.output_padding
        )

    def forward(self, x):
        out = self.deconv1(self.activation(self.bn1(x)))
        out = self.deconv2(self.activation(self.bn2(out)))
        out += self.deconv_skip(x)
        return out

class Ab_Encoder(nn.Module):
    def __init__(
        self,
        img_channels,
        res_channels,
        #layers: list = [5]
        num_blocks=2,
    ):

        self.img_channels = img_channels
        self.res_channels = res_channels
        self.num_blocks = num_blocks

        self.bn1 = nn.BatchNorm2d(res_channels)
        self.activation = F.relu

        self.conv1 = nn.Conv2d(self.img_channels, self.res_channels, kernel_size=3, stride=1, padding=1, bias=False)

        #will add multi-layer functionality as it becomes necessary
        self.layer = self._make_layer(self.res_channels, self.num_blocks)

    def _make_layer(self, res_channels = 4, num_blocks=2, kernel_size=5):
        """Makes a layer made up of blocks"""
        #strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                F_ResBlock(
                      res_channels,
                      kernel_size=kernel_size
                      )
            )
            res_channels*=2
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer(x)
        #for layer in self.layers:
        #    out = layer(out)
        out = torch.flatten(out, 1)
        return out

class Ab_Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_shape,
        num_blocks=2,
    ):

    #self.img_channels = img_channels
        self.c, self.h, self.w = hidden_shape
        self.hidden_dim = self.c*self.h*self.w
        self.res_channels = self.c
        self.num_blocks = num_blocks

        self.bn1 = nn.BatchNorm2d(res_channels)
        self.activation = F.relu

        self.linear = nn.Linear(latent_dim, self.hidden_dim)
        self.reshaper = nn.Unflatten(1, hidden_shape)

        #self.conv1 = nn.Conv2d(self.img_channels, self.res_channels, kernel_size=3, stride=1, padding=1, bias=False)
        #will add multi-layer functionality as it becomes necessary
        self.layer = self._make_layer(self.res_channels, self.num_blocks)

    def _make_layer(self, res_channels = 4, num_blocks=2, kernel_size=5):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                R_ResBlock(
                      res_channels,
                      kernel_size=kernel_size
                      )
            )
            res_channels /= 2
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        out = self.linear(x)
        out = self.reshaper(x)
        out = self.layer(x)
        #for layer in self.layers:
         #   out = layer(out)
        out = torch.flatten(out, 1)
        return out

class Ab_VAE(pl.LightningModule):
    def __init__(
        self,
        input_shape : tuple,
        latent_dim: int = 128,
        num_blocks: int = 2,
        lr: float = 1e-4
        #kl_coeff: float = 0.1,
    ):

        super(Ab_VAE, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.c, self.h, self.w = input_shape

        ##Finds next highest power of two for first conv. block
        self.res_channels = (lambda x : (1 if x == 0 else 2**(x - 1).bit_length()))(self.c)

        self.hidden_c = self.res_channels*2**(num_blocks)
        self.hidden_h = self.h // 2**(num_blocks)
        self.hidden_w = self.w // 2**(num_blocks)

        self.hidden_shape = (self.hidden_c, self.hidden_h, self.hidden_w)
        self.hidden_dim = self.hidden_c*self.hidden_h*self.hidden_w
        
        self.encoder = Ab_Encoder(self.c, self.res_channels, self.num_blocks)
        self.decoder = Ab_Decoder(self.latent_dim, self.hidden_shape, self.num_blocks)

        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim, self.latent_dim)  

        def forward(self, x):
            x = self.encoder(x)
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
            p, q, z = self.sample(mu, log_var)
            return self.decoder(z)          

        def sample(self, mu, log_var):
            std = torch.exp(log_var / 2)
            p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()
            return p, q, z     

        def _run_step(self, x):
            x = self.encoder(x)
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
            p, q, z = self.sample(mu, log_var)
            return z, self.decoder(z), p, q

        def step(self, batch, batch_idx):
            x, y = batch
            z, x_hat, p, q = self._run_step(x)

            recon_loss = F.mse_loss(x_hat, x, reduction='mean')

            log_qz = q.log_prob(z)
            log_pz = p.log_prob(z)

            kl = log_qz - log_pz
            kl = kl.mean()
            #kl *= self.kl_coeff

            loss = kl + recon_loss

            logs = {
                "recon_loss": recon_loss,
                "kl": kl,
                "loss": loss,
            }
            return loss, logs

        def training_step(self, batch, batch_idx):
            loss, logs = self.step(batch, batch_idx)
            self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
            return loss

        def validation_step(self, batch, batch_idx):
            loss, logs = self.step(batch, batch_idx)
            self.log_dict({f"val_{k}": v for k, v in logs.items()})
            return loss      

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)  

        





