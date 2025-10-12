import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels=3, base_channels=64, num_layers=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.num_layers = num_layers
        
        # Initial projection from latent to spatial
        self.initial_size = 4  # Start with 4x4 spatial dimension
        self.fc = nn.Linear(latent_dim, base_channels * (2 ** (num_layers - 1)) * self.initial_size * self.initial_size)
        
        layers = []
        in_channels = base_channels * (2 ** (num_layers - 1))
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** max(num_layers - i - 2, 0))
            
            if i == num_layers - 1:
                # Last layer outputs the desired channels
                layers.extend([
                    nn.ConvTranspose2d(in_channels, output_channels, 4, 2, 1),
                    nn.Tanh()  # Output in [-1, 1]
                ])
            else:
                layers.extend([
                    nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                ])
            
            in_channels = out_channels
            
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z):
        # Project and reshape
        x = self.fc(z)
        x = x.view(x.size(0), -1, self.initial_size, self.initial_size)
        
        # Decode
        x = self.decoder(x)
        
        return x


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder=None, latent_dim=128):
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 64, 64)  # Assuming image input
            encoder_out = self.encoder(dummy_input)
            if isinstance(encoder_out, dict):
                encoder_out = encoder_out['pooled'] if 'pooled' in encoder_out else list(encoder_out.values())[0]
            encoder_dim = encoder_out.shape[-1]
        
        # Projection to latent space
        self.encoder_projection = nn.Linear(encoder_dim, latent_dim)
        
        # Decoder
        if decoder is None:
            self.decoder = ConvDecoder(latent_dim, output_channels=3, base_channels=64, num_layers=4)
        else:
            self.decoder = decoder
    
    def encode(self, x):
        h = self.encoder(x)
        if isinstance(h, dict):
            h = h['pooled'] if 'pooled' in h else list(h.values())[0]
        z = self.encoder_projection(h)
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        
        return {
            'reconstruction': x_recon,
            'latent': z
        }


class VAE(nn.Module):
    def __init__(self, encoder, decoder=None, latent_dim=128):
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 64, 64)  # Assuming image input
            encoder_out = self.encoder(dummy_input)
            if isinstance(encoder_out, dict):
                encoder_out = encoder_out['pooled'] if 'pooled' in encoder_out else list(encoder_out.values())[0]
            encoder_dim = encoder_out.shape[-1]
        
        # Projection to latent parameters (mean and log variance)
        self.fc_mu = nn.Linear(encoder_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_dim, latent_dim)
        
        # Decoder
        if decoder is None:
            self.decoder = ConvDecoder(latent_dim, output_channels=3, base_channels=64, num_layers=4)
        else:
            self.decoder = decoder
    
    def encode(self, x):
        h = self.encoder(x)
        if isinstance(h, dict):
            h = h['pooled'] if 'pooled' in h else list(h.values())[0]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        
        return {
            'reconstruction': x_recon,
            'mu': mu,
            'logvar': logvar,
            'latent': z
        }
    
    def compute_loss(self, x, output, beta=1.0, reduction='mean'):
        """Compute VAE loss = Reconstruction loss + beta * KL divergence"""
        x_recon = output['reconstruction']
        mu = output['mu']
        logvar = output['logvar']
        
        # Reconstruction loss (can be MSE or BCE depending on the data)
        if x_recon.shape != x.shape:
            # Resize if needed
            x_recon = F.interpolate(x_recon, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        recon_loss = F.mse_loss(x_recon, x, reduction=reduction)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        if reduction == 'mean':
            kl_loss = kl_loss / x.shape[0]
        
        # Total loss
        loss = recon_loss + beta * kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def sample(self, num_samples, device):
        """Sample from the latent space"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


class ConditionalVAE(VAE):
    def __init__(self, encoder, decoder=None, latent_dim=128, condition_dim=10):
        super().__init__(encoder, decoder, latent_dim)
        self.condition_dim = condition_dim
        
        # Modify decoder to accept conditions
        if decoder is None:
            # Create conditional decoder
            self.decoder = ConditionalConvDecoder(
                latent_dim + condition_dim, 
                output_channels=3, 
                base_channels=64, 
                num_layers=4
            )
        
        # Update encoder projections to include condition
        encoder_dim = self.fc_mu.in_features
        self.fc_mu = nn.Linear(encoder_dim + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_dim + condition_dim, latent_dim)
    
    def encode(self, x, c):
        h = self.encoder(x)
        if isinstance(h, dict):
            h = h['pooled'] if 'pooled' in h else list(h.values())[0]
        
        # Concatenate condition
        h_c = torch.cat([h, c], dim=-1)
        
        mu = self.fc_mu(h_c)
        logvar = self.fc_logvar(h_c)
        return mu, logvar
    
    def decode(self, z, c):
        z_c = torch.cat([z, c], dim=-1)
        return self.decoder(z_c)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        
        return {
            'reconstruction': x_recon,
            'mu': mu,
            'logvar': logvar,
            'latent': z,
            'condition': c
        }
    
    def sample(self, num_samples, condition, device):
        """Conditional sampling"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        if condition.dim() == 1:
            condition = condition.unsqueeze(0).repeat(num_samples, 1)
        samples = self.decode(z, condition)
        return samples


class ConditionalConvDecoder(ConvDecoder):
    def __init__(self, latent_dim, output_channels=3, base_channels=64, num_layers=4):
        # latent_dim includes condition dimension
        super().__init__(latent_dim, output_channels, base_channels, num_layers)