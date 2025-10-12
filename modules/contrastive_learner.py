import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, similarity='cosine'):
        super().__init__()
        self.temperature = temperature
        self.similarity = similarity
        
    def forward(self, z1, z2, labels=None):
        """
        Compute contrastive loss
        z1, z2: Tensor of shape (batch_size, feature_dim)
        labels: Optional tensor of shape (batch_size,) for supervised contrastive
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # Normalize embeddings
        if self.similarity == 'cosine':
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Create positive mask
        positive_mask = torch.zeros((2 * batch_size, 2 * batch_size), device=device)
        positive_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = 1
        positive_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = 1
        
        # Remove diagonal
        mask = torch.eye(2 * batch_size, device=device)
        similarity_matrix = similarity_matrix.masked_fill(mask.bool(), -1e9)
        
        # Apply temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        
        if labels is not None:
            # Supervised contrastive
            labels = labels.contiguous().view(-1, 1)
            expanded_labels = torch.cat([labels, labels], dim=0)
            mask = torch.eq(expanded_labels, expanded_labels.T).float().to(device)
            mask = mask * (1 - torch.eye(2 * batch_size, device=device))
            
            # Compute log probability
            exp_sum = exp_sim.sum(dim=1, keepdim=True)
            log_prob = similarity_matrix - torch.log(exp_sum)
            
            # Compute mean of log-likelihoods over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            loss = -mean_log_prob_pos.mean()
        else:
            # Unsupervised contrastive (SimCLR style)
            numerator = exp_sim * positive_mask
            denominator = exp_sim.sum(dim=1, keepdim=True)
            
            loss_partial = -torch.log(numerator.sum(dim=1) / denominator.squeeze())
            loss = loss_partial.mean()
            
        return loss


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                
        self.projector = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.projector(x)


class ContrastiveLearner(nn.Module):
    def __init__(self, encoder_1, encoder_2=None, projection_dim=128, 
                 hidden_dim=2048, temperature=0.07, loss_type='simclr',
                 momentum=0.999, use_momentum_encoder=False):
        super().__init__()
        
        # Encoders
        self.encoder_1 = encoder_1
        self.encoder_2 = encoder_2 if encoder_2 is not None else encoder_1
        self.shared_encoder = encoder_2 is None
        
        # Get encoder output dimensions
        with torch.no_grad():
            dummy_input_1 = torch.randn(2, 3, 224, 224)  # Assuming image input
            encoder_out_1 = self.encoder_1(dummy_input_1)
            if isinstance(encoder_out_1, dict):
                encoder_out_1 = encoder_out_1['pooled'] if 'pooled' in encoder_out_1 else list(encoder_out_1.values())[0]
            
            encoder_dim_1 = encoder_out_1.shape[-1]
            
            if not self.shared_encoder:
                encoder_out_2 = self.encoder_2(dummy_input_1)
                if isinstance(encoder_out_2, dict):
                    encoder_out_2 = encoder_out_2['pooled'] if 'pooled' in encoder_out_2 else list(encoder_out_2.values())[0]
                encoder_dim_2 = encoder_out_2.shape[-1]
            else:
                encoder_dim_2 = encoder_dim_1
        
        # Projection heads
        self.projection_head_1 = ProjectionHead(encoder_dim_1, hidden_dim, projection_dim)
        self.projection_head_2 = ProjectionHead(encoder_dim_2, hidden_dim, projection_dim) if not self.shared_encoder else self.projection_head_1
        
        # Loss
        self.loss_type = loss_type
        self.temperature = temperature
        self.criterion = ContrastiveLoss(temperature=temperature)
        
        # Momentum encoder for MoCo
        self.use_momentum_encoder = use_momentum_encoder
        self.momentum = momentum
        
        if use_momentum_encoder:
            # Create momentum encoder
            self.momentum_encoder_1 = self._create_momentum_encoder(self.encoder_1)
            self.momentum_projection_1 = self._create_momentum_encoder(self.projection_head_1)
            
            if not self.shared_encoder:
                self.momentum_encoder_2 = self._create_momentum_encoder(self.encoder_2)
                self.momentum_projection_2 = self._create_momentum_encoder(self.projection_head_2)
            
            # Queue for negative samples
            self.register_buffer("queue_1", torch.randn(projection_dim, 65536))
            self.register_buffer("queue_2", torch.randn(projection_dim, 65536))
            self.queue_1 = F.normalize(self.queue_1, dim=0)
            self.queue_2 = F.normalize(self.queue_2, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            
    def _create_momentum_encoder(self, encoder):
        import copy
        momentum_encoder = copy.deepcopy(encoder)
        
        for param in momentum_encoder.parameters():
            param.requires_grad = False
            
        return momentum_encoder
    
    @torch.no_grad()
    def _momentum_update(self):
        """Update momentum encoders"""
        for param, param_m in zip(self.encoder_1.parameters(), self.momentum_encoder_1.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
            
        for param, param_m in zip(self.projection_head_1.parameters(), self.momentum_projection_1.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
            
        if not self.shared_encoder:
            for param, param_m in zip(self.encoder_2.parameters(), self.momentum_encoder_2.parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            for param, param_m in zip(self.projection_head_2.parameters(), self.momentum_projection_2.parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
    
    def forward(self, x1, x2=None, labels=None, return_embeddings=False):
        # Handle single input case (self-supervised)
        if x2 is None:
            x2 = x1
            
        # Encode
        h1 = self.encoder_1(x1)
        if isinstance(h1, dict):
            h1 = h1['pooled'] if 'pooled' in h1 else list(h1.values())[0]
            
        h2 = self.encoder_2(x2) if not self.shared_encoder else self.encoder_1(x2)
        if isinstance(h2, dict):
            h2 = h2['pooled'] if 'pooled' in h2 else list(h2.values())[0]
        
        # Project
        z1 = self.projection_head_1(h1)
        z2 = self.projection_head_2(h2) if not self.shared_encoder else self.projection_head_1(h2)
        
        # Compute loss
        if self.loss_type == 'simclr':
            loss = self.criterion(z1, z2, labels)
        elif self.loss_type == 'moco' and self.use_momentum_encoder:
            # MoCo style loss
            with torch.no_grad():
                self._momentum_update()
                
                # Compute momentum features
                h1_m = self.momentum_encoder_1(x1)
                if isinstance(h1_m, dict):
                    h1_m = h1_m['pooled'] if 'pooled' in h1_m else list(h1_m.values())[0]
                z1_m = self.momentum_projection_1(h1_m)
                z1_m = F.normalize(z1_m, dim=1)
                
            z1 = F.normalize(z1, dim=1)
            
            # Positive logits
            l_pos = torch.einsum('nc,nc->n', [z1, z1_m]).unsqueeze(-1)
            
            # Negative logits
            l_neg = torch.einsum('nc,ck->nk', [z1, self.queue_1.clone().detach()])
            
            # Logits
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.temperature
            
            # Labels: positive is at index 0
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
            
            loss = F.cross_entropy(logits, labels)
            
            # Dequeue and enqueue
            self._dequeue_and_enqueue(z1_m, self.queue_1)
        else:
            # Default to SimCLR style
            loss = self.criterion(z1, z2, labels)
        
        if return_embeddings:
            return {
                'loss': loss,
                'embeddings_1': h1,
                'embeddings_2': h2,
                'projections_1': z1,
                'projections_2': z2
            }
        else:
            return loss
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue):
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % queue.shape[1]
        
        self.queue_ptr[0] = ptr