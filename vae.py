import torch
import torch.nn.functional as F
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):

    def __init__(self, state_dim, action_space_size, action_dim, latent_dim, hidden_dim):
        super(VAE, self).__init__()

        self.action_embedding = nn.Embedding(num_embeddings=action_space_size, embedding_dim=action_dim)
        
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, action_space_size)

        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        action_embeddings = self.action_embedding(action.to(self.device))

        mean, std = self.encode(state, action_embeddings)
        z = mean + std * torch.randn_like(std)
        action_distributiuons = self.decode(state, z)
        
        return action_distributiuons, z, mean, std

    def encode(self, state, action_embeddings):
        z = F.relu(self.e1(torch.cat([state, action_embeddings], -1)))
        z = F.relu(self.e2(z))
        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        
        return mean, std

    def decode(self, state, z):
        a = F.relu(self.d1(torch.cat([state, z], -1)))
        a = F.relu(self.d2(a))
        action_logits = self.d3(a)

        return action_logits
