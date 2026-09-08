import csv
import numpy as np
import tqdm
import warnings

import argparse
import torch
from vae import VAE
import utils
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
# dataset
parser.add_argument('--dataset', type=str, default='roundrobin')  # greedy, roundrobin, PPO
parser.add_argument('--clusters', type=str, default='20000')
parser.add_argument('--save_dir', type=str, default='./tmp/')
# model
parser.add_argument('--state_dim', type=int, default=43)
parser.add_argument('--action_dim', type=int, default=8)
parser.add_argument('--action_space_size', type=int, default=10)
parser.add_argument('--hidden_dim', type=int, default=128) 
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--lambda_loss', type=float, default=40.0)
parser.add_argument('--gamma_loss', default=5.0, type=float)
parser.add_argument('--margin', default=0.3, type=float)
# train
parser.add_argument('--num_iters', type=int, default=int(2e4)) 
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--weight_decay', default=0.0001, type=float)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train vae
state_dim = args.state_dim
action_dim = args.action_dim
action_space_size = args.action_space_size
latent_dim = action_dim * 2
print('state_dim:', state_dim, 'action_dim:', action_dim, 'action_space_size:', action_space_size, 'latent_dim:', latent_dim)

# original dataset
normal_demo = utils.ReplayBuffer(state_dim, action_dim)
normal_demo.convert_selfdata('./datasets/datasets_split/'+'collected_data'+'_normal_'+args.clusters+'.hdf5')
normal_states = torch.from_numpy(normal_demo.state).to(device)
normal_actions = torch.from_numpy(normal_demo.action).to(device)
normal_size = normal_states.shape[0]

expert_demo = utils.ReplayBuffer(state_dim, action_dim)    
expert_demo.convert_selfdata('./datasets/datasets_split/'+'collected_data'+'_experts_'+args.clusters+'.hdf5')
expert_states = torch.from_numpy(expert_demo.state).to(device)
expert_actions = torch.from_numpy(expert_demo.action).to(device)
expert_size = expert_states.shape[0]
        
vae = VAE(state_dim, action_space_size, action_dim, latent_dim, hidden_dim=args.hidden_dim).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lambda_loss = args.lambda_loss
gamma_loss = args.gamma_loss
batch_size = args.batch_size
margin = args.margin

indices_z = torch.tensor([56, 57, 58, 59, 60, 61, 62, 63]).to(device)
indices_normal_z = torch.tensor(range(56)).to(device)
loss_values = []

for step in tqdm(range(args.num_iters), desc='train', ncols=100):
    normal_idx = torch.randperm(normal_size, device=device)[:batch_size-8]
    expert_idx = torch.randperm(expert_size, device=device)[:8]
    states_normal = normal_states[normal_idx]
    actions_normal = normal_actions[normal_idx]
    states_expert = expert_states[expert_idx]
    actions_expert = expert_actions[expert_idx]
    train_states = torch.cat((states_normal, states_expert), dim=0)
    train_actions = torch.cat((actions_normal, actions_expert), dim=0)

    # Variational Auto-Encoder Training
    action_distributions, z, mean, std = vae(train_states, train_actions)

    # Regularization Loss
    sub_z = torch.index_select(z, 0, indices_z).to(device)
    z_loss = torch.std(sub_z, 0, unbiased=False).mean()
    regularization_loss = lambda_loss * z_loss   

    # Negative Pair Loss
    sub_normal_z = torch.index_select(z, 0, indices_normal_z).to(device)
    center_z = torch.mean(sub_z, 0)
    center_z = center_z.repeat(sub_normal_z.shape[0], 1)
    distances_z = torch.norm(sub_normal_z - center_z, dim=1)
    negative_pair_loss = gamma_loss * (F.relu(margin - distances_z.mean()))

    # Total VAE Loss
    contrastive_loss = regularization_loss + negative_pair_loss
    recon_loss = F.cross_entropy(action_distributions, train_actions.long())
    KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
    
    vae_loss = recon_loss + args.beta * KL_loss + contrastive_loss

    optimizer.zero_grad()
    vae_loss.backward()
    optimizer.step()
    
torch.save(vae.state_dict(), './models/vae_model.pt')
