import torch
from torch import nn
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
import h5py

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
# dataset
parser.add_argument('--clusters', type=str, default='20000')
parser.add_argument('--tao', type=float, default=0.3)
# model
parser.add_argument('--state_dim', type=int, default=43)
parser.add_argument('--action_dim', type=int, default=8)
parser.add_argument('--action_space_size', type=int, default=10)
parser.add_argument('--hidden_dim', type=int, default=128) 
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--lambda_loss', type=float, default=50.0)
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

# expert dataset
expert_demo = utils.ReplayBuffer(state_dim, action_dim)    
expert_demo.convert_selfdata('./datasets/datasets_split/'+'collected_data'+'_experts_'+args.clusters+'.hdf5')
expert_states = torch.from_numpy(expert_demo.state).to(device)
expert_actions = torch.from_numpy(expert_demo.action).to(device)
expert_size = expert_states.shape[0]

vae = VAE(state_dim, action_space_size, action_dim, latent_dim, hidden_dim=args.hidden_dim).to(device)
vae.load_state_dict(torch.load('./models/vae_model.pt'))

_, z, _, _ = vae(expert_states, expert_actions)
center_z = torch.mean(z, 0)

file = h5py.File('./datasets/datasets_collect/' + 'collected_data' + '.hdf5', 'r')
observations = torch.from_numpy(file['observations'][:])
actions = torch.from_numpy(file['actions'][:])
rewards = torch.from_numpy(file['rewards'][:])
terminals = torch.from_numpy(file['terminals'][:])
file.close()

distances = []
for i in tqdm(range(observations.shape[0]), ncols=100):
    _, z_labling, mean, _ = vae(observations[i].to(device), actions[i].to(device))
    distances_z = torch.norm(z_labling - center_z)
    distances.append(distances_z.item())

distances = np.array(distances).astype(np.float32)
rewards = rewards.numpy()
C_r = -args.tao * distances / np.max(distances)
rewards = rewards + C_r
rewards = -rewards / np.min(rewards)

dataset = {
        'observations': observations.numpy(),
        'actions': actions.numpy(),
        'rewards': rewards,
        'terminals': terminals.numpy(),
        'distances': distances,
    }

labeled_file = h5py.File('./datasets/datasets_labeled/'+'collected_data_labeled.hdf5', 'w')
for key in dataset:
    labeled_file.create_dataset(key, data=dataset[key], compression='gzip')

labeled_file.close()
