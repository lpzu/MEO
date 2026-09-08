import h5py
import numpy as np
from tqdm import tqdm
import os


file = h5py.File('./datasets/datasets_kmeans/collected_data_kmeans.hdf5', 'r')
observations = file['observations'][:]
actions = file['actions'][:]
rewards = file['rewards'][:]
terminals = file['terminals'][:]

n_to_select = {20000: 1}

for n in tqdm(n_to_select, desc='data split', ncols=100):
    labels = file[f'label_{n}'][:]

    data_per_cluster = {i: [] for i in range(n)}
    for label, reward, observation, action, terminal in zip(labels, rewards, observations, actions, terminals):
        data_per_cluster[label].append((reward, observation, action, terminal))

    expert_data = {i: sorted(data, key=lambda x: x[0], reverse=True)[:n_to_select[n]] for i, data in data_per_cluster.items()}
    normal_data = {i: sorted(data, key=lambda x: x[0], reverse=True)[n_to_select[n]:] for i, data in data_per_cluster.items()}

    os.makedirs('./datasets/datasets_split/', exist_ok=True)
    with h5py.File('./datasets/datasets_split/'+'collected_data'+f'_experts_{n}.hdf5', 'w') as f:
        datasets_obs = []
        datasets_actions = []
        datasets_rewards = []
        datasets_terminals = []
        for i, data in expert_data.items():
            for item in data:
                datasets_obs.append(np.array(item[1]))
                datasets_actions.append(np.array(item[2]))
                datasets_rewards.append(np.array(item[0]))
                datasets_terminals.append(np.array(item[3]))

        datasets_obs = np.array(datasets_obs).astype(np.float32)
        datasets_actions = np.array(datasets_actions).astype(np.int32)
        datasets_rewards = np.array(datasets_rewards).astype(np.float32)
        datasets_terminals = np.array(datasets_terminals)

        dataset_size = len(datasets_obs)
        print(dataset_size)
        assert dataset_size == len(datasets_actions)
        assert dataset_size == len(datasets_rewards)
        assert dataset_size == len(datasets_terminals)

        dataset = {
            'observations': datasets_obs,
            'actions': datasets_actions,
            'rewards': datasets_rewards,
            'terminals': datasets_terminals,
            }
        for key in dataset:
            f.create_dataset(key, data=dataset[key], compression='gzip')
    f.close()

    with h5py.File('./datasets/datasets_split/'+'collected_data'+f'_normal_{n}.hdf5', 'w') as f:
        datasets_obs = []
        datasets_actions = []
        datasets_rewards = []
        datasets_terminals = []
        for i, data in normal_data.items():
            for item in data:
                datasets_obs.append(np.array(item[1]))
                datasets_actions.append(np.array(item[2]))
                datasets_rewards.append(np.array(item[0]))
                datasets_terminals.append(np.array(item[3]))

        datasets_obs = np.array(datasets_obs).astype(np.float32)
        datasets_actions = np.array(datasets_actions).astype(np.int32)
        datasets_rewards = np.array(datasets_rewards).astype(np.float32)
        datasets_terminals = np.array(datasets_terminals)

        dataset_size = len(datasets_obs)
        print(dataset_size)
        assert dataset_size == len(datasets_actions)
        assert dataset_size == len(datasets_rewards)
        assert dataset_size == len(datasets_terminals)

        dataset = {
            'observations': datasets_obs,
            'actions': datasets_actions,
            'rewards': datasets_rewards,
            'terminals': datasets_terminals,
            }
        for key in dataset:
            f.create_dataset(key, data=dataset[key], compression='gzip')
    f.close()

file.close()