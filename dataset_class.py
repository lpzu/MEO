import h5py
import numpy as np
import faiss
from tqdm import tqdm
import os


in_path  = './datasets/datasets_collect/collected_data.hdf5'
out_path = './datasets/datasets_kmeans/collected_data_kmeans.hdf5'

file = h5py.File(in_path, 'r')

observations = np.ascontiguousarray(file['observations'][:], dtype=np.float32)
actions      = file['actions'][:]
rewards      = file['rewards'][:]
terminals    = file['terminals'][:]
file.close()

size, state_dim = observations.shape       # 自动读 size
print('size:', size, 'state_dim:', state_dim)

n_to_select = [20000,]

res = faiss.StandardGpuResources()

dataset = {
    'observations': observations,
    'actions': actions,
    'rewards': rewards,
    'terminals': terminals,
}

for n in tqdm(n_to_select, desc='K-means training', ncols=100):
    kmeans = faiss.Kmeans(state_dim, n, niter=200, seed=1, verbose=False)
    kmeans.index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(state_dim))
    kmeans.train(observations)
    _, labels = kmeans.index.search(observations, 1)
    dataset[f'label_{n}'] = labels.flatten().astype(np.int32)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with h5py.File(out_path, 'w') as labeled_file:
    for key in dataset:
        labeled_file.create_dataset(key, data=dataset[key], compression='gzip')