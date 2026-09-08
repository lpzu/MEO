# MEO: Mining Reliable Expert Signals for Offline Reinforcement Learning in Wireless Networks

## Introduction

MEO mines reliable expert signals from imperfect offline datasets collected in
wireless networks, and converts them into a compensable reward that any offline
RL algorithm can be trained on. It consists of three steps:

1. **Cluster-wise Top-K expert selection.** K-means is applied over the
   observable states, and within each cluster the top-K samples by reward are
   taken as the expert set; the rest form the non-expert set.
2. **Expert-guided latent space structuring.** A CVAE is trained on state-action
   pairs with a contrastive objective that keeps expert embeddings compact while
   separating them from non-expert embeddings by a margin.
3. **Compensable reward.** The latent distance from each sample to the expert
   centroid is used to relabel the reward, so that samples closer to expert
   behavior receive more favorable training signals.

## Structure

```
MEO/
├── dataset_class.py         # Step 1a: K-means over observable states
├── dataset-split.py         # Step 1b: top-K split into expert / non-expert
├── train_vae.py             # Step 2: contrastive-enhanced CVAE
├── label_reward.py          # Step 3: latent distance to expert centroid
├── vae.py                   # CVAE architecture
├── utils.py                 # replay buffer / helpers
├── requirements.txt
├── datasets/
│   ├── datasets_collect/    # input: collected_data.hdf5
│   ├── datasets_kmeans/     # cluster labels
│   ├── datasets_split/      # expert / non-expert subsets
│   └── datasets_labeled/    # output: relabelled dataset
└── models/                  # trained CVAE
```

## Setup

```bash
pip install -r requirements.txt
```

Place the collected dataset at `datasets/datasets_collect/collected_data.hdf5`
with fields `observations`, `actions`, `rewards`, and `terminals`, each of
length `N`.

## Run order

All commands are run from inside `MEO/`. The cluster size must be identical at
every step, as it appears both in the HDF5 field name and in the intermediate
file names.

```bash
# Step 1a: K-means clustering
python dataset_class.py
#   -> datasets/datasets_kmeans/collected_data_kmeans.hdf5

# Step 1b: cluster-wise top-K split
python dataset-split.py
#   -> datasets/datasets_split/collected_data_{experts,normal}_<clusters>.hdf5

# Step 2: train the contrastive CVAE
python train_vae.py --clusters <clusters>
#   -> models/vae_model.pt

# Step 3: relabel the rewards
python label_reward.py --clusters <clusters> --tao <tau>
#   -> datasets/datasets_labeled/collected_data_labeled.hdf5
```

The cluster size and K are set in `dataset_class.py` and `dataset-split.py`;
state dimension, action space size, and the CVAE hyperparameters are exposed as
arguments of `train_vae.py` and `label_reward.py`. Values used for each scenario
are reported in the paper.

## Output

`collected_data_labeled.hdf5` is the relabelled dataset, ready to be used
directly for offline RL. Its `rewards` field already carries the compensable
reward: the latent distance is scaled by `tau`, added to the original reward,
and the result is normalized. The raw `distances` field is also kept, so a
different `tau` can be applied without rerunning the pipeline.

## Offline RL training

MEO is independent of the downstream algorithm and can be combined with any
offline RL method. We use [d3rlpy](https://github.com/takuseno/d3rlpy) in the
paper; load the relabelled dataset, apply the reward transformation above, and
train as usual.
