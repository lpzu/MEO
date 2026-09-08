# Resource Allocation Testbed (ORAN-Cloud Scenario)

Simulation of server resource allocation in an ORAN-Cloud Centralized Unit.
Incoming tasks arrive over time and the agent assigns each one to a server;
when a server has no free CPU or memory the task is queued. The reward trades
off power consumption, load balance across servers, and queueing delay.

## Data

The workload is driven by the **Alibaba Cluster Trace**, available at
<https://github.com/alibaba/clusterdata>. 

```
data/
├── machine_meta.csv
└── batch_task.csv
```


## Requirements

```
python >= 3.8
numpy, pandas, h5py
```

## Files

```
env.py         # environment: servers, tasks, power and queue model
argparser.py   # simulation parameters
run_env.py     # entry point, runs a greedy baseline policy
```

## Configuration

Set in `argparser.py`:

- `-n_servers` — number of servers, which is also the size of the action space
- `-n_tasks` — tasks per episode, after the warm-up
- `-warmup` — warm-up tasks used to reach a realistic load before logging starts
- `-w1`, `-w2`, `-w3` — reward weights for power, load balance, and queue penalty
- `-P_0`, `-P_100` — server power at idle and at full CPU utilisation
- `-seed` — random seed

## Running

```bash
python run_env.py
```