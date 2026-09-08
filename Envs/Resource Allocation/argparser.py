import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-n_servers', type=int, default=10)
parser.add_argument('-n_resources', type=int, default=3)
parser.add_argument('-n_tasks', type=int, default=200, help='Use all tasks by default.')
parser.add_argument('-warmup', type=int, default=1000, help='Use all tasks by default.')
parser.add_argument('-w1', type=float, default=0.1)
parser.add_argument('-w2', type=float, default=0.0)
parser.add_argument('-w3', type=float, default=0.2)
parser.add_argument('-P_0', type=int, default=87)
parser.add_argument('-P_100', type=int, default=145)
parser.add_argument('-T_on', type=int, default=30)
parser.add_argument('-T_off', type=int, default=30)
parser.add_argument('-seed', type=int, default=0)
args = parser.parse_args()





