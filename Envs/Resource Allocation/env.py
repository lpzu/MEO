# -*- coding: utf-8 -*-
import os
import heapq
import numpy as np
from collections import deque
from argparser import args
import random
import pandas as pd


class Env():
    def __init__(self, seed=args.seed):
        
        self.P_0 = args.P_0
        self.P_100 = args.P_100
        self.T_on = args.T_on
        self.T_off = args.T_off
        self.n_servers = args.n_servers
        self.w1 = args.w1
        self.w2 = args.w2
        self.w3 = args.w3
        self.seed = seed
        
        np.random.seed(self.seed)
    
        #  data paths
        self.machine_meta_path = os.path.join('data', 'machine_meta.csv')
        self.batch_task_path = os.path.join('data', 'batch_task.csv')

        #  data columns
        self.machine_meta_cols = [
            'machine_id',  # uid of machine
            'time_stamp',  # time stamp, in second
            'failure_domain_1',  # one level of container failure domain
            'failure_domain_2',  # another level of container failure domain
            'cpu_num',  # number of cpu on a machine
            'mem_size',  # normalized memory size. [0, 100]
            'status',  # status of a machine
        ]

        self.batch_task_cols = [
            'task_name',  # task name. unique within a job
            'instance_num',  # number of instances  
            'job_name',  # job name
            'task_type',  # task type
            'status',  # task status
            'start_time',  # start time of the task
            'end_time',  # end of time the task
            'plan_cpu',  # number of cpu needed by the task, 100 is 1 core
            'plan_mem'  # normalized memorty size, [0, 100]
        ]

        self.cur = 0
        self.loadcsv()
        self.latency = []
        
        
    def loadcsv(self):

        #  read csv into DataFrames
        self.machine_meta = pd.read_csv(self.machine_meta_path, header=None, names=self.machine_meta_cols)
        self.machine_meta = self.machine_meta[self.machine_meta['time_stamp'] == 0]
        self.machine_meta = self.machine_meta[['machine_id', 'cpu_num', 'mem_size']]

        self.batch_task = pd.read_csv(self.batch_task_path, header=None, names=self.batch_task_cols)
        self.batch_task = self.batch_task[self.batch_task['status'] == 'Terminated']
        #compute the last time of each task and output the count that last time is more than 10
        self.batch_task['last_time'] = self.batch_task['end_time'] - self.batch_task['start_time']
        self.batch_task['last_time'] = self.batch_task['last_time'].astype(int)
        self.batch_task = self.batch_task[self.batch_task['last_time'] >= 3]
        self.batch_task = self.batch_task[self.batch_task['last_time'] <= 2000]
        self.batch_task['start_time'] = self.batch_task['start_time'] / 2
        self.batch_task['start_time'] = (self.batch_task['start_time'] - np.min(self.batch_task['start_time']))
        #print(np.min(self.batch_task['start_time']))
        #print(self.batch_task['start_time'])
        self.batch_task['start_time'] = self.batch_task['start_time'].astype(int)
        self.batch_task['end_time'] = self.batch_task['start_time'] + self.batch_task['last_time'].astype(int)

        #exit()

        #rescale the plan_cpu and plan_mem
        self.batch_task = self.batch_task[self.batch_task['plan_cpu'] <= 100]
        self.batch_task = self.batch_task[self.batch_task['plan_mem'] <= 1.]
        #print("task_num", len(self.batch_task))
        #rescale the plan_cpu and plan_mem to [0, 100]
        #high occupation of cpu and mem
        #self.batch_task['plan_cpu'] = 10 + self.batch_task['plan_cpu'] * 30 / 500.
        #self.batch_task['plan_mem'] = 20 + self.batch_task['plan_mem'] * 20.
        #low occupation of cpu and mem
        self.batch_task['plan_cpu'] = 3 + self.batch_task['plan_cpu'] * 17 / 100.
        self.batch_task['plan_mem'] = 3 + self.batch_task['plan_mem'] * 22.
        #add noise to the plan_cpu
        self.batch_task['plan_cpu'] = self.batch_task['plan_cpu'] + np.random.normal(0, 2, len(self.batch_task)) 
        #rescale the plan_cpu to [0, 100], if the plan_cpu is less than 5, set it to 5, if the plan_cpu is more than 100, set it to 100
        self.batch_task['plan_cpu'] = self.batch_task['plan_cpu'].apply(lambda x: 1 if x < 1 else x)
        self.batch_task['plan_cpu'] = self.batch_task['plan_cpu'].apply(lambda x: 100 if x > 100 else x)
        #add noise to the plan_mem
        self.batch_task['plan_mem'] = self.batch_task['plan_mem'] + np.random.normal(0, 2, len(self.batch_task))
        #rescale the plan_mem to [0, 100], if the plan_mem is less than 5, set it to 5, if the plan_mem is more than 100, set it to 100
        self.batch_task['plan_mem'] = self.batch_task['plan_mem'].apply(lambda x: 1 if x < 1 else x)
        self.batch_task['plan_mem'] = self.batch_task['plan_mem'].apply(lambda x: 100 if x > 100 else x)
        #set plan_cpu and plan_mem to int
        self.batch_task['plan_cpu'] = self.batch_task['plan_cpu'].astype(int)
        self.batch_task['plan_mem'] = self.batch_task['plan_mem'].astype(int)
        #output the count of batch_task
        #print("task_num", len(self.batch_task))

        self.batch_task = self.batch_task.sort_values(by='start_time')
        self.total_task_num = len(self.batch_task)
        self.n_machines = self.n_servers
        self.n_tasks = args.n_tasks
        self.warmup = args.warmup

    def reset(self):
        self.cur = 0
        self.power_usage = 0
        self.latency = 0
        self.machines = [ Machine(
            100, 100,
            self.machine_meta.iloc[i]['machine_id']
        ) for i in range(self.n_machines) ]
        start_task = np.random.randint(0, self.total_task_num - self.n_tasks - self.warmup)
        self.tasks = [ Task(
            self.batch_task.iloc[i]['task_name'],
            self.batch_task.iloc[i]['start_time'],
            self.batch_task.iloc[i]['end_time'],
            self.batch_task.iloc[i]['plan_cpu'],
            self.batch_task.iloc[i]['plan_mem'],
        ) for i in range(start_task, start_task + self.n_tasks + self.warmup) ]

        return np.array(self.get_states()+[0]*self.n_machines)

    def step(self, action):
        #print("action", action)
        self.cur_time = self.tasks[self.cur].start_time
        cur_task = self.tasks[self.cur]
        
        done = False
        self.letency_len = 0
        self.cur += 1
        if self.cur == self.n_tasks + self.warmup:
            latency = [self.tasks[i].latency for i in range(self.warmup, self.cur)]
            begin = [self.tasks[i].begin for i in range(self.warmup, self.cur)]
            # select the tasks that have been started
            latency = [latency[i] for i in range(len(latency)) if begin[i] and latency[i] > 0]

            if len(latency) > 0:
                self.latency = np.mean(latency)    
                self.letency_len = len(latency)

            done = True
            self.cur = 0

        if self.cur == self.warmup+1:
            self.power_usage_warmup = np.sum([m.power_usage for m in self.machines]) 
        if self.cur >= self.warmup+1:
            self.power_usage = np.round(np.sum([m.power_usage for m in self.machines]) - self.power_usage_warmup, 4)


        nxt_task = self.tasks[self.cur]

        for m in self.machines:
            m.process(self.cur_time)

        i = 0
        self.machine_running = []
        self.machine_pending = []
        for m in self.machines:
            i += 1
            if self.cur >= self.warmup:
                machine_running, machine_pending = self.running_load(m,i)
                self.machine_running.append(machine_running)
                self.machine_pending.append(machine_pending)
        
        penalty = self.calc_queue_penalty()
        self.machines[action].add_task(cur_task)
        state = self.get_states() + penalty + [nxt_task.plan_cpu/100., nxt_task.plan_mem/100., nxt_task.last_time/2000.]
        state = np.array(state)
        reward = self.get_reward(action, penalty)
        return state, reward, done, [self.latency, self.letency_len, self.power_usage]
    
    def get_states(self):
        states = [m.cpu_idle /100 for m in self.machines] + \
                 [m.mem_empty /100 for m in self.machines] + \
                 [len(m.pending_queue) / 10 for m in self.machines]
        
        return states  # scale

    def get_reward(self, action, penalty):
        total_power = self.calc_total_power()
        #total_idle = self.calc_total_idle()
        server_balance = self.calc_server_balance()
        pending_penalty = penalty[action] / (np.sum(penalty)+1e-10)
        return -self.w1*total_power -self.w2*server_balance -self.w3*pending_penalty
       
    def calc_total_power(self):
        total_power = 0
        for m in self.machines:
            total_power += 2 * m.cpu() - m.cpu()**(1.4)
        return total_power
    
    def calc_total_idle(self):
        total_idle = 0
        for m in self.machines:
            total_idle += 1 - m.cpu()
        return total_idle
        
    def calc_total_latency(self):
        for t in self.tasks:
            latency = [t.start_time - t.arrive_time]
        for i in range(1, len(latency)):
            latency[i] = latency[i] + latency[i - 1]
        return np.sum(latency)
    
    #def calc_server_balance using var(server_loads)
    def calc_server_balance(self):
        server_loads_cpu = [m.cpu() for m in self.machines]
        server_loads_mem = [m.mem() for m in self.machines]
        balance_cpu = max(server_loads_cpu) - min(server_loads_cpu)
        balance_mem = max(server_loads_mem) - min(server_loads_mem)

        return np.std(server_loads_cpu)*0.5 + balance_cpu
    
    def calc_queue_penalty(self):
        penalty = [self.calc_queue_factor(m) for m in self.machines]
        penalty = penalty / (np.sum(penalty)+1e-10)

        return list(penalty)
    
    def calc_queue_factor(self, m):
        factor = 0
        for i in range(len(m.pending_queue)):
            factor += (m.pending_queue[i].plan_cpu+m.pending_queue[i].plan_mem)*m.pending_queue[i].last_time
        return factor
    
    
    def running_load(self, machine, i):
        return len(machine.running_queue), len(machine.pending_queue)
            #print("machine", i-1)                
            #print("pending_queue", len(machine.pending_queue))               
            #print("running_queue", len(machine.running_queue))

    def calc_total_pending(self):
        total_pending = 0
        for m in self.machines:
            total_pending += len(m.pending_queue)
        return total_pending


class Task(object):
    def __init__(self, name, start_time, end_time, plan_cpu, plan_mem):
        self.end_time = end_time
        self.name = name
        self.arrive_time = start_time
        self.last_time = end_time - start_time
        self.plan_cpu = plan_cpu
        self.plan_mem = plan_mem
        self.start_time = self.arrive_time
        self.begin = False
        self.latency = 0

    def start(self, start_time):
        self.latency = start_time - self.arrive_time
        self.begin = True
        self.start_time = start_time
        self.end_time = start_time + self.last_time

    def __lt__(self, other):
        return self.end_time < other.end_time


class Machine():
    def __init__(self, cpu_num, mem_size, machine_id):
        self.machine_id = machine_id
        self.P_0 = args.P_0
        self.P_100 = args.P_100

        self.pending_queue = deque()
        self.running_queue = []

        self.cpu_num = cpu_num
        self.mem_size = mem_size
        self.cpu_idle = cpu_num
        self.mem_empty = mem_size

        self.cur_time = 0
        self.power_usage = 0
        

    def cpu(self):
        return 1 - self.cpu_idle / self.cpu_num
    
    def mem(self):
        return 1 - self.mem_empty / self.mem_size

    def add_task(self, task):
        self.pending_queue.append(task)
        self.process_new_task()

    def process_new_task(self):
        if len(self.pending_queue) == 1 and self.enough_resource(self.pending_queue[0]):
            task = self.pending_queue.popleft()
            task.start(self.cur_time)
            self.cpu_idle -= task.plan_cpu
            self.mem_empty -= task.plan_mem
            heapq.heappush(self.running_queue, task)

    def process_running_queue(self, cur_time):
        if self.is_empty(self.running_queue):
            return False
        
        if self.running_queue[0].end_time <= cur_time:
            task = heapq.heappop(self.running_queue)

            self.power_usage += self.calc_power(task.end_time)
            self.cur_time = task.end_time
            
            self.cpu_idle += task.plan_cpu
            self.mem_empty += task.plan_mem
            return True
        
        return False

    def process_pending_queue(self):
        if self.is_empty(self.pending_queue):
            return False
        if not self.enough_resource(self.pending_queue[0]):
            return False

        task = self.pending_queue.popleft()
        task.start(self.cur_time)
        self.cpu_idle -= task.plan_cpu
        self.mem_empty -= task.plan_mem
        heapq.heappush(self.running_queue, task)

        return True

    def process(self, cur_time):
        if self.cur_time == 0:  ## the first time, no task has come before 
            self.cur_time = cur_time
            return

        while self.process_running_queue(cur_time):
            while self.process_pending_queue():
                pass
            
        self.power_usage += self.calc_power(cur_time)
        self.cur_time = cur_time

    def enough_resource(self, task):
        return task.plan_cpu <= self.cpu_idle and task.plan_mem <= self.mem_empty

    def is_empty(self, queue):
        return len(queue) == 0
    
    def calc_power(self, cur_time):
        cpu = self.cpu()
        return (self.P_0 + (self.P_100 - self.P_0) * (2*cpu - cpu**1.4)) * (cur_time - self.cur_time)
        
