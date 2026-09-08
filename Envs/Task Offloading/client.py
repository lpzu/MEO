# -*- coding: utf-8 -*-
import d3rlpy
import socket
import time
import os
import torch
import numpy as np
import pickle
import random
from PIL import Image
from ultralytics import YOLO
from multiprocessing import Process, Pipe
import h5py
from tqdm import tqdm
import subprocess
import argparse
 

# Client settings
SERVER_IP = '192.168.0.75'
PORT = 5000

# Paths to GPU power sensor files
VOLTAGE_PATH = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input"
CURRENT_PATH = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr1_input"

# Sampling interval for power monitoring
SAMPLING_INTERVAL = 0.01


def power_monitor(pipe, initial_energy):
    """Monitor GPU power consumption and update remaining energy."""
    total_energy = 0.0
    remaining_energy = initial_energy
    last_time = time.time()

    print("[Power Monitor] Starting power monitoring...")

    while True:
        try:
            with open(VOLTAGE_PATH, 'r') as v_file, open(CURRENT_PATH, 'r') as c_file:
                voltage_mv = int(v_file.read().strip())  # Voltage in mV
                current_ma = int(c_file.read().strip())  # Current in mA

            # Convert to Watts
            power_w = (voltage_mv * current_ma) / 1e6

            # Calculate elapsed time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Accumulate energy consumption in Joules (W * s)
            total_energy += power_w * dt

            # Update remaining energy (Wh)
            remaining_energy -= power_w * dt / 3600

            # Send latest remaining energy to main process
            if pipe.poll():
                command = pipe.recv()
                if command == "GET_REMAINING":
                    pipe.send(remaining_energy)
                elif command == "STOP":
                    print("[Power Monitor] Stopping...")
                    break

            time.sleep(SAMPLING_INTERVAL)

        except Exception as e:
            print(f"[Power Monitor] Error: {e}")
            break

class Env_Edge:
    def __init__(self):
        print("Client is initializing ... ...")
        self.model = YOLO("./models/yolo11n.pt").to('cuda')
        self.model_seg = YOLO("./models/yolo11n-seg.pt").to('cuda')
        self.model_pose = YOLO("./models/yolo11n-pose.pt").to('cuda')
        self.model_cls = YOLO("./models/yolo11n-cls.pt").to('cuda')
        self.model_obb = YOLO("./models/yolo11n-obb.pt").to('cuda')

        print("Loading dataset ... ...")
        self.image_coco_dir = "./datasets/coco/"
        self.image_coco_files = [os.path.join(self.image_coco_dir, f) for f in os.listdir(self.image_coco_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        self.image_dota_dir = "./datasets/dota/"
        self.image_dota_files = [os.path.join(self.image_dota_dir, f) for f in os.listdir(self.image_dota_dir) if f.lower().endswith(('.jpg', '.png'))]
        self.image_imagenet_dir = "./datasets/imagenet/"
        self.image_imagenet_files = [os.path.join(self.image_imagenet_dir, f) for f in os.listdir(self.image_imagenet_dir) if f.lower().endswith(('.jpg', '.jpeg'))]

        self.initialize_model()
        self.initialize_socket()
        self.start_power_monitor()
        self.seed = 11

    def initialize_model(self):
        imgs = self.load_images(self.image_coco_files, 2)
        imgs_seg = self.load_images(self.image_coco_files, 2)
        imgs_pose = self.load_images(self.image_coco_files, 2)
        imgs_cls = self.load_images(self.image_imagenet_files, 2)
        imgs_obb = self.load_images(self.image_dota_files, 2)

        # Run predictions using reusable function
        self.run_predictions(imgs, self.model, initialized=True) 
        self.run_predictions(imgs_seg, self.model_seg, initialized=True)  
        self.run_predictions(imgs_pose, self.model_pose, initialized=True)  
        self.run_predictions(imgs_cls, self.model_cls, initialized=True)  
        self.run_predictions(imgs_obb, self.model_obb, initialized=True)  

    def initialize_socket(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((SERVER_IP, PORT))
        print("Client is connected to the server ... ...")

    def start_power_monitor(self):
        self.parent_conn, child_conn = Pipe()
        initial_energy = random.uniform(1.0, 2.0)  # Set initial energy

        self.power_monitor_proc = Process(target=power_monitor, args=(child_conn, initial_energy))
        self.power_monitor_proc.start()
    
    def get_remaining_energy(self):
        self.parent_conn.send("GET_REMAINING")
        #if self.parent_conn.poll():
        remaining_energy = self.parent_conn.recv()
        return remaining_energy

    def stop_power_monitor(self):
        self.parent_conn.send("STOP")
        self.power_monitor_proc.join()
    
    def load_images(self, file_list, num_images):
        sampled_files = random.sample(file_list, num_images)
        return [np.array(Image.open(img_path)) for img_path in sampled_files]

    def run_predictions(self, images, model, initialized=False):
        if not initialized:
            for image in images:
                results = model.predict(image, device='cuda')
        else:
            for _ in range(5): 
                for image in images:
                    results = model.predict(image, device='cuda')

    def model_clarify(self, task_type):
        if task_type[0] == 1:
            return self.model, "det"  # Return the model and task type for detection
        elif task_type[1] == 1:
            return self.model_seg, "seg"  # Return the model and task type for segmentation
        elif task_type[2] == 1:
            return self.model_pose, "pose"  # Return the model and task type for pose estimation
        elif task_type[3] == 1:
            return self.model_cls, "cls"  # Return the model and task type for classification
        elif task_type[4] == 1:
            return self.model_obb, "obb"  # Return the model and task type for OBB
        else:
            raise ValueError("Invalid task type")
        
    def set_gpu_freq(self, remaining_energy):
        # Example frequencies: 306000000 under 20%; 408000000 under %50; 510000000 under %80; 612000000
        if remaining_energy < 0.2:
            freq = 306000000
        elif remaining_energy < 0.5:
            freq = 408000000
        elif remaining_energy < 0.8:
            freq = 510000000
        else:
            freq = 612000000

        password = ""
        try:
            subprocess.run(
            f"echo {password} | sudo -S sh -c 'echo {freq} > /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/max_freq'", 
            shell=True, check=True)
        
            # Write min_freq
            subprocess.run(
            f"echo {password} | sudo -S sh -c 'echo {freq} > /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/min_freq'", 
            shell=True, check=True)

        except Exception as e:
            print(f"Failed to set GPU frequency: {e}")

        return freq  # Return the selected frequency for the GPU
    
    def train_gpu_freq(self):
        freq = 816000000        
        password = ""
        try:
            subprocess.run(
            f"echo {password} | sudo -S sh -c 'echo {freq} > /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/max_freq'", 
            shell=True, check=True)
        
            # Write min_freq
            subprocess.run(
            f"echo {password} | sudo -S sh -c 'echo {freq} > /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/min_freq'", 
            shell=True, check=True)

        except Exception as e:
            print(f"Failed to set GPU frequency: {e}")

    def reset(self):
        # Reset the environment to a new state
        random.seed(self.seed)
        self.seed  = 11 + (self.seed-10) % 5
        self.done = False  
        self.task_type = np.zeros(5)  # 0: detection, 1: segmentation, 2: pose estimation, 3: classification, 4: OBB
        type_index = random.choices([0, 1, 2, 3, 4], [0.25, 0.2, 0.2, 0.3, 0.05])[0]
        self.task_type[type_index] = 1  # Randomly select one task type
        self.batch_size = random.choices([4, 8, 16, 24, 32], [0.2, 0.2, 0.3, 0.2, 0.1])[0]  # Randomly select batch size
        if self.task_type[0] == 1 or self.task_type[1] == 1 or self.task_type[2] == 1:
            self.images = self.load_images(self.image_coco_files, self.batch_size)
        elif self.task_type[3] == 1:
            self.images = self.load_images(self.image_imagenet_files, self.batch_size)
        elif self.task_type[4] == 1:
            self.images = self.load_images(self.image_dota_files, self.batch_size)
        self.avg_image_size = [np.mean([img.shape[0] for img in self.images])/640., np.mean([img.shape[1] for img in self.images])/640.]

        # Power monitoring setup
        self.stop_power_monitor()  # Stop any previous monitoring
        time.sleep(.1)  # Ensure the previous process has stopped
        self.start_power_monitor()
        time.sleep(.1)  # Wait for the power monitor to start
        self.remaining_energy = self.get_remaining_energy()/2.
        self.gpu_freq = self.set_gpu_freq(self.remaining_energy)/1e8

        state = np.concatenate((self.task_type, [self.batch_size], self.avg_image_size, [self.remaining_energy], [self.gpu_freq]))
        self.steps = 0
        
        return state  # Return the state vector for the selected task type and batch size

    def step(self, action): # 0: local; 1: send 1/4; 2: send 1/2; 3: send 3/4; 4: send all

        self.steps += 1

        cur_model, type = self.model_clarify(self.task_type)
        start_time = time.time()
        if action == 0:
            self.run_predictions(self.images, cur_model)
        else:
            send_size = int(len(self.images) * action / 4)
            self.send(self.images[:send_size], type)
            if action != 4:
                self.run_predictions(self.images[send_size:], cur_model)
            self.receive()
        end_time = time.time()
        time_cost = end_time - start_time
        # get next task
        self.task_type = np.zeros(5)  # 0: detection, 1: segmentation, 2: pose estimation, 3: classification, 4: OBB
        type_index = random.choices([0, 1, 2, 3, 4], [0.25, 0.2, 0.2, 0.3, 0.05])[0]
        self.task_type[type_index] = 1  # Randomly select one task type
        self.batch_size = random.choices([4, 8, 16, 24, 32], [0.2, 0.2, 0.3, 0.2, 0.1])[0]  # Randomly select batch size
        if self.task_type[0] == 1 or self.task_type[1] == 1 or self.task_type[2] == 1:
            self.images = self.load_images(self.image_coco_files, self.batch_size)
        elif self.task_type[3] == 1:
            self.images = self.load_images(self.image_imagenet_files, self.batch_size)
        elif self.task_type[4] == 1:
            self.images = self.load_images(self.image_dota_files, self.batch_size)
        self.avg_image_size = [np.mean([img.shape[0] for img in self.images])/640., np.mean([img.shape[1] for img in self.images])/640.]

        self.remaining_energy = self.get_remaining_energy()/2.
        self.gpu_freq = self.set_gpu_freq(self.remaining_energy)/1e8

        state = np.concatenate((self.task_type, [self.batch_size], self.avg_image_size, [self.remaining_energy], [self.gpu_freq]))
        time_cost = min(10, time_cost)
        reward = -time_cost/2. + self.remaining_energy -1.0

        if self.remaining_energy <= 0 or self.steps >= 200:
            self.done = True
            self.train_gpu_freq()
        # rl step
        return state, reward, self.done, {"time_cost": time_cost, "remaining_energy": self.remaining_energy, "gpu_freq": self.gpu_freq}

    def send(self, images, type): 
        serialized_data = pickle.dumps(images)
        data_size = len(serialized_data)

        # Send fixed-length message (8 bytes)
        if type == "det":
            self.client_socket.sendall(b"det".ljust(8, b'\x00'))
        elif type == "seg":
            self.client_socket.sendall(b"seg".ljust(8, b'\x00'))
        elif type == "pose":
            self.client_socket.sendall(b"pose".ljust(8, b'\x00'))
        elif type == "cls":
            self.client_socket.sendall(b"cls".ljust(8, b'\x00'))
        elif type == "obb":
            self.client_socket.sendall(b"obb".ljust(8, b'\x00'))
        else:
            raise ValueError("Invalid task type")

        self.client_socket.sendall(data_size.to_bytes(8, 'big'))  
        self.client_socket.sendall(serialized_data)

    def receive(self):
        data_size = int.from_bytes(self.client_socket.recv(8), 'big')
        received_data = b""
        while len(received_data) < data_size:
            chunk = self.client_socket.recv(min(65536, max(4096, data_size - len(received_data))))
            if not chunk:
                break
            received_data += chunk
        # if len(received_data) == data_size:
        #     results = pickle.loads(received_data)

    def greedy_heuristic(self):
        # Greedy Decision Based on Energy and GPU State
        if self.remaining_energy > 0.9:
            action = 0
        elif self.remaining_energy > 0.8:
            action = 1
        elif self.remaining_energy > 0.55:
            action = 2
        elif self.remaining_energy > 0.3:
            action = 3
        else:
            action = 4
        
        return action
    

def main(args):
    env.train_gpu_freq()
    env.p = [0.2, 0.2, 0.3, 0.2, 0.1]

    returns = []
    for _ in range(args.n_trials):
        state = env.reset()
        done = False
        total = 0.0
        while not done:
            action = env.greedy_heuristic()
            state, reward, done, info = env.step(action)
            total += reward
        returns.append(total)
        print(f"  episode return: {total:.2f}, steps: {env.steps}")
    print(f"p={p}  mean={np.mean(returns):.2f}  std={np.std(returns):.2f}\n")

    env.stop_power_monitor()

if __name__ == '__main__':
    env = Env_Edge()
    env.train_gpu_freq()
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-file_path', type=str, default='greedy')
    parser.add_argument('-hidden', type=int, default=64)
    parser.add_argument('-learn_rate', type=float, default=1e-4)
    parser.add_argument('-update_interval', type=int, default=20000)
    parser.add_argument('-weight_decay', type=float, default=5e-5)
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-n_steps', type=int, default=20000)
    parser.add_argument('-n_trials', type=int, default=5)
    # CQL
    parser.add_argument('-n_quantiles', type=int, default=100)
    parser.add_argument('-alpha', type=float, default=1.0)
    # reward
    parser.add_argument('-dataset', type=str, default='edge_greedy')
    parser.add_argument('-tao', type=float, default=0.3)
    args = parser.parse_args()

    main(args)

    
    
