import socket
import time
import os
import torch
import numpy as np
import pickle
import random
from PIL import Image
from ultralytics import YOLO

# Server settings
PORT = 5000

class Server:
    def __init__(self):
        print("Server is initializing ... ...")
        self.model = YOLO("./models/yolo11n.pt").to('cuda')
        self.model_seg = YOLO("./models/yolo11n-seg.pt").to('cuda')
        self.model_pose = YOLO("./models/yolo11n-pose.pt").to('cuda')
        self.model_cls = YOLO("./models/yolo11n-cls.pt").to('cuda')
        self.model_obb = YOLO("./models/yolo11n-obb.pt").to('cuda')

        self.initialize_model()
        self.initialize_socket()
        
    def initialize_socket(self):     
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('0.0.0.0', PORT))
    
    def initialize_model(self):
        image_coco_dir = "/home/zu/Documents/yolov11/datasets/coco/"
        image_coco_files = [os.path.join(image_coco_dir, f) for f in os.listdir(image_coco_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        image_dota_dir = "/home/zu/Documents/yolov11/datasets/dota/"
        image_dota_files = [os.path.join(image_dota_dir, f) for f in os.listdir(image_dota_dir) if f.lower().endswith(('.jpg', '.png'))]
        image_imagenet_dir = "/home/zu/Documents/yolov11/datasets/imagenet/"
        image_imagenet_files = [os.path.join(image_imagenet_dir, f) for f in os.listdir(image_imagenet_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        imgs = self.load_images(image_coco_files, 2)
        imgs_seg = self.load_images(image_coco_files, 2)
        imgs_pose = self.load_images(image_coco_files, 2)
        imgs_cls = self.load_images(image_imagenet_files, 2)
        imgs_obb = self.load_images(image_dota_files, 2)

        # Run predictions using reusable function
        _ = self.run_predictions(imgs, self.model, initialized=True) 
        _ = self.run_predictions(imgs_seg, self.model_seg, initialized=True)  
        _ = self.run_predictions(imgs_pose, self.model_pose, initialized=True)  
        _ = self.run_predictions(imgs_cls, self.model_cls, initialized=True)  
        _ = self.run_predictions(imgs_obb, self.model_obb, initialized=True)  

    def clarify_model(self, name):
        if name =="det":
            return self.model
        elif name == "seg":
            return self.model_seg
        elif name == "pose":
            return self.model_pose
        elif name == "cls":
            return self.model_cls
        elif name == "obb":
            return self.model_obb
        else:
            raise ValueError(f"Unknown model type: {name}. Please choose from 'detection', 'segmentation', 'pose', 'classification', or 'obb'.")

    def load_images(self, file_list, num_images):
        sampled_files = random.sample(file_list, num_images)
        return [np.array(Image.open(img_path)) for img_path in sampled_files]

    def run_predictions(self, images, model, initialized=False):
        results = []
        if not initialized:
            for image in images:
                if len(image.shape) < 3:
                    continue
                result = model.predict(image, device='cuda')              
                results.append(result[0].plot())

            return results
        else:
            for _ in range(5): 
                for image in images:
                    result = model.predict(image, device='cuda')
            return results

    def receive(self):
        self.server_socket.listen(1)
        print(f"Server is listening on port {PORT} ... ...")
        conn, addr = self.server_socket.accept()
        print(f"Connected by {addr}")

        while True:
            # Receive exactly 8 bytes (consistent size)
            data = conn.recv(8)
            model_name = data.strip(b'\x00').decode('utf-8')
            cur_model = self.clarify_model(model_name)

            data_size = int.from_bytes(conn.recv(8), 'big')
            received_data = b""
            while len(received_data) < data_size:
                chunk_size = min(65536, max(4096, data_size // 16))
                chunk = conn.recv(min(chunk_size, data_size - len(received_data)))

                if not chunk:
                    break
                received_data += chunk
            
            imgs = pickle.loads(received_data)
            server_results = self.run_predictions(imgs, cur_model)
            self.send_back(server_results, conn)

    def send_back(self, results, conn):
        serialized_results = pickle.dumps(results)
        total_size = len(serialized_results)
        conn.sendall(total_size.to_bytes(8, 'big'))
        conn.sendall(serialized_results)

if __name__ == '__main__':
    server = Server()
    server.receive()
