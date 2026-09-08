# Task Offloading Testbed (Edge Computing Scenario)

Client–server implementation of the task offloading use case. The client runs
YOLO inference on batches of images and decides at each step what fraction of
the batch to offload to the server, trading off latency against remaining
battery.

Actions: `0` = full local, `1`/`2`/`3` = offload 25%/50%/75%, `4` = full offload.

## Hardware

The client reads its power sensor and sets its GPU frequency through
device-specific sysfs paths, so **this code depends on the device it runs on**.
We use an NVIDIA Jetson Orin Nano as the client and a desktop GPU as the server;
on other hardware the paths and frequency values must be changed.

## Requirements

```
python >= 3.8
torch, ultralytics, numpy, pillow
```

### Models

Download the five YOLO11n weights from
[Ultralytics](https://docs.ultralytics.com/models/yolo11/) and place them under
`./models/` on **both** machines:

```
yolo11n.pt        # detection
yolo11n-seg.pt    # segmentation
yolo11n-pose.pt   # pose estimation
yolo11n-cls.pt    # classification
yolo11n-obb.pt    # oriented bounding box
```

### Datasets

Download the images used to build task batches and place them in three folders
on **both** machines:

| dataset | used by | source |
|---|---|---|
| COCO | detection, segmentation, pose | https://cocodataset.org/#download |
| ImageNet | classification | https://www.image-net.org/download.php |
| DOTA | oriented bounding box | https://captain-whu.github.io/DOTA/dataset.html |


## Configuration

Edit before running:

- `SERVER_IP`, `PORT` — must match on both sides
- dataset directories — in `Env_Edge.__init__` and `Server.initialize_model`
- `VOLTAGE_PATH`, `CURRENT_PATH` — INA3221 hwmon nodes (`grep -r . /sys/class/hwmon/hwmon*/name`)
- GPU devfreq path and frequency values — in `set_gpu_freq` / `train_gpu_freq` (`ls /sys/class/devfreq/`)
- `password` — used for `sudo -S`; better to grant passwordless access to those
  sysfs nodes via `/etc/sudoers.d/` and leave it empty

## Running

Start the server first.

```bash
python server.py          # on the server
python client.py -n_trials 5   # on the client
```
