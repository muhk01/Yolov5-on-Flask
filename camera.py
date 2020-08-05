import os
import cv2
from base_camera import BaseCamera
import torch
import torch.nn as nn
import torchvision
import numpy as np
import argparse
import paho.mqtt.client as mqtt
import json
from utils.datasets import *
from utils.utils import *

MQTT_TOPIC = "v1/devices/me/telemetry"
mylist = []
mycount = []
broker_url = "demo.thingsboard.io"
broker_port = 1883

username = "aTbDLqllhSQAyOPeE6rx"
password = '' 

client = mqtt.Client()
client.username_pw_set(username)
client.connect(broker_url, broker_port)
client.loop_start()


class Camera(BaseCamera):
    video_source = '4.mp4'

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        out, weights, imgsz = \
        'inference/output', 'weights/yolov5s.pt', 640
        source = '4.mp4'
        device = torch_utils.select_device()
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

        # Load model
        google_utils.attempt_download(weights)
        model = torch.load(weights, map_location=device)['model']
        
        model.to(device).eval()

        # Second-stage classifier
        classify = False
        if classify:
            modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Half precision
        half = False and device.type != 'cpu' 
        print('half = ' + str(half))

        if half:
            model.half()

        # Set Dataloader
        vid_path, vid_writer = None, None
        dataset = LoadImages(source, img_size=imgsz)
        #dataset = LoadStreams(source, img_size=imgsz)
        names = model.names if hasattr(model, 'names') else model.modules.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        classSend = []
        countSend = []
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=False)[0]
            
            # Apply NMS
            pred = non_max_suppression(pred, 0.4, 0.5,
                               fast=True, classes=None, agnostic=False)
            t2 = torch_utils.time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %s, ' % (n, names[int(c)])  # add to string
                        listDet = ['person','bicycle','car','motorbike','bus','truck','bird','cat','dog','horse','cow','backpack','umbrella','handbag','kite','cell phone']
                        
                        if(str(names[int(c)]) in listDet): #filter certain object want to detect
                            countSend.append('%s' % (names[int(c)]))
                            classSend.append('%g' % (n))

                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                data_set = {"Object": classSend, "Count": countSend}    #publish to mqtt both object and counts
                MQTT_MSG = json.dumps(data_set)
                client.publish(MQTT_TOPIC, MQTT_MSG)
                print('%sDone. (%.3fs)' % (s, t2 - t1))
                del classSend[:]
                del countSend[:]
 
            yield cv2.imencode('.jpg', im0)[1].tobytes()

    
            
