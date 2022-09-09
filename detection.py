import socket
import time

import cv2
import numpy as np
from elements.yolo import OBJ_DETECTION

HOST = '0.0.0.0'  # Standard loopback interface address (localhost)
PORT = 7777  # Port to listen on (non-privileged ports are > 1023)

def warp_coords(xy: np.float32, matrix) -> np.float32:
        return cv2.warpPerspective(xy, matrix, (800, 600), flags=cv2.INTER_LINEAR)

def get_center_coords(obj) -> list:
        [(xmin, ymin), (xmax, ymax)] = obj['bbox']
        return np.float32([(xmax - xmin) / 2, (ymax - ymin) / 2])

def push_coords(coords: np.float32) -> None:
        print(coords)
        #with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
             #s.bind((HOST, PORT))
             #s.listen(1)
             #conn, addr = s.accept()
             #while True:
                 #conn.sendall(coords.tobytes())
                 #time.sleep(1)

VIDEO_SIZE = (1280, 720)

Object_classes = ['cig_butt']
#Object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush' ]

Object_colors = list(np.random.rand(80,3)*255)
Object_detector = OBJ_DETECTION('weights/best.pt', Object_classes)
#Object_detector = OBJ_DETECTION('weights/yolov5s.pt', Object_classes)

# get perspective transformation (calibrate camera)
input_pts = np.float32([[0, 0], [800, 800], [0, 800], [800, 0]])
output_pts = np.float32([[0, 0], [800, 800], [0, 800], [800, 0]])
warp_matrix = cv2.getPerspectiveTransform(input_pts, output_pts)

cap = cv2.VideoCapture(0)
if cap.isOpened():
        while True:
                ret, frame = cap.read()
                frame = cv2.resize(frame, VIDEO_SIZE, fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
                #frame = cv2.warpPerspective(frame, warp_matrix, VIDEO_SIZE, flags=cv2.INTER_LINEAR)
                if ret:
                        objs = Object_detector.detect(frame)
                        for obj in objs:
                                center_coords = get_center_coords(obj)
                                #warped_coords = warp_coords(center_coords, warp_matrix)
                                push_coords(center_coords)
                                #push_coords(warped_coords)
                        else:
                                None
        cap.release()
else:
        print("Unable to open camera")
