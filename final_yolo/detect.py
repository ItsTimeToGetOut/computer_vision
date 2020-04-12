import os
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
import struct
import cv2
import sys
from support import make_yolov3_model, WeightReader, preprocess_input, decode_netout, correct_yolo_boxes, do_nms, draw_boxes

np.set_printoptions(threshold=sys.maxsize)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def _main_():
    net_h, net_w = 416, 416
    obj_thresh, nms_thresh = 0.5, 0.45
    anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    yolov3 = make_yolov3_model()
    weight_reader = WeightReader('yolov3.weights')
    weight_reader.load_weights(yolov3)
    # image_path='lo.jpg'
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    while True:
        _, frame = cap.read()
        # frame = cv2.resize(frame, (416, 416))
        image=frame

        image_h, image_w, _ = image.shape
        new_image = preprocess_input(image, net_h, net_w)

        yolos = yolov3.predict(new_image)
        boxes = []

        for i in range(len(yolos)):
            boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
        do_nms(boxes, nms_thresh)     
        draw_boxes(image, boxes, labels, obj_thresh)

        cv2.imshow('Input', image)
        c = cv2.waitKey(1)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    # image = cv2.imread(image_path)
    # image_h, image_w, _ = image.shape
    # new_image = preprocess_input(image, net_h, net_w)

    # yolos = yolov3.predict(new_image)
    # boxes = []

    # for i in range(len(yolos)):
    #     boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)
    # correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
    # do_nms(boxes, nms_thresh)     
    # draw_boxes(image, boxes, labels, obj_thresh) 
    # cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], (image).astype('uint8')) 

if __name__ == '__main__':
    _main_()