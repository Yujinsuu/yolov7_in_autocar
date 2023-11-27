import os
import cv2
import time
import torch
import numpy as np
from cv_bridge import CvBridge

from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

dataset = "traffic"
path_ = ["train","valid","test"]
classname = {'part':['A1','A2','A3','B1','B2','B3'], 'whole':['A1','A2','A3','B1','B2','B3'],
             'traffic':['green','left','red','stleft','yellow']}
WEIGHTS = 'weights/trafficlight.pt'
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.70
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False

# Initialize
device = select_device(DEVICE)
half = device.type != 'cpu'  # half precision only supported on CUDA
print('device:', device)

# Load model
model = attempt_load(WEIGHTS, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(IMG_SIZE, s=stride)  # check img_size
if half:
    model.half()  # to FP16

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


# Detect function
def detect(frame):
    ids = []
    # Load image
    img0 = frame

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t0 = time_synchronized()
    pred = model(img, augment=AUGMENT)[0]

    # Apply NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

    # Process detections
    det = pred[0]

    s = ''
    s += '%gx%g ' % img.shape[2:]  # print string

    if len(det):
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        for *xyxy, conf, cls in reversed(det):
            ids.append(int(cls))
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

    # return results
    return ids


# main
check_file = ""
check_requirements(exclude=('pycocotools', 'thop'))
with torch.no_grad():
    os.chdir(os.path.join('dataset', dataset))
    for dir in path_:
        os.chdir(os.path.join(dir, "labels"))
        for file_ in os.listdir(os.getcwd()):
            if file_.endswith(".txt") and file_.split("_")[0] != 'background':
                with open(file_) as myfile:
                    total_lines = sum(1 for line in myfile)
                f = open(file_)
                lines = f.readlines(0)
                label = []
                for i in range(total_lines):
                    id = lines[i].split(" ")[0]
                    label.append(int(id))

                basename=os.path.basename(file_)
                filename=os.path.splitext(basename)[0]
                os.chdir("..")
                os.chdir("..")
                os.chdir("..")
                img_file = dataset + '/' + dir + '/images/' + filename + '.jpg'
                txt_file = dataset + '/' + dir + '/labels/' + filename + '.txt'
                img = cv2.imread(img_file)
                detected = detect(img)
                
                if not set(label).issubset(set(detected)):
                    print('Not Match : ' + filename +'\nLabel = ' + str(label) + '\tDetected = ' + str(detected))
                    check_file += filename + '\n'
                    
                    img_height, img_width = img.shape[:2]

                    box_size = []
                    for _ in range(total_lines):
                        for line in lines:
                            id,x,y,width,height = map(float,line.split())
                            class_id = int(id)
                            name = classname[dataset][class_id]
                            
                            x_min = int((x - (width / 2))  * img_width)
                            y_min = int((y - (height / 2)) * img_height)
                            x_max = int((x + (width / 2))  * img_width)
                            y_max = int((y + (height / 2)) * img_height)
                            box_size.append((x_max-x_min)*(y_max-y_min))
                            if (x_max-x_min) < (y_max-y_min) or (x_max-x_min) < 10: box_size.append(1)
                            cv2.rectangle(img, (x_min,y_min),(x_max,y_max),(255,255,255),2,lineType=cv2.LINE_AA)
                            t_size = cv2.getTextSize(name,0,1,2)[0]
                            cv2.putText(img,name,(x_min,y_max+20 if y_max < 320 else y_min-20),0,1,[0,0,255],1,cv2.LINE_AA)

                    if any(box < 30 for box in box_size):
                        os.remove(img_file)
                        os.remove(txt_file)
                    else:
                        start = time.time()
                        now = time.time()
                        while(now-start < 3):
                            now = time.time()
                            cv2.imshow('Not Match', img)
                            cv2.waitKey(1)
                        cv2.destroyAllWindows()
                        cut = bool(input("Do you want to remove?"))
                        if cut:
                            print("Success to remove")
                            os.remove(img_file)
                            os.remove(txt_file)

                os.chdir(os.path.join(dataset, dir, "labels"))
        os.chdir("..")
        os.chdir("..")

with open('need_to_check.txt','w') as t:
    t.writelines(check_file)
