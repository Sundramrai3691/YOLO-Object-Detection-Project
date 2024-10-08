from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import * 

cap = cv2.VideoCapture('../Videos/people.mp4')

model = YOLO('../YOLO-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread('mask-1.png.')
mask = cv2.resize(mask,(1280,720))

#Tracking
tracker = Sort(max_age = 20, min_hits=3, iou_threshold= 0.3 )


limitsUp = [103,161,296,161]
limitsDown = [527,489,735,489]

totalCountUp = []
totalCountDown = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)

    imgGraphics = cv2.imread('graphics-1.png',cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,imgGraphics,(730,260))
    results = model(imgRegion, stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w,h = x2-x1, y2-y1

            #Confidence
            conf = math.ceil((box.conf[0]*100))/100 # to hget value upto 2 decimal places
            print(conf)

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == 'person' and conf>0.3:

                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))
    resultsTracker = tracker.update(detections)
    cv2.line(img,(limitsUp[0],limitsUp[1]), (limitsUp[2],limitsUp[3]),(0,0,255),5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1,y1,x2,y2,Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt = 2, colorR=(255,0,0))
        cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1+w //2, y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),-1)

        if limitsUp[0]<cx<limitsUp[2] and limitsUp[1]-15<cy<limitsUp[1]+15:
            if totalCountUp.count(Id) ==0: # it will count the no of times Id is present
                totalCountUp.append(Id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if limitsDown[0]<cx<limitsDown[2] and limitsDown[1]-15<cy<limitsDown[1]+15:
            if totalCountDown.count(Id) ==0: # it will count the no of times Id is present
                totalCountDown.append(Id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)


    cv2.putText(img,str(len(totalCountUp)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
    cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)

    cv2.imshow('Image', img)

    cv2.waitKey(1)
