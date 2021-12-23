import cv2
import numpy as np


cap = cv2.VideoCapture('39.mp4')
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

classFile = 'coco.names'
classNames = []
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, cT = img.shape
    #hT = img.shape[0]
    #wT = img.shape[1]
    bbox = []
    class_Ids = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_Id = np.argmax(scores)
            confidence = scores[class_Id]
            if confidence > confThreshold:
                w = int(det[2]*wT)
                h = int(det[3]*hT)
                x = int((det[0]*wT)-w/2)
                y = int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                class_Ids.append(class_Id)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    print(indices)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[class_Ids[0]].upper()} {int(confs[i]*100)}%',
                    (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),2)


while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img,1/255,(whT, whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [(layerNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs,img)

    cv2.imshow('CAMERA', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
