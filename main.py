import cv2

thres = 0.45 # Threshold to detect object

#name of the classes from coco names
classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#pre trained model used and configuring their path for detection
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# function to check mark the detected object into box and provide their cordinates and names as output
def getObjects(img,objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=0.2)
    #print(classIds,bbox)
    objectInfo=[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId-1]
            if className in objects:
                objectInfo.append([box,className])
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                cv2.putText(img,className.upper(),(box[0]+10,box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    return img,objectInfo


#to call the function to detect the object
if __name__=="__main__":
    #video resolution is set to 640 * 480
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #  cap.set(10,70)

    while True:
        success,img = cap.read()
        #main intruder in farms cow, sheep and horse and so on
        result,objectInfo = getObjects(img,objects=['cow','sheep','horse','elephant'])
        #printing detected object info
        print(objectInfo)
        cv2.imshow("Output",img)
        cv2.waitKey(1)