import cv2

import numpy as np
import time
from pushbullet import Pushbullet

#fil = 'access.txt'
API_KEY = 'o.Co4fBx5Zg5MqjzrQXdlwbMOr3Q4vZijI'


        
# Load the YOLO model
net = cv2.dnn.readNet("./weights/yolov3_training_2000.weights", "./configuration/yolov3_testing.cfg")


classes = []
with open("./configuration/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# open the video footage
cap = cv2.VideoCapture("1.mp4")
cap.set(cv2.CAP_PROP_FPS, 60)


font = cv2.FONT_HERSHEY_SIMPLEX
starting_time = time.time()
frame_id = 0
count=0 

while True:
    # Read video
    _, frame = cap.read()
    frame_id += 1
    height, width, channels = frame.shape

    # Detecting man
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Visualising data
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
                
         
                
            if confidence > 0.1 and class_id==0:
                # knife detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                #print("intruder detected")
            
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    people=0
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    now=time.time()
    curr=time.ctime(now)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
            cv2.putText(frame, str(classes[class_ids[i]]) + " " + str(round(confidence, 2)), (x, y + 30), font, 1.5,(0,0,255), 3)
            push  = pb.push_note("intruder alert",str(classes[class_ids[i]])+ "  detected at" +str(curr))
            #count=count+1
            #people=people+1
    #print(people)        

    
    cv2.putText(frame, "FPS: " + str(round(fps, 1)), (40, 670), font, .3, (0, 255, 255), 1)
    cv2.putText(frame, "press [esc] to exit", (40, 690), font, .45, (0, 0, 255), 1)
    cv2.imshow("Surveillence system", frame)
    
    

   # if count%100 == 0 and people >=1 :
       # push  = pb.push_note('test','new')
        #email_alert("Alert",str(people)+" Intruder detected at :"+ curr,"keerthankirthan969@gmail.com")
        #print("alert"+str(count))
    
    key = cv2.waitKey(1)
    if key == 27:
        print("[button pressed] ///// [esc].")
        print("[feedback] ///// Videocapturing succesfully stopped")
        break

cap.release()
cv2.destroyAllWindows()