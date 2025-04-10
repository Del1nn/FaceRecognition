import cv2
import os

base_path = os.path.dirname(__file__)

config_file = os.path.join(base_path, 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt') 
frozen_model = os.path.join(base_path, 'frozen_inference_graph.pb') 

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
filename = os.path.join(base_path, 'labels.txt')
with open(filename, 'rt') as spt:
    classLabels = spt.read().rstrip('\n').split('\n')
    
model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    classIds, confidences, boxes = model.detect(frame, confThreshold=0.5)

    person_count = 0
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confidences.flatten(), boxes):
            if len(classLabels) > classId:
                label = classLabels[classId - 1]
                if label == "person":
                    person_count += 1
                    (startX, startY, width, height) = box
                    endX = startX + width
                    endY = startY + height

                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label_text = f"{label}: {confidence:.2f}"
                    cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
    cv2.putText(frame, f"People Count: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    

    cv2.imshow("People Detection & Count", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
