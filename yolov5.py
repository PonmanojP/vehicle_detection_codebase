import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_webcam.mp4', fourcc, 20.0, 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

target_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    

    results = model(frame)
    detections = results.pandas().xyxy[0] 

    for _, row in detections.iterrows():
        if row['name'] in target_classes:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            conf = row['confidence']
            cls = row['name']
            color = (0, 255, 0) if cls == 'person' else (255, 0, 0)  # Green for person, Blue for vehicles
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Webcam Detection', frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
