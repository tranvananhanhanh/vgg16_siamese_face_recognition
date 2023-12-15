import cv2
import os
import time

output_dir = '/Users/jmac/Desktop/siamese/data/anchor'
num_images = 300
wait_time = 0.5
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Không thể mở webcam.")

for i in range(num_images):
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow('Webcam', frame)
    image_path = os.path.join(output_dir, f'image_{i}.jpg')
    cv2.imwrite(image_path, frame)
    time.sleep(wait_time)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()