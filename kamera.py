

import cv2

cap=cv2.VideoCapture("rtmp://213.226.117.171/hypegenai/CAM1")
while True:
    _, frame = cap.read()
    cv2.imshow("frame",frame)
    cv2.waitKey(1)


    # rtsp://:@:/cam/realmonitor?channel=&subtype=