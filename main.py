from __future__ import print_function
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv.VideoCapture(0)

if not capture.isOpened():
    print('Unable to open video source')
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv.imshow('Frame', frame)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break