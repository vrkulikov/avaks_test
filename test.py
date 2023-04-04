import cv2 as cv
import numpy as np
from geometry import rotationMatrixToEulerAngles

from position_detector import PositionDetector
from board import board



mtx = np.load('mtx.npy')
dist = np.load('dist.npy')

detector = PositionDetector(mtx, dist, board)

cap = cv.VideoCapture('DJI_0015.MP4')
key = 0
norms,rvecs,tvecs = [],[],[]
DEPTH = 6
while key != 27:
    key = cv.waitKey(1)
    ret, frame = cap.read()
    if ret:
        frame, rvec, tvec = detector(frame)
        if rvec is not None:
            norm = np.linalg.norm(tvec)
            norms.append(norm)
            rvecs.append(rvec)
            tvecs.append(tvec)
            id = np.argsort(norms[-DEPTH:])[len(norms[-DEPTH:])//2]
            norm = norms[-DEPTH:][id]
            rvec = rvecs[-DEPTH:][id]
            tvec = tvecs[-DEPTH:][id]
            euler = rotationMatrixToEulerAngles(cv.Rodrigues(rvec)[0])
            frame = cv.putText(frame, f'Distance: {norm:.3f}', (0,1070), cv.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0),2)
            frame = cv.putText(frame, f'Euler: {euler[0]:.3f} {euler[1]:.3f} {euler[2]:.3f}', (0,990), cv.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0),2)
            frame = cv.putText(frame, f'Position: {tvec[0][0]:.3f} {tvec[1][0]:.3f} {tvec[2][0]:.3f}', (0,1030), cv.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0),2)
        # frame = cv.resize(frame, (1920//4,1080//4))
        cv.imshow('win', frame)
        # Пауза при нажатии пробела
        if key == 32: 
            while cv.waitKey(1) != 32:
                pass
    else:
        break
cv.destroyAllWindows()

