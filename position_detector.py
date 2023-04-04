import cv2 as cv
import numpy as np

class PositionDetector:
    def __init__(self, mtx, dist, board):
        self.mtx = mtx
        self.dist = dist
        self.board = board
        self.dictionary = board.getDictionary()
        self.pars = cv.aruco.DetectorParameters()
        self.pars.cornerRefinementMethod = 1
        self.aruco_detector = cv.aruco.ArucoDetector(board.getDictionary(), self.pars )

    def __call__(self, frame):
        corners, detected_ids, _ = self.aruco_detector.detectMarkers(cv.cvtColor(frame,cv.COLOR_BGR2GRAY))

        if detected_ids is not None:
            obj_points, img_points = self.board.matchImagePoints(corners, detected_ids)
            if obj_points is not None:
                ret, rvec, tvec = cv.solvePnP(obj_points, img_points, self.mtx, self.dist)
                frame = cv.aruco.drawDetectedMarkers(frame, corners, detected_ids)
                frame = cv.drawFrameAxes(frame, self.mtx, self.dist, rvec, tvec, 100)
                return frame, rvec, tvec
        return frame, None, None
    
