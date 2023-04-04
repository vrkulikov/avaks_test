import numpy as np
import cv2 as cv



objPoints = np.array([
    [[ 1, 2, 0],[ 2, 2, 0],[ 2, 1, 0],[ 1, 1, 0]],
    [[ 4, 8, 0],[ 8, 8, 0],[ 8, 4, 0],[ 4, 4, 0]],
    [[-2, 2, 0],[-1, 2, 0],[-1, 1, 0],[-2, 1, 0]],
    [[-8, 8, 0],[-4, 8, 0],[-4, 4, 0],[-8, 4, 0]],
    [[-2,-1, 0],[-1,-1, 0],[-1,-2, 0],[-2,-2, 0]],
    [[-8,-4, 0],[-4,-4, 0],[-4,-8, 0],[-8,-8, 0]],
    [[ 1,-1, 0],[ 2,-1, 0],[ 2,-2, 0],[ 1,-2, 0]],
    [[ 4,-4, 0],[ 8,-4, 0],[ 8,-8, 0],[ 4,-8, 0]],
    ], dtype=np.float32)

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)

ids = np.array([[5, 1, 6, 2, 7, 3, 8, 4]], dtype=np.int32)

board = cv.aruco.Board(objPoints*246/16, dictionary, ids)
