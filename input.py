import numpy as np
import cv2 as cv
import glob
import pickle

def ImageInput():

    ######### Chess board size ############
    chessboardSize = (24,17)
    #Set the frame size of the image
    frameSize = (1440,1080)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    images = glob.glob(r'C:\Users\Asus\Desktop\camera calibration and depth estimation using opencv python\Reference\ComputerVision\cameraCalibration\calibration\*.png')


    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for image in images:

        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        # If found Chessboard, add object points, image points (after refining them)
        if ret == True:

            objpoints.append(objp)
            #default paramaters
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(1000)
    cv.destroyAllWindows()

    return [objpoints ,imgpoints , frameSize]


def Calibrate():

    data = ImageInput()
    objpoints = data[0]
    imgpoints = data[1]
    frameSize = data[2]

    ############## CALIBRATION #######################################################
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    # Save the camera calibration results.
    calib_result_pickle = {}
    calib_result_pickle['cameraMatrix'] = cameraMatrix
    calib_result_pickle['dist'] = dist
    calib_result_pickle['rvecs'] = rvecs
    calib_result_pickle['tvecs'] = tvecs
    calib_result_pickle['objpoints'] = objpoints
    calib_result_pickle['imgpoints'] = imgpoints

    pickle.dump(calib_result_pickle, open("camera_calib_pickle.p", "wb" )) 

if __name__=='__main__':

    Calibrate()






