import cv2 as cv
import numpy as np
import pickle


def final():

    ############## UNDISTORTION #####################################################

    #Live webcam 
    cap = cv.VideoCapture(0)

    # Setting the frame width and height
    #cap.set(cv.CAP_PROP_FRAME_WIDTH, 1440)
    #cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    while(cap.isOpened()):

        ret, frame = cap.read()

        h,  w = frame.shape[:2]

        #Getting details from pickle file
        calib_result_pickle = pickle.load(open("camera_calib_pickle.p", "rb" ))

        cameraMatrix = calib_result_pickle['cameraMatrix']

        dist = calib_result_pickle['dist']

        rvecs = calib_result_pickle['rvecs']

        tvecs = calib_result_pickle['tvecs']

        objpoints = calib_result_pickle['objpoints']

        imgpoints = calib_result_pickle['imgpoints'] 

        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

        # Undistort
        dst = cv.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)

        # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]
        # cv.imwrite('caliResult1.png', dst)

        # Undistort with Remapping
        # mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
        # dst = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)

        # Reprojection Error
        mean_error = 0

        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error

        print( "total error: {}".format(mean_error/len(objpoints)) )

        
        cv.imshow('frame', dst) 
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':

    final()

