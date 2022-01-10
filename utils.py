import cv2 as cv

def nothing(x):
    return x

def initializeTrackbars(initialTrackbarVals=0):
    cv.namedWindow("Trackbars")
    cv.resizeWindow('Trackbars', 360, 240)
    cv.createTrackbar('Threshold1','Trackbars',200,255, nothing)
    cv.createTrackbar('Threshold2,', 'Trackbars', 200, 255, nothing)

def valTrackbars():
    Threshold1 = cv.getTrackbarPos('Threshold1', "Trackbars")
    Threshold2 = cv.getTrackbarPos('Threshold2', 'Trackbars')
    src = Threshold1,Threshold2
    return src