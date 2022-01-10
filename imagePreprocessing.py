import cv2 as cv
import utils

#print(cv.__version__)

def nothing(x):
    return x

def imagePreprocessing(path, thres1, thres2):
    img = cv.imread(path)
    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey,(3,3),cv.BORDER_DEFAULT)
    sharp = cv.addWeighted(grey, 1+1.5, blur, -1.5, 0)
    #ret, thresh = cv.threshold(sharp, thres1, thres2, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return sharp

if __name__ == "__main__":
    path = 'picts/raspi.jpg'
    img = cv.imread(path)

    cv.namedWindow('Result')

    cv.createTrackbar("Thres1", "Result", 0, 255, nothing)
    cv.createTrackbar("Thres2", "Result", 0, 255, nothing)

    #thres = utils.initializeTrackbars(0)
    while True:
        cv.imshow('Result', img)

        thres1 = cv.getTrackbarPos("Thres1", "Result")
        thres2 = cv.getTrackbarPos("Thres2", "Result")

        img = imagePreprocessing(path, thres1, thres2)


        k = cv.waitKey(1)
        if k % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break

    cv.destroyAllWindows()