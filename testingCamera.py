import cv2 as cv
import numpy as np

heightImg = 640
widthImg  = 480

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv.contourArea(i)
        if area > 5000:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def drawRectangle(img,biggest,thickness):
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv.putText(img, 'Objek ditemukan!', (biggest[0][0][0], (biggest[0][0][1])-10), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    return img


def recorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

def contourRecord():
    cam = cv.VideoCapture(0)

    while True:
        check, frame = cam.read()
        imgGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gaussianFilter = cv.GaussianBlur(imgGray, (3,3), 10)
        unsharpMasking = cv.addWeighted(imgGray, 1+1.5, gaussianFilter, -1.5, 0)
        ret, thresh = cv.threshold(unsharpMasking, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        #cv.drawContours(frame, contours, -1, (0, 255, 0), 10)

        ## FIND THE BIGGEST CONTOUR
        biggest, maxArea = biggestContour(contours)
        print(biggest)
        if biggest.size != 0 :
            biggest = recorder(biggest)
            cv.drawContours(frame, biggest, -1, (0, 255, 0), 10)
            result = drawRectangle(frame, biggest, 5)
            try:
                cv.imshow("Hasil", result)
            except Exception as e:
                print(str(e))
        else:
            cv.imshow("Hasil", frame)





        key = cv.waitKey(1)
        if key % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            cam.release()
            break

def contourImage(img):
    scale_percent = 100

    img = cv.imread(img)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width,height)
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gaussianFilter = cv.GaussianBlur(imgGray, (11, 11), 10)
    unsharpMasking = cv.addWeighted(imgGray, 1 + 1.5, gaussianFilter, -0.5, 0)
    ret, thresh = cv.threshold(unsharpMasking, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img = cv.drawContours(img, contours, -1, (0, 255, 0), 10)

    biggest, maxArea = biggestContour(contours)
    print(biggest)
    if biggest.size != 0:
        biggest = recorder(biggest)
        cv.drawContours(img, biggest, -1, (0, 255, 0), 20)
        img = drawRectangle(img, biggest, 2)

    cv.imshow("Hasil",img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    path = 'picts/raspi.jpg'
    contourRecord()
    #contourImage(path)