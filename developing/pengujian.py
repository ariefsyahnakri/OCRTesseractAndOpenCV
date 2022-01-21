import cv2 as cv
import numpy as np
import pytesseract
from datetime import date, datetime


#tesseractFile = "C:\Program Files\Tesseract-OCR\Tesseract.exe"
#pytesseract.pytesseract.tesseract_cmd = tesseractFile

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

def drawRectangle(img,biggest,thickness):
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv.putText(img, 'Objek ditemukan!', (biggest[0][0][0], (biggest[0][0][1])-10), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    return img

def imgToText(img):
    now = datetime.now()
    imgFileRaw = now.strftime("Gambar Raw %d/%m/%Y %H:%M:%S").replace(":", "-") + ".jpg"
    imgFileRes = now.strftime("Gambar Res %d/%m/%Y %H:%M:%S").replace(":", "-") + ".jpg"
    txtFile = now.strftime("Text %d/%m/%Y %H:%M:%S").replace(":", "-") + ".txt"
    cv.imwrite(imgFileRaw,img)
    print("{} written".format(imgFileRaw))

    # Reading image
    img = cv.imread(imgFileRaw)

    # Color2Gray process
    imgGrey = cv.cvtColor(img,cv.COLOR_RGB2GRAY)

    # Unsharp masking process
    gaussianFilter = cv.GaussianBlur(imgGrey, (11,11), 10)
    unsharpMasking = cv.addWeighted(imgGrey, 1+1.5, gaussianFilter, -0.5, 0)

    # Otsu thresholding process
    ret, thresh = cv.threshold(unsharpMasking, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    #Tesseract process
    textOCR = pytesseract.image_to_string(thresh, lang='eng')


    ## Drawing rectangle on picture for analysis
    height, width = img.shape
    boxes = pytesseract.image_to_data(img)
    for x, b in enumerate(boxes.splitlines()):
        if x != 0:
            b = b.split()
            print(b)
            if len(b) == 12:
                x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                cv.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 3)
                cv.putText(img, b[11], (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

    cv.namedWindow('Hasil')
    cv.imshow('Hasil',thresh)
    cv.waitKey(0)
    cv.destroyAllWindows


    cv.imwrite(imgFileRes,thresh)
    print("{} written".format(imgFileRes))

    with open(txtFile, 'w+') as f:
        f.write(textOCR)
        print("{} written".format(txtFile))


def main():
    cam = cv.VideoCapture(0)
    cv.namedWindow('Test')

    while True:
        red, img = cam.read()
        img = cv.transpose(img)
        img = cv.flip(img, flipCode=0)
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gaussianFilter = cv.GaussianBlur(imgGray, (3, 3), 10)
        unsharpMasking = cv.addWeighted(imgGray, 1 + 1.5, gaussianFilter, -1.5, 0)
        ret, thresh = cv.threshold(unsharpMasking, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        #cv.drawContours(rgb, contours, -1, (0, 255, 0), 10)

        w, h = 480, 640
        # Finding biggest contour
        biggest, maxArea = biggestContour(contours)
        if biggest.size != 0:
            biggest = recorder(biggest)
            cv.drawContours(rgb, biggest, -1, (0, 255, 0), 10)
            rgb = drawRectangle(rgb, biggest, 5)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            matrix = cv.getPerspectiveTransform(pts1, pts2)
            wrapped = cv.warpPerspective(rgb, matrix, (w, h))
            wrapped = wrapped[20:wrapped.shape[0] - 20, 20:wrapped.shape[1] - 20]
        else:
            rgb = rgb.copy()


        cv.imshow('Test',rgb)

        k = cv.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            imgToText(wrapped)


if __name__ == "__main__":
    main()

