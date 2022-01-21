import cv2
import numpy as np
import pytesseract

#tesseractFile = "C:\Program Files\Tesseract-OCR\Tesseract.exe"
#pytesseract.pytesseract.tesseract_cmd = tesseractFile

def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d][c]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, lables[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver


def recorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def drawRectangle(img, biggest, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    return img


def image():
    heightImg = 640
    widthImg = 480
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)


    path = "/home/pi/Documents/arief/github/OCRTesseractAndOpenCV/developing/ARIAL/UnwrappedDocument0.jpg"
    img = cv2.imread(path)
    img = cv2.resize(img,(widthImg,heightImg))

    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgBlur = cv2.GaussianBlur(imgGrey,(21,21),10) ## Gaussian Filter
    imgSharp = cv2.addWeighted(imgGrey, 2.0, imgBlur, -1.0, 0)

    ret, imgThres = cv2.threshold(imgSharp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    ## FIND CONTOURS


    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgThres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    biggest, maxArea = biggestContour(contours)
    print(biggest)
    if biggest.size != 0:
        biggest = recorder(biggest)
        cv2.drawContours(imgBigContour,biggest,-1, (0,255,0),20)
        imgBigContour = drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        imgWarpColored = cv2.warpPerspective(imgThres, matrix,(widthImg,heightImg))
        

        imgWDilated = cv2.dilate(imgWarpColored,None,iterations=2)




    imageArray = ([img, imgGrey,imgBlur,imgSharp],[imgThres,imgBigContour,imgWarpColored,imgWDilated])

    lables = [["Original", "Gray", "Gaussian Filter", "Image Sharping"],
              ["Image Sharping","Image Big Contour","Warp Colored","Dilated"]]

    stackedImage = stackImages(imageArray,0.75,lables)

    textOCR = pytesseract.image_to_string(imgThres, lang='eng')
    #print("Hasil OCR:  {}".format(textOCR))

    cv2.imshow("Hasil",stackedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def camera():
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("test")
        img_counter=0

        while True:
            red, img = cam.read()
            img=cv2.transpose(img)
            img=cv2.flip(img,flipCode=0)
            rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gaussianFilter = cv2.GaussianBlur(imgGray, (3, 3), 10)
            unsharpMasking = cv2.addWeighted(imgGray, 1 + 1.5, gaussianFilter, -1.5, 0)
            ret, thresh = cv2.threshold(unsharpMasking, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            w,h = 480,640
            # Finding biggest contour
            biggest, maxArea = biggestContour(contours)
            if biggest.size != 0:
                biggest = recorder(biggest)
                cv2.drawContours(rgb, biggest, -1, (0, 255, 0), 10)
                rgb = drawRectangle(rgb, biggest, 5)
                pts1 = np.float32(biggest)
                pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                wrapped = cv2.warpPerspective(rgb, matrix, (w, h))
                wrapped = wrapped[20:wrapped.shape[0] - 20, 20:wrapped.shape[1] - 20]
            else:
                rgb = rgb.copy()

            cv2.imshow("test", rgb)


            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "ARIAL/opencv_frame_{}.jpg".format(img_counter)
                cv2.imwrite(img_name, wrapped)
                print("{} written!".format(img_name))
                img_counter += 1



if __name__ == "__main__":
    image()
    #camera()


