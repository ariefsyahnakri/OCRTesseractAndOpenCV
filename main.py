import pytesseract
from tkinter import*
from tkinter import filedialog
import cv2 as cv
import numpy as np

tesseractFile = "C:\Program Files\Tesseract-OCR\Tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseractFile



def cameraRecord():
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
            cv.imshow("Hasil", result)
        else:
            cv.imshow("Hasil", frame)

        key = cv.waitKey(1)
        if key % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            cam.release()
            break

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
    # Reading image
    img = cv.imread(img)

    # Color2Gray process
    imgGrey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # Unsharp masking process
    gaussianFilter = cv.GaussianBlur(imgGrey, (11,11), 10)
    unsharpMasking = cv.addWeighted(imgGrey, 1+1.5, gaussianFilter, -0.5, 0)

    # Otsu thresholding process
    ret, thresh = cv.threshold(unsharpMasking, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    cv.imshow('Hasil', thresh)
    cv.waitKey(0)

    #Tesseract process
    textOCR = pytesseract.image_to_string(thresh, lang='eng')
    return textOCR

def main():
    ## Settings untuk GUI
    win = Tk()
    win.geometry("600x600+200+30")
    win.resizable(False, False)
    win.configure(bg='#1b407a')
    w = 400
    h = 300
    color = "#581845"
    frame_1 = Frame(win, width=600, height=320, bg=color).place(x=0, y=0)
    frame_2 = Frame(win, width=600, height=320, bg=color).place(x=0, y=350)
    v = Label(frame_1, width=w, height=h)
    v.place(x=10, y=10)

    cam = cv.VideoCapture(0)
    cv.namedWindow("Text Capture")
    counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv.imshow("test", frame)

        k = cv.waitKey(1)
        if k % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break

        if k % 256 == 32:  # SPACE pressed
            img_name = "gambar_{}.jpg".format(counter)
            cv.imwrite(img_name, frame)
            print("{} written.".format(img_name))
            textOCR = imgToText(img_name)
            with open("Output_{}.txt".format(counter), "w+") as text_file:
                print(textOCR, file=text_file)
            counter += 1
    cam.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()


