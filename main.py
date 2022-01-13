import pytesseract
from tkinter import*
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2 as cv
import numpy as np

tesseractFile = "C:\Program Files\Tesseract-OCR\Tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseractFile

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


def select_img(img):
    image = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image)
    v.configure(image=imgtk)
    v.image = imgtk
    v.after(10, select_img)


def take_copy(im):
    la = Label(frame_2, width=w-100, height=h-100)
    la.place(x=10, y=h-10)
    copy = im.copy()
    img = cv.resize(copy, (w-100, h-100))
    image = Image.fromarray(copy)
    imgtk = ImageTk.PhotoImage(image)
    la.configure(image=imgtk)
    la.image = imgtk
    save = Button(win,text = "save",command=lambda : Save(img))
    save.place(x=450,y=500)


def Save(img):
    file = filedialog.asksaveasfilename(filetypes=[("PNG", ".png")])
    image = Image.fromarray(img)
    image.save(file+'.png')
    imgToText(image,)
    print(file)


def main():
    global w, h, v, frame_1, frame_2, win

    ## Settings untuk GUI
    win = Tk()
    widthScreen = win.winfo_screenwidth()
    heightScreen = win.winfo_screenheight()
    win.geometry("%dx%d" % (widthScreen, heightScreen))
    win.resizable(False, False)
    win.configure(bg='#1b407a')
    w = 400
    h = int(heightScreen/2)
    color = "#581845"
    frame_1 = Frame(win, width=widthScreen, height=heightScreen/2, bg=color).place(x=0, y=0)
    frame_2 = Frame(win, width=widthScreen, height=heightScreen/2, bg=color).place(x=0, y=350)
    v = Label(frame_1, width=w, height=h)
    v.place(x=10, y=10)
    snap = Button(win, text="capture", command=lambda: take_copy(wrapped))
    snap.place(x=450, y=150, width=60, height=50)

    cam = cv.VideoCapture(0)

    while True:
        ## Starting Camera
        check, img = cam.read()
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gaussianFilter = cv.GaussianBlur(imgGray, (3, 3), 10)
        unsharpMasking = cv.addWeighted(imgGray, 1 + 1.5, gaussianFilter, -1.5, 0)
        ret, thresh = cv.threshold(unsharpMasking, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Finding biggest contour
        biggest, maxArea = biggestContour(contours)
        if biggest.size != 0:
            biggest = recorder(biggest)
            cv.drawContours(img, biggest, -1, (0, 255, 0), 10)
            img = drawRectangle(img, biggest, 5)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            matrix = cv.getPerspectiveTransform(pts1, pts2)
            wrapped= cv.warpPerspective(img, matrix, (w, h))
            wrapped = wrapped[20:wrapped.shape[0] - 20, 20:wrapped.shape[1] - 20]
        else:
            img = img.copy()
        select_img(img)
        win.update()


if __name__ == "__main__":
    main()


