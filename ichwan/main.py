import cv2 as cv
import imutils
import numpy as np
import tensorflow as tf
from tf.keras.models import load_model
import tkinter as tk
from tkinter import*
from PIL import Image, ImageTk
from tkinter import filedialog
from datetime import date,datetime

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G'
    ,7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q'
    ,17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'
    ,26:'0' ,27 :'1', 28:'2', 29:'3', 30 :'4', 31:'5', 32:'6'
    ,33:'7' ,34:'8', 35:'9'
            }
model = load_model('model_v2.h5')



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

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "bottom-to-top" or method == "bottom-to-top":
        i=1
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts,boundingBoxes),key = lambda b:b[1][i], reverse=reverse))
    return (cnts,boundingBoxes)

def get_letters(img):
    letters = []
    image = cv.imread(img)
    scale_percent = 60  # percent of original size
    width = int(768)
    height = int(1024)
    dim = (width, height)
    # resize image
    image = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # blurred = cv.GaussianBlur(gray, (5,5), 0)
    # thresh = cv.adaptiveThreshold(blurred, 250, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 23, 10)
    # dilated = cv.dilate(thresh, None, iterations=1)
    # cv.imwrite('thresh.jpg',image)
    # cv.imshow("tresh", thresh)
    # cv.waitKey(0)
    # # #
    # # apply morphology close to form rows
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (200, 1))
    # morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    # rows_img = image.copy()
    # boxes_img = image.copy()
    # rowboxes = []
    # rowcontours = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # rowcontours = rowcontours[0] if len(rowcontours) == 2 else rowcontours[1]
    # index = 1
    # for rowcntr in rowcontours:
    #     xr, yr, wr, hr = cv.boundingRect(rowcntr)
    #     cv.rectangle(rows_img, (xr, yr), (xr + wr, yr + hr), (0, 0, 255), 1)
    #     rowboxes.append((xr, yr, wr, hr))
    #
    # # sort rowboxes on y coordinate
    # def takeSecond(elem):
    #     return elem[1]
    #
    # rowboxes.sort(key=takeSecond)
    #
    # # loop over each row
    # for rowbox in rowboxes:
    #     # crop the image for a given row
    #     xr = rowbox[0]
    #     yr = rowbox[1]
    #     wr = rowbox[2]
    #     hr = rowbox[3]
    #     row = thresh[yr:yr + hr, xr:xr + wr]
    #     bboxes = []
    #     # find contours of each character in the row
    #     contours = cv.findContours(row, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #     contours = contours[0] if len(contours) == 2 else contours[1]
    #     for cntr in contours:
    #         x, y, w, h = cv.boundingRect(cntr)
    #         bboxes.append((x + xr, y + yr, w, h))
    #
    #     # sort bboxes on x coordinate
    #     def takeFirst(elem):
    #         return elem[0]
    #
    #     bboxes.sort(key=takeFirst)
    #     # draw sorted boxes
    #     for box in bboxes:
    #         xb = box[0]
    #         yb = box[1]
    #         wb = box[2]
    #         hb = box[3]
    #         cv.rectangle(boxes_img, (xb, yb), (xb + wb, yb + hb), (0, 0, 255), 1)
    #         cv.putText(boxes_img, str(index), (xb, yb), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0, 255, 0), 1)
    #         index = index + 1
    # # cv.imshow("thresh", thresh)
    # # cv.imshow("morph", morph)
    # # cv.imshow("rows_img", rows_img)
    # cv.imshow("boxes_img", boxes_img)
    # cv.waitKey(0)

    cnts = cv.findContours( image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method='top-to-bottom')[0]

    for c in cnts:
        if cv.contourArea(c) > 80:
            (x,y,w,h) = cv.boundingRect(c)
            cv.rectangle(image, (x,y) , (x+w, y+h) , (0, 255, 0) , 2)
            roi = image[y:y + h, x:x + w]
            thresh = cv.threshold(roi, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
            thresh = cv.resize(thresh, (32, 32), interpolation=cv.INTER_CUBIC)
            thresh = thresh.astype("float32") / 255.0
            thresh = np.expand_dims(thresh, axis=-1)
            thresh = cv.resize(thresh, (28,28))
            thresh = np.reshape(thresh, (1,28,28,1))

            # prediction OCR
            pred = word_dict[np.argmax(model.predict(thresh))]
            label_text = f"{pred},{word_dict * 100:.1f}%"
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(image, label_text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            [x] = pred
            letters.append(x)
            print("",len(letters))
    return letters, image

def get_word(letter):
    word = "".join(letter)
    return word



def main():
    cam = cv.VideoCapture(0)
    cv.namedWindow('Camera')

    motherDir = "HasilOCR"
    i=0

    while True:
        ret, img = cam.read()
        img = cv.transpose(img)
        img = cv.flip(img, flipCode=0)
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(imgGray, (5,5), 0)
        thresh = cv.adaptiveThreshold(blurred, 250, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 23, 10)
        dilated = cv.dilate(thresh, None, iterations=1)
        contours, hierarchy = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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

        cv.imshow('Camera',rgb)
        cv.waitKey(1)
        k = cv.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            imgFile = "{}/UnwrappedDocument{}.jpg".format(motherDir, i)
            imgFileRaw = "{}/WrappedDocument{}.jpg".format(motherDir, i)
            cv.imwrite(imgFile, img)
            cv.imwrite(imgFileRaw, wrapped)
            print("{} written".format(imgFileRaw))
            print("{} written".format(imgFile))
            letter, image = get_letters(imgFile)
            word = get_word(letter)
            print('hasil dari ocr = {}'.format(word))
            i += 1

# def main():
#     letter, image = get_letters('WrappedDocument21.jpg')
#     word = get_word(letter)
#     print('hasil dari ocr = {}'.format(word))
#     cv.imshow('hasil',image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#     cv.imwrite('images.png',image)

if __name__ == '__main__':
    main()
