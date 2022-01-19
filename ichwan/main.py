import cv2 as cv
import imutils
import numpy as np
from keras.models import load_model
import h5py




model = load_model('dataset.h5')

#labelNames = "0123456789"
#labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#labelNames = [l for l in labelNames]
labelNames = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i=1
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts,boundingBoxes),key = lambda b:b[1][i], reverse=reverse))
    return (cnts,boundingBoxes)

def get_letters(img):
    letters = []

    image = cv.imread(img)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV)
    dilated = cv.dilate(thresh, None, iterations=2)


    cnts = cv.findContours(dilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method='left-to-right')[0]

    for c in cnts:
        if cv.contourArea(c) > 10:
            (x,y,w,h) = cv.boundingRect(c)
            cv.rectangle(image, (x,y) , (x+w, y+h) , (0, 255, 0) , 2)
            roi = gray[y:y + h, x:x + w]
            thresh = cv.threshold(roi, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
            thresh = cv.resize(thresh, (32, 32), interpolation=cv.INTER_CUBIC)
            thresh = thresh.astype("float32") / 255.0
            thresh = np.expand_dims(thresh, axis=-1)
            thresh = cv.resize(thresh, (28,28))
            thresh = np.reshape(thresh, (1,28,28,1))

            # prediction OCR
            preds = model.predict(thresh)
            i = np.argmax(preds)
            label = labelNames[i]
            #prob = preds[i]
            #print("[INFO] {} - {:.2f}%".format(label, prob * 100))

            cv.putText(image, label, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            [x] = label
            letters.append(x)

    return letters, image

def get_word(letter):
    word = "".join(letter)
    return word

def main():
    imagePath = 'WORLD.png'
    fn = imagePath.split(".")
    letter, image = get_letters(imagePath)
    word = get_word(letter)
    print('hasil dari ocr = {}'.format(word))

    cv.imshow('hasil',image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite(fn[0]+'hasil.png', image)



if __name__ == '__main__':
    main()
