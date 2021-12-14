import pytesseract
import cv2 as cv


tesseractFile = "C:\Program Files\Tesseract-OCR\Tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseractFile


def imgToText(img):
    # Function untuk memasukkan path folder image dan akan menghasilkan text
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

    # Tesseract process
    # text = pytesseract.image_to_string(thresh, lang='eng')
    # return text

def main():
    cam = cv.VideoCapture(0)

    cv.namedWindow("Text Capture")

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
            img_name = "gambar_.jpg"
            cv.imwrite(img_name, frame)
            print("{} written.".format(img_name))
            text = imgToText(img_name)
            print(text)
    cam.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()


