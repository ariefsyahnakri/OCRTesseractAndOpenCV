import cv2 as cv 


def main():
    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv.CAP_PROP_FRAME_WIDTH,1920)

    cv.namedWindow("Coba")
    #cv.resizeWindow("Coba",640,480)
    i = 0
    while True: 
        ret, img = cam.read()


        cv.imshow('Coba', img)
        k = cv.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            imgFile = "nyari resolusi yang gede {}.jpg".format(i)
            #img = cv.resize(img,(3280,2464))
            cv.imwrite(imgFile,img)
            print("{} written".format(imgFile))


        

if __name__ == "__main__":
    main()
