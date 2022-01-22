import cv2 as cv 


def main():
    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 2464)
    cam.set(cv.CAP_PROP_FRAME_WIDTH,3280)
    cv.namedWindow("Coba")


    while True: 
        ret, img = cam.read()


        cv.imshow('Coba', img)
        k = cv.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        

if __name__ == "__main__":
    main()
