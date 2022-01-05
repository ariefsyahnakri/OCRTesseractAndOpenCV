import cv2 as cv
import numpy as np
from tkinter import *
from PIL import Image

root = Tk()
root.geometry("700x700")
root.configure(bg='black')
root.attributes('-fullscreen', True)
Label(root,text='Tesseract',font=('Times New Roman',30,'bold'),bg='black',fg='red').pack()
f1 = LabelFrame(root,bg='red')
f1.pack()
L1 = Label(f1,bg='red')
L1.pack

root.mainloop()
#while True:
    #root.update
