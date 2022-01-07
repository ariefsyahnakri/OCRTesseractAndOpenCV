from tkinter import*
from PIL import Image,ImageTk
import cv2
from tkinter import filedialog

win = Tk()
win.geometry("600x600+200+30")
win.resizable(False, False)
win.configure(bg ='#1b407a')
w = 400
h = 300

color = "#581845"
frame_1 = Frame(win,width = 600,height =320,bg = color).place(x=0,y=0)
frame_2 = Frame(win,width = 600,height =320,bg = color).place(x=0,y=350)

v = Label(frame_1, width=w, height=h)
v.place(x=10, y=10)
cap = cv2.VideoCapture(0)


def Save():
    file = filedialog.asksaveasfilename(filetypes=[("PNG", ".png")])
    image = Image.fromarray(rgb)
    image.save(file+'.png')
    print(file)

def take_copy(im):
    la = Label(frame_2, width=w-100, height=h-100)
    la.place(x=10, y=370)
    copy = im.copy()
    copy = cv2.resize(copy, (w-100, h-100))
    rgb = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(copy)
    imgtk = ImageTk.PhotoImage(image)
    la.configure(image=imgtk)
    la.image = imgtk
    save = Button(win,text = "save",command=lambda : Save())
    save.place(x=450,y=500)



def select_img():
    global rgb
    _, img = cap.read()
    img = cv2.resize(img, (w, h))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image)
    v.configure(image=imgtk)
    v.image = imgtk
    v.after(10, select_img)




select_img()
snap = Button(win, text="capture", command=lambda: take_copy(rgb))
snap.place(x=450, y=150, width=60, height=50)

win.mainloop()