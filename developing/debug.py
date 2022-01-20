from datetime import date,datetime

now = datetime.now()
text = now.strftime("Gambar %d/%m/%Y %H:%M:%S").replace(":","-")+ ".jpg"


print(text)