import cv2
import numpy as np 
import msgpack
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

path = "zdjecia"

gafiles = [f for f in listdir(path) if isfile(join(path, f))]
search_jpg = lambda a, b: [a + '/'+ f for f in listdir(a) if f.endswith(b)]
fextension = '.tif'
myfiles = search_jpg(path, fextension)

pole_array = []
time_array = []
time = 0.0

for i in myfiles:
    img = cv2.imread(i, cv2.IMREAD_UNCHANGED)
    kropla = img[150:500,0:1024]
    gaus_blur = cv2.GaussianBlur(kropla, (5, 5), 0)
    ret3, th3 = cv2.threshold(gaus_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th3 = cv2.cvtColor(th3,cv2.COLOR_GRAY2RGB)
    not_th3 = 255-th3
    closing = cv2.morphologyEx(not_th3, cv2.MORPH_CLOSE, np.ones((51,51),np.uint8))
    pole = np.sum(closing>200)
    pole_array.append(int(pole))
    time_array.append(round(time,2))
    time = time + 0.01

dane = [pole_array,time_array]
dane_serializacja = msgpack.packb(dane)

with open('data.msgpack', 'wb') as file:
    file.write(dane_serializacja)

with open('data.msgpack', 'rb') as file:
    dane_serializacja_plik = file.read()

dane_po_serializacji = msgpack.unpackb(dane_serializacja_plik)

plt.plot(dane_po_serializacji[1], dane_po_serializacji[0])
plt.title('Wykres pola od czasu dla analizowanej kropli')
plt.xlabel('Czas [s]')
plt.ylabel('Pole [pix^2]')
plt.legend()
plt.grid()
plt.show()
